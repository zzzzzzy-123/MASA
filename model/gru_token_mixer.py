import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.tcn_plus import MASA_TCN_Regressor


class UncertaintyLossWeighting(nn.Module):
    """Learn homoscedastic uncertainty weights for Huber and rank losses."""

    def __init__(self):
        super(UncertaintyLossWeighting, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, huber_loss, rank_loss):
        log_vars = torch.clamp(self.log_vars, min=-3.0, max=3.0)
        huber_weight = torch.exp(-log_vars[0])
        rank_weight = torch.exp(-log_vars[1])
        loss = huber_weight * huber_loss + log_vars[0] + rank_weight * rank_loss + log_vars[1]
        return loss

    def current_values(self):
        with torch.no_grad():
            log_vars = torch.clamp(self.log_vars.detach(), min=-3.0, max=3.0)
            weights = torch.exp(-log_vars)
            return {
                "log_var_huber": float(log_vars[0].cpu().item()),
                "log_var_rank": float(log_vars[1].cpu().item()),
                "weight_huber": float(weights[0].cpu().item()),
                "weight_rank": float(weights[1].cpu().item()),
            }


class PLI_GRUTokenMixer_Regressor(nn.Module):
    """PLI-only MASA-TCN backbone with a lightweight GRU token mixer.

    This is the useful architecture kept after the exploratory cleanup:
    PLI -> MASA-TCN sequence features -> GRU temporal mixer -> three output heads.
    When ``gru_gate`` is 0, the model falls back to the MASA-TCN pooled embedding.
    """

    def __init__(
        self,
        hidden_channels=None,
        kernel_sizes=None,
        dropout=0.3,
        max_si_score=30,
        use_channel_attention=True,
        aux_num_classes=2,
        gru_hidden_dim=None,
        gru_num_layers=1,
        gru_bidirectional=False,
        gru_gate=0.05,
        input_channels=1,
    ):
        super(PLI_GRUTokenMixer_Regressor, self).__init__()
        hidden_channels = hidden_channels or [64, 64]
        kernel_sizes = kernel_sizes or [2, 4, 6]
        hidden_dim = hidden_channels[-1]
        gru_hidden_dim = gru_hidden_dim or hidden_dim

        self.gru_gate = float(gru_gate)
        self.pli_backbone = MASA_TCN_Regressor(
            num_channels_list=hidden_channels,
            num_eeg_chan=28,
            freq=5,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            max_si_score=max_si_score,
            use_channel_attention=use_channel_attention,
            aux_num_classes=aux_num_classes,
            input_channels=input_channels,
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=gru_bidirectional,
        )
        gru_out_dim = gru_hidden_dim * (2 if gru_bidirectional else 1)
        self.gru_proj = nn.Linear(gru_out_dim, hidden_dim)

        self.latest_gru_shapes = None
        self.latest_gru_gate = None

    def extract_fused_embedding(self, x):
        h_seq = self.pli_backbone.extract_sequence_features(x)  # [B, T, D]
        h_base = torch.mean(h_seq, dim=1)  # [B, D]

        gru_out, _ = self.gru(h_seq)
        h_gru = gru_out.mean(dim=1)
        h_gru_proj = self.gru_proj(h_gru)

        h_fused = h_base + self.gru_gate * h_gru_proj

        self.latest_gru_shapes = {
            "h_seq": tuple(h_seq.shape),
            "h_base": tuple(h_base.shape),
            "h_gru": tuple(h_gru.shape),
            "h_gru_proj": tuple(h_gru_proj.shape),
            "h_fused": tuple(h_fused.shape),
        }
        self.latest_gru_gate = self.gru_gate

        return h_fused

    def forward(self, x):
        h_fused = self.extract_fused_embedding(x)
        return self.pli_backbone.apply_output_heads(h_fused)


class PLI_GRUTokenMixer_RegOnly_Regressor(PLI_GRUTokenMixer_Regressor):
    """Regression-only PLI + MASA-TCN + GRU model."""

    def __init__(self, *args, use_uncertainty_loss_weighting=False, **kwargs):
        super(PLI_GRUTokenMixer_RegOnly_Regressor, self).__init__(*args, **kwargs)
        self.loss_weighting = (
            UncertaintyLossWeighting() if use_uncertainty_loss_weighting else None
        )

    def forward(self, x):
        h_fused = self.extract_fused_embedding(x)
        direct_score = self.pli_backbone.score_regressor(h_fused)
        ordinal_logits = self.pli_backbone.regressor(h_fused)
        self.latest_gru_shapes.update({
            "direct_score": tuple(direct_score.shape),
            "ordinal_logits": tuple(ordinal_logits.shape),
            "expected_score": tuple(direct_score.shape),
            "regression_only": True,
        })
        return direct_score, ordinal_logits


class PLIEncoder(nn.Module):
    """PLI-oriented encoder for Delta-PLI edge-frequency-time tensors."""

    def __init__(self, input_dim=140, hidden_dim=64, dropout=0.35):
        super(PLIEncoder, self).__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.latest_shapes = {}

    def forward(self, x_pli):
        # Delta-PLI is stored as 5 bands x 28 functional edges over 40 temporal slices.
        if x_pli.dim() != 4 or x_pli.shape[1] != 1 or x_pli.shape[2] != 140:
            raise ValueError(f"PLIEncoder expected [B, 1, 140, 40], got {tuple(x_pli.shape)}")

        batch_size = x_pli.shape[0]
        x_squeezed = x_pli.squeeze(1)                  # [B, 140, 40]
        x_reshaped = x_squeezed.reshape(batch_size, 5, 28, 40)
        x_permuted = x_reshaped.permute(0, 3, 1, 2)    # [B, 40, 5, 28]
        x_flat = x_permuted.reshape(batch_size, 40, 140)

        h_seq = self.token_mlp(x_flat)                 # [B, 40, 64]
        h_base = h_seq.mean(dim=1)                     # [B, 64]

        self.latest_shapes = {
            "x_pli": tuple(x_pli.shape),
            "after_squeeze": tuple(x_squeezed.shape),
            "after_reshape": tuple(x_reshaped.shape),
            "after_permute": tuple(x_permuted.shape),
            "after_flatten": tuple(x_flat.shape),
            "h_seq": tuple(h_seq.shape),
            "h_base": tuple(h_base.shape),
        }
        return h_seq, h_base


class PLIEncoderBandEdgeAttention(PLIEncoder):
    """PLI encoder with lightweight frequency-band and functional-edge attention."""

    def __init__(self, input_dim=140, hidden_dim=64, dropout=0.35, attn_scale=0.1):
        super(PLIEncoderBandEdgeAttention, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.attn_scale = float(attn_scale)
        self.band_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.GELU(),
            nn.Linear(16, 5),
            nn.Sigmoid(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(28, 64),
            nn.GELU(),
            nn.Linear(64, 28),
            nn.Sigmoid(),
        )

    def _stats(self, tensor):
        detached = tensor.detach()
        return {
            "mean": float(detached.mean().cpu().item()),
            "std": float(detached.std(unbiased=False).cpu().item()),
            "min": float(detached.min().cpu().item()),
            "max": float(detached.max().cpu().item()),
        }

    def forward(self, x_pli):
        if x_pli.dim() != 4 or x_pli.shape[1] != 1 or x_pli.shape[2] != 140:
            raise ValueError(f"PLIEncoderBandEdgeAttention expected [B, 1, 140, 40], got {tuple(x_pli.shape)}")

        batch_size = x_pli.shape[0]
        x_squeezed = x_pli.squeeze(1)                  # [B, 140, 40]
        x_reshaped = x_squeezed.reshape(batch_size, 5, 28, 40)
        x_permuted = x_reshaped.permute(0, 3, 1, 2)    # [B, 40, 5, 28]
        x_original = x_permuted

        band_desc = x_original.mean(dim=(1, 3))        # [B, 5]
        band_weight_flat = self.band_mlp(band_desc)    # [B, 5]
        band_weight = band_weight_flat[:, None, :, None]

        edge_desc = x_original.mean(dim=(1, 2))         # [B, 28]
        edge_weight_flat = self.edge_mlp(edge_desc)     # [B, 28]
        edge_weight = edge_weight_flat[:, None, None, :]

        x_attn = x_original * band_weight * edge_weight
        x_weighted = x_original + self.attn_scale * x_attn
        x_flat = x_weighted.reshape(batch_size, 40, 140)

        h_seq = self.token_mlp(x_flat)
        h_base = h_seq.mean(dim=1)

        self.latest_shapes = {
            "x_pli": tuple(x_pli.shape),
            "after_squeeze": tuple(x_squeezed.shape),
            "after_reshape": tuple(x_reshaped.shape),
            "after_permute": tuple(x_permuted.shape),
            "band_edge_attention": True,
            "band_desc": tuple(band_desc.shape),
            "band_weight": tuple(band_weight_flat.shape),
            "band_weight_stats": self._stats(band_weight_flat),
            "edge_desc": tuple(edge_desc.shape),
            "edge_weight": tuple(edge_weight_flat.shape),
            "edge_weight_stats": self._stats(edge_weight_flat),
            "x_after_band_edge_attention": tuple(x_weighted.shape),
            "attn_scale": self.attn_scale,
            "after_flatten": tuple(x_flat.shape),
            "h_seq": tuple(h_seq.shape),
            "h_base": tuple(h_base.shape),
        }
        return h_seq, h_base


class PLIEncoderTemporalBandEdgeAttention(PLIEncoder):
    """PLI encoder with time-dependent band and edge gates.

    The Delta-PLI input is already an edge-frequency-time tensor. This block keeps
    the 40 temporal slices intact and learns a separate band/edge modulation for
    each slice, instead of averaging away the temporal context before attention.
    """

    def __init__(
        self,
        input_dim=140,
        hidden_dim=64,
        dropout=0.35,
        gamma_band=0.2,
        gamma_edge=0.2,
        band_hidden_dim=16,
        edge_hidden_dim=64,
        attention_dropout=0.0,
    ):
        super(PLIEncoderTemporalBandEdgeAttention, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.gamma_band = float(gamma_band)
        self.gamma_edge = float(gamma_edge)
        self.band_hidden_dim = int(band_hidden_dim)
        self.edge_hidden_dim = int(edge_hidden_dim)
        self.attention_dropout = float(attention_dropout)
        self.band_mlp = nn.Sequential(
            nn.Linear(5, self.band_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.attention_dropout),
            nn.Linear(self.band_hidden_dim, 5),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(28, self.edge_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.attention_dropout),
            nn.Linear(self.edge_hidden_dim, 28),
        )
        self.latest_attention_regularization = None

    def _stats(self, tensor):
        detached = tensor.detach()
        return {
            "mean": float(detached.mean().cpu().item()),
            "std": float(detached.std(unbiased=False).cpu().item()),
            "min": float(detached.min().cpu().item()),
            "max": float(detached.max().cpu().item()),
        }

    def forward(self, x_pli):
        if x_pli.dim() != 4 or x_pli.shape[1] != 1 or x_pli.shape[2] != 140:
            raise ValueError(
                f"PLIEncoderTemporalBandEdgeAttention expected [B, 1, 140, 40], got {tuple(x_pli.shape)}"
            )

        batch_size = x_pli.shape[0]
        x_squeezed = x_pli.squeeze(1)                  # [B, 140, 40]
        x_reshaped = x_squeezed.reshape(batch_size, 5, 28, 40)
        x_permuted = x_reshaped.permute(0, 3, 1, 2)    # [B, 40, 5, 28]

        band_desc = x_permuted.mean(dim=3)             # [B, 40, 5]
        band_logits = self.band_mlp(band_desc)         # [B, 40, 5]
        band_prob = F.softmax(band_logits, dim=-1)
        band_gate_flat = band_prob * 5.0
        band_gate_flat = 1.0 + self.gamma_band * (band_gate_flat - 1.0)
        band_gate = band_gate_flat.unsqueeze(-1)        # [B, 40, 5, 1]

        edge_desc = x_permuted.mean(dim=2)              # [B, 40, 28]
        edge_logits = self.edge_mlp(edge_desc)          # [B, 40, 28]
        edge_prob = F.softmax(edge_logits, dim=-1)
        edge_gate_flat = edge_prob * 28.0
        edge_gate_flat = 1.0 + self.gamma_edge * (edge_gate_flat - 1.0)
        edge_gate = edge_gate_flat.unsqueeze(2)         # [B, 40, 1, 28]

        x_attn = x_permuted * band_gate * edge_gate
        x_out = x_permuted + 0.5 * (x_attn - x_permuted)
        self.latest_attention_regularization = (
            (band_gate - 1.0).pow(2).mean()
            + (edge_gate - 1.0).pow(2).mean()
        )
        x_flat = x_out.reshape(batch_size, 40, 140)

        h_seq = self.token_mlp(x_flat)
        h_base = h_seq.mean(dim=1)

        self.latest_shapes = {
            "x_pli": tuple(x_pli.shape),
            "after_squeeze": tuple(x_squeezed.shape),
            "after_reshape": tuple(x_reshaped.shape),
            "after_permute": tuple(x_permuted.shape),
            "temporal_band_edge_attention": True,
            "band_attention_time_dependent": True,
            "edge_attention_time_dependent": True,
            "band_desc": tuple(band_desc.shape),
            "band_logits": tuple(band_logits.shape),
            "band_gate": tuple(band_gate.shape),
            "band_gate_stats": self._stats(band_gate_flat),
            "edge_desc": tuple(edge_desc.shape),
            "edge_logits": tuple(edge_logits.shape),
            "edge_gate": tuple(edge_gate.shape),
            "edge_gate_stats": self._stats(edge_gate_flat),
            "x_out": tuple(x_out.shape),
            "gamma_band": self.gamma_band,
            "gamma_edge": self.gamma_edge,
            "band_mlp_hidden_dim": self.band_hidden_dim,
            "edge_mlp_hidden_dim": self.edge_hidden_dim,
            "attention_dropout": self.attention_dropout,
            "attn_reg": float(self.latest_attention_regularization.detach().cpu().item()),
            "after_flatten": tuple(x_flat.shape),
            "h_seq": tuple(h_seq.shape),
            "h_base": tuple(h_base.shape),
        }
        return h_seq, h_base


class PLIEncoderGRURegOnlyRegressor(nn.Module):
    """Delta-PLI -> PLIEncoder -> GRU token mixer -> direct/ordinal regression heads."""

    def __init__(
        self,
        hidden_dim=64,
        dropout=0.35,
        max_si_score=30,
        gru_hidden_dim=None,
        gru_num_layers=1,
        gru_bidirectional=False,
        gru_gate=0.05,
        use_gru=True,
        use_band_edge_attention=False,
        use_temporal_band_edge_attention=False,
        attn_scale=0.1,
        gamma_band=0.2,
        gamma_edge=0.2,
        band_mlp_hidden_dim=16,
        edge_mlp_hidden_dim=64,
        attention_dropout=0.0,
        use_uncertainty_loss_weighting=True,
        learnable_score_fusion=False,
        init_score_fusion_alpha=0.7,
        **kwargs,
    ):
        super(PLIEncoderGRURegOnlyRegressor, self).__init__()
        gru_hidden_dim = gru_hidden_dim or hidden_dim
        self.use_gru = bool(use_gru)
        self.gru_gate = float(gru_gate)
        self.num_classes = int(max_si_score)

        if use_temporal_band_edge_attention:
            self.pli_encoder = PLIEncoderTemporalBandEdgeAttention(
                input_dim=140,
                hidden_dim=hidden_dim,
                dropout=dropout,
                gamma_band=gamma_band,
                gamma_edge=gamma_edge,
                band_hidden_dim=band_mlp_hidden_dim,
                edge_hidden_dim=edge_mlp_hidden_dim,
                attention_dropout=attention_dropout,
            )
        elif use_band_edge_attention:
            self.pli_encoder = PLIEncoderBandEdgeAttention(
                input_dim=140,
                hidden_dim=hidden_dim,
                dropout=dropout,
                attn_scale=attn_scale,
            )
        else:
            self.pli_encoder = PLIEncoder(input_dim=140, hidden_dim=hidden_dim, dropout=dropout)
        if self.use_gru:
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=gru_hidden_dim,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=0.0,
                bidirectional=gru_bidirectional,
            )
            gru_out_dim = gru_hidden_dim * (2 if gru_bidirectional else 1)
            self.gru_proj = nn.Linear(gru_out_dim, hidden_dim)
        else:
            self.gru = None
            self.gru_proj = None

        self.score_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.ordinal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_classes),
        )
        self.regressor = self.ordinal_head
        self.loss_weighting = (
            UncertaintyLossWeighting() if use_uncertainty_loss_weighting else None
        )
        self.learnable_score_fusion = bool(learnable_score_fusion)
        init_alpha = float(init_score_fusion_alpha)
        init_alpha = min(max(init_alpha, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_alpha / (1.0 - init_alpha))
        self.score_fusion_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32),
            requires_grad=self.learnable_score_fusion,
        )
        self.latest_gru_shapes = None
        self.latest_gru_gate = None

    def get_score_fusion_alpha(self):
        if not self.learnable_score_fusion:
            return None
        return torch.sigmoid(self.score_fusion_logit)

    def get_score_fusion_values(self):
        alpha = self.get_score_fusion_alpha()
        if alpha is None:
            return None
        return {
            "fusion_logit": float(self.score_fusion_logit.detach().cpu().item()),
            "fusion_alpha_direct": float(alpha.detach().cpu().item()),
            "fusion_alpha_ordinal": float((1.0 - alpha).detach().cpu().item()),
        }

    def extract_fused_embedding(self, x):
        h_seq, h_base = self.pli_encoder(x)
        if self.use_gru:
            gru_out, _ = self.gru(h_seq)
            h_gru = gru_out.mean(dim=1)
            h_gru_proj = self.gru_proj(h_gru)
            h_fused = h_base + self.gru_gate * h_gru_proj
            h_gru_shape = tuple(h_gru.shape)
            h_gru_proj_shape = tuple(h_gru_proj.shape)
            fusion_desc = "h_base + gru_gate * h_gru_proj"
            latest_gate = self.gru_gate
        else:
            h_fused = h_base
            h_gru_shape = None
            h_gru_proj_shape = None
            fusion_desc = "h_base"
            latest_gate = None

        self.latest_gru_shapes = dict(self.pli_encoder.latest_shapes)
        self.latest_gru_shapes.update({
            "gru_enabled": self.use_gru,
            "h_gru": h_gru_shape,
            "h_gru_proj": h_gru_proj_shape,
            "h_fused": tuple(h_fused.shape),
            "h_fused_formula": fusion_desc,
        })
        self.latest_gru_gate = latest_gate
        return h_fused

    def attention_regularization(self):
        regularization = getattr(self.pli_encoder, "latest_attention_regularization", None)
        if regularization is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return regularization

    def forward(self, x):
        h_fused = self.extract_fused_embedding(x)
        direct_score = self.score_regressor(h_fused)
        ordinal_logits = self.ordinal_head(h_fused)
        ordinal_score = torch.sum(torch.sigmoid(ordinal_logits), dim=1, keepdim=True)
        alpha = self.get_score_fusion_alpha()
        self.latest_gru_shapes.update({
            "direct_score": tuple(direct_score.shape),
            "ordinal_logits": tuple(ordinal_logits.shape),
            "ordinal_score": tuple(ordinal_score.shape),
            "expected_score": tuple(direct_score.shape),
            "regression_only": True,
            "pli_oriented_encoder": True,
            "learnable_score_fusion": self.learnable_score_fusion,
            "fusion_logit": float(self.score_fusion_logit.detach().cpu().item()),
            "score_fusion_alpha": None if alpha is None else float(alpha.detach().cpu().item()),
            "fusion_alpha_ordinal": None if alpha is None else float((1.0 - alpha).detach().cpu().item()),
        })
        return direct_score, ordinal_logits


class TCTRegrLiteBlock(nn.Module):
    """Regression-oriented temporal context block: LN -> Linear -> BiGRU -> MLP."""

    def __init__(self, hidden_dim=64, dropout=0.2, mlp_ratio=2.0):
        super(TCTRegrLiteBlock, self).__init__()
        mlp_hidden_dim = max(1, int(hidden_dim * float(mlp_ratio)))
        self.mlp_hidden_dim = mlp_hidden_dim
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, h_seq):
        input_shape = tuple(h_seq.shape)
        z_reg = self.proj(self.norm_in(h_seq))
        gru_out, _ = self.gru(z_reg)
        gru_out = self.dropout(gru_out)
        z_reg_1 = h_seq + gru_out
        mlp_out = self.mlp(self.norm_mlp(z_reg_1))
        z_reg_2 = z_reg_1 + mlp_out
        h_reg_context = torch.mean(z_reg_2, dim=1)
        block_info = {
            "input_shape": input_shape,
            "output_shape": tuple(z_reg_2.shape),
            "gru_out_shape": tuple(gru_out.shape),
            "mlp_hidden_dim": self.mlp_hidden_dim,
        }
        return h_reg_context, gru_out, z_reg_2, block_info


class TCTRegrLiteStack(nn.Module):
    """Stack independent TCT-Regr Lite blocks for depth ablation."""

    def __init__(self, hidden_dim=64, dropout=0.2, num_blocks=1, mlp_ratio=2.0):
        super(TCTRegrLiteStack, self).__init__()
        self.num_blocks = int(num_blocks)
        self.blocks = nn.ModuleList([
            TCTRegrLiteBlock(hidden_dim=hidden_dim, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(self.num_blocks)
        ])

    def forward(self, h_seq):
        out = h_seq
        block_infos = []
        last_gru_out = None
        for idx, block in enumerate(self.blocks):
            h_context, gru_out, out, info = block(out)
            info["block_index"] = idx
            block_infos.append(info)
            last_gru_out = gru_out
        h_context = torch.mean(out, dim=1)
        return h_context, last_gru_out, out, block_infos
