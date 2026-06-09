import torch
import torch.nn as nn
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

    def __init__(
        self,
        input_dim=140,
        hidden_dim=64,
        dropout=0.35,
        attn_scale=0.1,
        attention_dropout=0.0,
    ):
        super(PLIEncoderBandEdgeAttention, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.attn_scale = float(attn_scale)
        self.attention_dropout = float(attention_dropout)
        self.latest_attention_regularization = None
        self.band_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.GELU(),
            nn.Dropout(self.attention_dropout),
            nn.Linear(16, 5),
            nn.Sigmoid(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(28, 64),
            nn.GELU(),
            nn.Dropout(self.attention_dropout),
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
        self.latest_attention_regularization = (
            (band_weight_flat - 0.5).pow(2).mean()
            + (edge_weight_flat - 0.5).pow(2).mean()
        )
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
            "attention_dropout": self.attention_dropout,
            "attn_reg": float(self.latest_attention_regularization.detach().cpu().item()),
            "after_flatten": tuple(x_flat.shape),
            "h_seq": tuple(h_seq.shape),
            "h_base": tuple(h_base.shape),
        }
        return h_seq, h_base


class TemporalContextTransformerLite(nn.Module):
    """Lightweight temporal context block over the 40 Delta-PLI time tokens."""

    def __init__(
        self,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        ffn_dim=128,
        dropout=0.25,
        max_tokens=40,
        use_pos_embed=True,
        use_attention_pooling=True,
    ):
        super(TemporalContextTransformerLite, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.dropout = float(dropout)
        self.max_tokens = int(max_tokens)
        self.use_pos_embed = bool(use_pos_embed)
        self.use_attention_pooling = bool(use_attention_pooling)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, self.hidden_dim))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_attn": nn.LayerNorm(self.hidden_dim),
                "attn": nn.MultiheadAttention(
                    embed_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                ),
                "drop_attn": nn.Dropout(self.dropout),
                "norm_ffn": nn.LayerNorm(self.hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(self.hidden_dim, self.ffn_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.ffn_dim, self.hidden_dim),
                    nn.Dropout(self.dropout),
                ),
            })
            for _ in range(self.num_layers)
        ])
        self.pool_score = nn.Linear(self.hidden_dim, 1)
        self.latest_shapes = {}
        self.latest_alpha = None

    def forward(self, h_seq):
        # h_seq keeps the real Delta-PLI temporal order: [B, 40, 64].
        if h_seq.dim() != 3:
            raise ValueError(f"TemporalContextTransformerLite expected [B, T, D], got {tuple(h_seq.shape)}")

        batch_size, num_tokens, hidden_dim = h_seq.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"TemporalContextTransformerLite hidden dim mismatch: expected {self.hidden_dim}, got {hidden_dim}")

        x = h_seq
        if self.use_pos_embed:
            x = x + self.pos_embed[:, :num_tokens, :]

        layer_infos = []
        for layer_idx, layer in enumerate(self.layers):
            attn_in = layer["norm_attn"](x)
            attn_out, _ = layer["attn"](attn_in, attn_in, attn_in, need_weights=False)
            x = x + layer["drop_attn"](attn_out)
            ffn_out = layer["ffn"](layer["norm_ffn"](x))
            x = x + ffn_out
            layer_infos.append({
                "layer": layer_idx,
                "output_shape": tuple(x.shape),
            })

        if self.use_attention_pooling:
            alpha = torch.softmax(self.pool_score(x), dim=1)
            h_tct = torch.sum(alpha * x, dim=1)
            alpha_flat = alpha.squeeze(-1)
        else:
            h_tct = x.mean(dim=1)
            alpha_flat = torch.full(
                (batch_size, num_tokens),
                1.0 / max(num_tokens, 1),
                device=h_seq.device,
                dtype=h_seq.dtype,
            )

        self.latest_alpha = alpha_flat.detach()
        alpha_detached = alpha_flat.detach()
        self.latest_shapes = {
            "tct_lite_enabled": True,
            "num_tct_layers": self.num_layers,
            "tct_num_heads": self.num_heads,
            "tct_ffn_dim": self.ffn_dim,
            "tct_dropout": self.dropout,
            "temporal_pos_embed": self.use_pos_embed,
            "temporal_attention_pooling": self.use_attention_pooling,
            "h_tct_input": tuple(h_seq.shape),
            "h_tct_sequence": tuple(x.shape),
            "h_tct": tuple(h_tct.shape),
            "temporal_alpha": tuple(alpha_flat.shape),
            "temporal_alpha_mean": float(alpha_detached.mean().cpu().item()),
            "temporal_alpha_std": float(alpha_detached.std(unbiased=False).cpu().item()),
            "temporal_alpha_min": float(alpha_detached.min().cpu().item()),
            "temporal_alpha_max": float(alpha_detached.max().cpu().item()),
            "tct_layers": layer_infos,
        }
        return h_tct, alpha_flat


class TemporalConvLite(nn.Module):
    """Lightweight dilated temporal convolution over PLIEncoder time tokens."""

    def __init__(
        self,
        hidden_dim=64,
        channels=64,
        kernel_size=3,
        dilations=None,
        dropout=0.20,
        pooling="attention",
        residual=True,
    ):
        super(TemporalConvLite, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilations = [1, 2, 4] if dilations is None else [int(d) for d in dilations]
        self.dropout = float(dropout)
        self.pooling = str(pooling)
        self.residual = bool(residual)

        if self.channels != self.hidden_dim:
            raise ValueError("TemporalConvLite first version expects tcn_channels == hidden_dim.")

        blocks = []
        for dilation in self.dilations:
            padding = dilation * (self.kernel_size - 1) // 2
            blocks.append(nn.Sequential(
                nn.Conv1d(
                    self.channels,
                    self.channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    padding=padding,
                ),
                nn.BatchNorm1d(self.channels),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.pool_score = nn.Linear(self.hidden_dim, 1)
        self.latest_alpha = None
        self.latest_shapes = {}

    def forward(self, h_seq):
        # h_seq is token-level temporal context from the PLI encoder: [B, 40, 64].
        if h_seq.dim() != 3:
            raise ValueError(f"TemporalConvLite expected [B, T, D], got {tuple(h_seq.shape)}")

        batch_size, num_tokens, hidden_dim = h_seq.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"TemporalConvLite hidden dim mismatch: expected {self.hidden_dim}, got {hidden_dim}")

        x_input = h_seq.transpose(1, 2)  # [B, 64, 40]
        x = x_input
        layer_infos = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            layer_infos.append({
                "layer": idx,
                "dilation": self.dilations[idx],
                "output_shape": tuple(x.shape),
            })

        x_out = x_input + x if self.residual else x
        h_tcn_seq = x_out.transpose(1, 2)  # [B, 40, 64]

        if self.pooling == "attention":
            alpha = torch.softmax(self.pool_score(h_tcn_seq), dim=1)
            h_tcn = torch.sum(alpha * h_tcn_seq, dim=1)
            alpha_flat = alpha.squeeze(-1)
        else:
            h_tcn = h_tcn_seq.mean(dim=1)
            alpha_flat = torch.full(
                (batch_size, num_tokens),
                1.0 / max(num_tokens, 1),
                device=h_seq.device,
                dtype=h_seq.dtype,
            )

        self.latest_alpha = alpha_flat.detach()
        alpha_detached = alpha_flat.detach()
        self.latest_shapes = {
            "tcn_lite_enabled": True,
            "tcn_channels": self.channels,
            "tcn_kernel_size": self.kernel_size,
            "tcn_dilations": self.dilations,
            "tcn_dropout": self.dropout,
            "tcn_pooling": self.pooling,
            "tcn_residual": self.residual,
            "h_tcn_input": tuple(h_seq.shape),
            "h_tcn_conv_input": tuple(x_input.shape),
            "h_tcn_seq": tuple(h_tcn_seq.shape),
            "h_tcn": tuple(h_tcn.shape),
            "temporal_alpha": tuple(alpha_flat.shape),
            "temporal_alpha_mean": float(alpha_detached.mean().cpu().item()),
            "temporal_alpha_std": float(alpha_detached.std(unbiased=False).cpu().item()),
            "temporal_alpha_min": float(alpha_detached.min().cpu().item()),
            "temporal_alpha_max": float(alpha_detached.max().cpu().item()),
            "tcn_layers": layer_infos,
        }
        return h_tcn, alpha_flat


class ParallelMultiScaleTCNLite(nn.Module):
    """Parallel lightweight TCN branches over ordered Delta-PLI temporal tokens."""

    def __init__(
        self,
        hidden_dim=64,
        dilations=None,
        kernel_size=3,
        dropout=0.20,
        pooling="attention",
        residual=True,
    ):
        super(ParallelMultiScaleTCNLite, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.dilations = [1, 2, 4] if dilations is None else [int(d) for d in dilations]
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.pooling = str(pooling)
        self.residual = bool(residual)

        branches = []
        pool_scores = []
        for dilation in self.dilations:
            padding = dilation * (self.kernel_size - 1) // 2
            branches.append(nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size,
                          dilation=dilation, padding=padding),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ))
            pool_scores.append(nn.Linear(self.hidden_dim, 1))
        self.branches = nn.ModuleList(branches)
        self.branch_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in self.dilations])
        self.pool_scores = nn.ModuleList(pool_scores)
        self.scale_logits = nn.Parameter(torch.zeros(len(self.dilations)))
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.latest_branch_alphas = None
        self.latest_scale_weight = None
        self.latest_shapes = {}

    def forward(self, h_seq):
        # h_seq is [B, 40, 64], preserving the real temporal order of Delta-PLI slices.
        if h_seq.dim() != 3:
            raise ValueError(f"ParallelMultiScaleTCNLite expected [B, T, D], got {tuple(h_seq.shape)}")

        batch_size, num_tokens, hidden_dim = h_seq.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"ParallelMultiScaleTCNLite hidden dim mismatch: expected {self.hidden_dim}, got {hidden_dim}"
            )

        x = h_seq.transpose(1, 2)  # [B, 64, 40]
        branch_outputs = []
        branch_sequences = []
        branch_alphas = []
        branch_infos = []

        for idx, (dilation, branch, norm, pool_score) in enumerate(
            zip(self.dilations, self.branches, self.branch_norms, self.pool_scores)
        ):
            y = branch(x).transpose(1, 2)  # [B, 40, 64]
            y = norm(h_seq + y) if self.residual else norm(y)
            if self.pooling == "attention":
                alpha = torch.softmax(pool_score(y), dim=1)
                h_branch = torch.sum(alpha * y, dim=1)
                alpha_flat = alpha.squeeze(-1)
            else:
                h_branch = y.mean(dim=1)
                alpha_flat = torch.full(
                    (batch_size, num_tokens),
                    1.0 / max(num_tokens, 1),
                    device=h_seq.device,
                    dtype=h_seq.dtype,
                )
            branch_sequences.append(y)
            branch_outputs.append(h_branch)
            branch_alphas.append(alpha_flat)
            branch_infos.append({
                "branch": idx,
                "dilation": dilation,
                "sequence_shape": tuple(y.shape),
                "pooled_shape": tuple(h_branch.shape),
                "alpha_shape": tuple(alpha_flat.shape),
            })

        scale_weight = torch.softmax(self.scale_logits, dim=0)
        h_tcn = torch.zeros_like(branch_outputs[0])
        for idx, h_branch in enumerate(branch_outputs):
            h_tcn = h_tcn + scale_weight[idx] * h_branch
        h_tcn = self.out_norm(h_tcn)

        self.latest_branch_alphas = [alpha.detach() for alpha in branch_alphas]
        self.latest_scale_weight = scale_weight.detach()
        scale_cpu = scale_weight.detach().cpu()
        alpha_stats = {}
        for dilation, alpha in zip(self.dilations, branch_alphas):
            detached = alpha.detach()
            alpha_stats[f"branch_alpha_d{dilation}_mean"] = float(detached.mean().cpu().item())
            alpha_stats[f"branch_alpha_d{dilation}_std"] = float(detached.std(unbiased=False).cpu().item())
            alpha_stats[f"branch_alpha_d{dilation}_min"] = float(detached.min().cpu().item())
            alpha_stats[f"branch_alpha_d{dilation}_max"] = float(detached.max().cpu().item())

        self.latest_shapes = {
            "parallel_multiscale_tcn_enabled": True,
            "parallel_tcn_dilations": self.dilations,
            "parallel_tcn_kernel_size": self.kernel_size,
            "parallel_tcn_channels": self.hidden_dim,
            "parallel_tcn_dropout": self.dropout,
            "parallel_tcn_pooling": self.pooling,
            "parallel_tcn_residual": self.residual,
            "scale_fusion": "softmax",
            "branch_pooling": self.pooling,
            "h_parallel_tcn_input": tuple(h_seq.shape),
            "h_parallel_tcn_conv_input": tuple(x.shape),
            "h_tcn": tuple(h_tcn.shape),
            "scale_weight": tuple(scale_weight.shape),
            "scale_weight_values": [float(v) for v in scale_cpu.tolist()],
            "parallel_tcn_branches": branch_infos,
        }
        for dilation, y, h_branch, alpha in zip(self.dilations, branch_sequences, branch_outputs, branch_alphas):
            self.latest_shapes[f"branch_d{dilation}_seq"] = tuple(y.shape)
            self.latest_shapes[f"h_d{dilation}"] = tuple(h_branch.shape)
            self.latest_shapes[f"branch_alpha_d{dilation}"] = tuple(alpha.shape)
        for idx, dilation in enumerate(self.dilations):
            self.latest_shapes[f"scale_weight_d{dilation}"] = float(scale_cpu[idx].item())
        self.latest_shapes.update(alpha_stats)
        return h_tcn, scale_weight, branch_alphas, branch_outputs


class TemporalBlockGraph(nn.Module):
    """Fixed temporal graph over the 40 Delta-PLI time tokens."""

    def __init__(
        self,
        hidden_dim=64,
        num_tokens=40,
        edge_distances=None,
        dropout=0.20,
        pooling="attention",
        residual=True,
        use_self_loop=True,
    ):
        super(TemporalBlockGraph, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.edge_distances = [1] if edge_distances is None else [int(d) for d in edge_distances]
        self.dropout = float(dropout)
        self.pooling = str(pooling)
        self.residual = bool(residual)
        self.use_self_loop = bool(use_self_loop)

        a = torch.zeros(self.num_tokens, self.num_tokens, dtype=torch.float32)
        for distance in self.edge_distances:
            if distance <= 0:
                continue
            for idx in range(self.num_tokens - distance):
                a[idx, idx + distance] = 1.0
                a[idx + distance, idx] = 1.0
        self.num_edges_without_self_loop = int(a.sum().item())
        if self.use_self_loop:
            a.fill_diagonal_(1.0)
        self.num_edges_with_self_loop = int(a.sum().item())

        degree = a.sum(dim=1)
        degree_inv_sqrt = torch.pow(torch.clamp(degree, min=1e-6), -0.5)
        a_norm = degree_inv_sqrt[:, None] * a * degree_inv_sqrt[None, :]
        self.register_buffer("A_norm", a_norm)
        self.a_norm_density = float((a_norm != 0).float().mean().item())

        self.graph_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.graph_norm = nn.LayerNorm(self.hidden_dim)
        self.graph_act = nn.GELU()
        self.graph_dropout = nn.Dropout(self.dropout)
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.pool_score = nn.Linear(self.hidden_dim, 1)
        self.latest_alpha = None
        self.latest_shapes = {}

    def forward(self, h_seq):
        # h_seq is the ordered sequence of 40 Delta-PLI temporal tokens: [B, 40, 64].
        if h_seq.dim() != 3:
            raise ValueError(f"TemporalBlockGraph expected [B, T, D], got {tuple(h_seq.shape)}")

        batch_size, num_tokens, hidden_dim = h_seq.shape
        if num_tokens != self.num_tokens:
            raise ValueError(f"TemporalBlockGraph token mismatch: expected {self.num_tokens}, got {num_tokens}")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"TemporalBlockGraph hidden dim mismatch: expected {self.hidden_dim}, got {hidden_dim}")

        h_msg = torch.einsum("ij,bjd->bid", self.A_norm, h_seq)
        h_graph_seq = self.graph_proj(h_msg)
        h_graph_seq = self.graph_norm(h_graph_seq)
        h_graph_seq = self.graph_act(h_graph_seq)
        h_graph_seq = self.graph_dropout(h_graph_seq)
        if self.residual:
            h_graph_seq = h_seq + h_graph_seq
        h_graph_seq = self.out_norm(h_graph_seq)

        if self.pooling == "attention":
            alpha = torch.softmax(self.pool_score(h_graph_seq), dim=1)
            h_graph = torch.sum(alpha * h_graph_seq, dim=1)
            alpha_flat = alpha.squeeze(-1)
        else:
            h_graph = h_graph_seq.mean(dim=1)
            alpha_flat = torch.full(
                (batch_size, num_tokens),
                1.0 / max(num_tokens, 1),
                device=h_seq.device,
                dtype=h_seq.dtype,
            )

        self.latest_alpha = alpha_flat.detach()
        alpha_detached = alpha_flat.detach()
        self.latest_shapes = {
            "temporal_block_graph_enabled": True,
            "num_temporal_nodes": self.num_tokens,
            "edge_distances": self.edge_distances,
            "num_edges_without_self_loop": self.num_edges_without_self_loop,
            "num_edges_with_self_loop": self.num_edges_with_self_loop,
            "A_norm": tuple(self.A_norm.shape),
            "A_norm_density": self.a_norm_density,
            "graph_dropout": self.dropout,
            "graph_pooling": self.pooling,
            "graph_residual": self.residual,
            "use_self_loop": self.use_self_loop,
            "use_fixed_temporal_graph": True,
            "h_graph_input": tuple(h_seq.shape),
            "graph_seq": tuple(h_graph_seq.shape),
            "h_graph": tuple(h_graph.shape),
            "graph_alpha": tuple(alpha_flat.shape),
            "temporal_alpha": tuple(alpha_flat.shape),
            "temporal_alpha_mean": float(alpha_detached.mean().cpu().item()),
            "temporal_alpha_std": float(alpha_detached.std(unbiased=False).cpu().item()),
            "temporal_alpha_min": float(alpha_detached.min().cpu().item()),
            "temporal_alpha_max": float(alpha_detached.max().cpu().item()),
        }
        return h_graph, alpha_flat


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
        attn_scale=0.1,
        attention_dropout=0.0,
        use_tct_lite=False,
        tct_gate=0.05,
        num_tct_layers=1,
        tct_num_heads=4,
        tct_ffn_dim=128,
        tct_dropout=0.25,
        temporal_pos_embed=True,
        temporal_attention_pooling=True,
        use_tcn_lite=False,
        tcn_gate=0.05,
        tcn_channels=64,
        tcn_kernel_size=3,
        tcn_dilations=None,
        tcn_dropout=0.20,
        tcn_pooling="attention",
        tcn_residual=True,
        use_parallel_multiscale_tcn=False,
        parallel_tcn_gate=0.10,
        parallel_tcn_dilations=None,
        parallel_tcn_kernel_size=3,
        parallel_tcn_dropout=0.20,
        parallel_tcn_pooling="attention",
        parallel_tcn_residual=True,
        use_temporal_block_graph=False,
        graph_gate=0.10,
        edge_distances=None,
        graph_dropout=0.20,
        graph_pooling="attention",
        graph_residual=True,
        use_self_loop=True,
        use_uncertainty_loss_weighting=True,
        learnable_score_fusion=False,
        init_score_fusion_alpha=0.7,
        use_multi_pli=False,
        multi_pli_branches="delta_task",
        multi_pli_fusion_type="gated_residual",
        fusion_init_delta_weight=0.80,
        aux_residual_scale=0.10,
        shared_pli_encoder=True,
        log_branch_gates=True,
        **kwargs,
    ):
        super(PLIEncoderGRURegOnlyRegressor, self).__init__()
        gru_hidden_dim = gru_hidden_dim or hidden_dim
        self.use_gru = bool(use_gru)
        self.use_tct_lite = bool(use_tct_lite)
        self.use_tcn_lite = bool(use_tcn_lite)
        self.use_parallel_multiscale_tcn = bool(use_parallel_multiscale_tcn)
        self.use_temporal_block_graph = bool(use_temporal_block_graph)
        self.use_multi_pli = bool(use_multi_pli)
        self.multi_pli_branches = str(multi_pli_branches)
        self.multi_pli_fusion_type = str(multi_pli_fusion_type)
        self.aux_residual_scale = float(aux_residual_scale)
        self.shared_pli_encoder = bool(shared_pli_encoder)
        self.log_branch_gates = bool(log_branch_gates)
        self.gru_gate = float(gru_gate)
        self.tct_gate = float(tct_gate)
        self.tcn_gate = float(tcn_gate)
        self.parallel_tcn_gate = float(parallel_tcn_gate)
        self.graph_gate = float(graph_gate)
        self.num_classes = int(max_si_score)

        if use_band_edge_attention:
            self.pli_encoder = PLIEncoderBandEdgeAttention(
                input_dim=140,
                hidden_dim=hidden_dim,
                dropout=dropout,
                attn_scale=attn_scale,
                attention_dropout=attention_dropout,
            )
        else:
            self.pli_encoder = PLIEncoder(input_dim=140, hidden_dim=hidden_dim, dropout=dropout)
        self.gate_task_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.gate_rest_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        init_delta = min(max(float(fusion_init_delta_weight), 1e-4), 1.0 - 1e-4)
        init_delta_logit = math.log(init_delta / (1.0 - init_delta))
        self.pli_fusion_logits = nn.Parameter(
            torch.tensor([init_delta_logit, 0.0], dtype=torch.float32)
        )
        self.fusion_prior_target_task = 1.0 - init_delta
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
        if self.use_tct_lite:
            self.tct_lite = TemporalContextTransformerLite(
                hidden_dim=hidden_dim,
                num_layers=num_tct_layers,
                num_heads=tct_num_heads,
                ffn_dim=tct_ffn_dim,
                dropout=tct_dropout,
                max_tokens=40,
                use_pos_embed=temporal_pos_embed,
                use_attention_pooling=temporal_attention_pooling,
            )
        else:
            self.tct_lite = None
        if self.use_tcn_lite:
            self.tcn_lite = TemporalConvLite(
                hidden_dim=hidden_dim,
                channels=tcn_channels,
                kernel_size=tcn_kernel_size,
                dilations=tcn_dilations,
                dropout=tcn_dropout,
                pooling=tcn_pooling,
                residual=tcn_residual,
            )
        else:
            self.tcn_lite = None
        if self.use_parallel_multiscale_tcn:
            self.parallel_multiscale_tcn = ParallelMultiScaleTCNLite(
                hidden_dim=hidden_dim,
                dilations=parallel_tcn_dilations,
                kernel_size=parallel_tcn_kernel_size,
                dropout=parallel_tcn_dropout,
                pooling=parallel_tcn_pooling,
                residual=parallel_tcn_residual,
            )
        else:
            self.parallel_multiscale_tcn = None
        if self.use_temporal_block_graph:
            self.temporal_block_graph = TemporalBlockGraph(
                hidden_dim=hidden_dim,
                num_tokens=40,
                edge_distances=edge_distances,
                dropout=graph_dropout,
                pooling=graph_pooling,
                residual=graph_residual,
                use_self_loop=use_self_loop,
            )
        else:
            self.temporal_block_graph = None

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
        self.latest_tct_gate = None
        self.latest_tcn_gate = None
        self.latest_parallel_tcn_gate = None
        self.latest_graph_gate = None
        self.latest_temporal_alpha = None
        self.latest_parallel_branch_alphas = None
        self.latest_parallel_scale_weight = None
        self.latest_branch_gates = None
        self.latest_multi_pli_input_shape = None

    def _split_multi_pli_inputs(self, x):
        self.latest_multi_pli_input_shape = tuple(x.shape)
        if not self.use_multi_pli:
            return x, None, None
        if x.dim() != 4:
            raise ValueError(f"Multi-PLI input must be [B, branches, 140, 40], got {tuple(x.shape)}")
        needed = {
            "delta_task": 2,
            "delta_rest": 2,
            "delta_task_rest": 3,
        }.get(self.multi_pli_branches)
        if needed is None:
            raise ValueError(f"Unsupported multi_pli_branches={self.multi_pli_branches}")
        if x.shape[1] < needed:
            raise ValueError(
                f"Multi-PLI branch {self.multi_pli_branches} requires {needed} input branches, "
                f"but got shape {tuple(x.shape)}. Regenerate/read Task/Rest PLI features."
            )
        x_delta = x[:, 0:1, :, :]
        x_task = None
        x_rest = None
        if self.multi_pli_branches == "delta_task":
            x_task = x[:, 1:2, :, :]
        elif self.multi_pli_branches == "delta_rest":
            x_rest = x[:, 1:2, :, :]
        elif self.multi_pli_branches == "delta_task_rest":
            x_task = x[:, 1:2, :, :]
            x_rest = x[:, 2:3, :, :]
        return x_delta, x_task, x_rest

    def extract_pli_branch_features(self, x):
        x_delta, x_task, x_rest = self._split_multi_pli_inputs(x)
        h_delta_seq, h_delta_base = self.pli_encoder(x_delta)
        if not self.use_multi_pli:
            self.latest_branch_gates = None
            return h_delta_seq, h_delta_base

        h_seq = h_delta_seq
        h_base = h_delta_base
        gate_task = torch.sigmoid(self.gate_task_logit)
        gate_rest = torch.sigmoid(self.gate_rest_logit)
        used_task = x_task is not None
        used_rest = x_rest is not None

        if self.multi_pli_fusion_type == "learnable_softmax":
            if self.multi_pli_branches != "delta_task" or x_task is None:
                raise ValueError(
                    "learnable_softmax Multi-PLI fusion currently supports only "
                    "multi_pli_branches='delta_task' with x_task present."
                )
            h_task_seq, h_task_base = self.pli_encoder(x_task)
            fusion_weights = torch.softmax(self.pli_fusion_logits, dim=0)
            w_delta = fusion_weights[0]
            w_task = fusion_weights[1]
            h_seq = w_delta * h_delta_seq + w_task * h_task_seq
            h_base = w_delta * h_delta_base + w_task * h_task_base
            fusion_prior_loss = (w_task - self.fusion_prior_target_task) ** 2
            self.latest_branch_gates = {
                "gate_task": 0.0,
                "gate_rest": 0.0,
                "effective_task_scale": 0.0,
                "effective_rest_scale": 0.0,
                "fusion_w_delta": float(w_delta.detach().cpu().item()),
                "fusion_w_task": float(w_task.detach().cpu().item()),
                "fusion_prior_loss": float(fusion_prior_loss.detach().cpu().item()),
                "aux_residual_scale": self.aux_residual_scale,
                "multi_pli_branches": self.multi_pli_branches,
                "multi_pli_fusion_type": self.multi_pli_fusion_type,
                "shared_pli_encoder": self.shared_pli_encoder,
            }
            return h_seq, h_base

        if used_task:
            h_task_seq, h_task_base = self.pli_encoder(x_task)
            h_seq = h_seq + self.aux_residual_scale * gate_task * h_task_seq
            h_base = h_base + self.aux_residual_scale * gate_task * h_task_base
        if used_rest:
            h_rest_seq, h_rest_base = self.pli_encoder(x_rest)
            h_seq = h_seq + self.aux_residual_scale * gate_rest * h_rest_seq
            h_base = h_base + self.aux_residual_scale * gate_rest * h_rest_base

        self.latest_branch_gates = {
            "gate_task": float(gate_task.detach().cpu().item()),
            "gate_rest": float(gate_rest.detach().cpu().item()),
            "effective_task_scale": float((self.aux_residual_scale * gate_task).detach().cpu().item()) if used_task else 0.0,
            "effective_rest_scale": float((self.aux_residual_scale * gate_rest).detach().cpu().item()) if used_rest else 0.0,
            "aux_residual_scale": self.aux_residual_scale,
            "multi_pli_branches": self.multi_pli_branches,
            "multi_pli_fusion_type": self.multi_pli_fusion_type,
            "shared_pli_encoder": self.shared_pli_encoder,
        }
        return h_seq, h_base

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
        h_seq, h_base = self.extract_pli_branch_features(x)
        h_tct_shape = None
        h_tcn_seq_shape = None
        h_tcn_shape = None
        parallel_branch_shapes = {}
        graph_seq_shape = None
        h_graph_shape = None
        temporal_alpha_shape = None
        if self.use_temporal_block_graph:
            h_graph, graph_alpha = self.temporal_block_graph(h_seq)
            h_fused = h_base + self.graph_gate * h_graph
            h_gru_shape = None
            h_gru_proj_shape = None
            graph_seq_shape = self.temporal_block_graph.latest_shapes.get("graph_seq")
            h_graph_shape = tuple(h_graph.shape)
            temporal_alpha_shape = tuple(graph_alpha.shape)
            fusion_desc = "h_base + graph_gate * h_graph"
            latest_gate = None
            self.latest_tct_gate = None
            self.latest_tcn_gate = None
            self.latest_parallel_tcn_gate = None
            self.latest_graph_gate = self.graph_gate
            self.latest_temporal_alpha = graph_alpha.detach()
            self.latest_parallel_branch_alphas = None
            self.latest_parallel_scale_weight = None
        elif self.use_parallel_multiscale_tcn:
            h_tcn, scale_weight, branch_alphas, _ = self.parallel_multiscale_tcn(h_seq)
            h_fused = h_base + self.parallel_tcn_gate * h_tcn
            h_gru_shape = None
            h_gru_proj_shape = None
            h_tcn_shape = tuple(h_tcn.shape)
            fusion_desc = "h_base + tcn_gate * h_parallel_multiscale_tcn"
            latest_gate = None
            self.latest_tct_gate = None
            self.latest_tcn_gate = None
            self.latest_parallel_tcn_gate = self.parallel_tcn_gate
            self.latest_graph_gate = None
            self.latest_temporal_alpha = None
            self.latest_parallel_branch_alphas = [alpha.detach() for alpha in branch_alphas]
            self.latest_parallel_scale_weight = scale_weight.detach()
            parallel_branch_shapes = dict(self.parallel_multiscale_tcn.latest_shapes)
        elif self.use_tcn_lite:
            h_tcn, temporal_alpha = self.tcn_lite(h_seq)
            h_fused = h_base + self.tcn_gate * h_tcn
            h_gru_shape = None
            h_gru_proj_shape = None
            h_tcn_seq_shape = self.tcn_lite.latest_shapes.get("h_tcn_seq")
            h_tcn_shape = tuple(h_tcn.shape)
            temporal_alpha_shape = tuple(temporal_alpha.shape)
            fusion_desc = "h_base + tcn_gate * h_tcn"
            latest_gate = None
            self.latest_tct_gate = None
            self.latest_tcn_gate = self.tcn_gate
            self.latest_parallel_tcn_gate = None
            self.latest_graph_gate = None
            self.latest_temporal_alpha = temporal_alpha.detach()
            self.latest_parallel_branch_alphas = None
            self.latest_parallel_scale_weight = None
        elif self.use_tct_lite:
            h_tct, temporal_alpha = self.tct_lite(h_seq)
            h_fused = h_base + self.tct_gate * h_tct
            h_gru_shape = None
            h_gru_proj_shape = None
            h_tct_shape = tuple(h_tct.shape)
            temporal_alpha_shape = tuple(temporal_alpha.shape)
            fusion_desc = "h_base + tct_gate * h_tct"
            latest_gate = None
            self.latest_tct_gate = self.tct_gate
            self.latest_tcn_gate = None
            self.latest_parallel_tcn_gate = None
            self.latest_graph_gate = None
            self.latest_temporal_alpha = temporal_alpha.detach()
            self.latest_parallel_branch_alphas = None
            self.latest_parallel_scale_weight = None
        elif self.use_gru:
            gru_out, _ = self.gru(h_seq)
            h_gru = gru_out.mean(dim=1)
            h_gru_proj = self.gru_proj(h_gru)
            h_fused = h_base + self.gru_gate * h_gru_proj
            h_gru_shape = tuple(h_gru.shape)
            h_gru_proj_shape = tuple(h_gru_proj.shape)
            fusion_desc = "h_base + gru_gate * h_gru_proj"
            latest_gate = self.gru_gate
            self.latest_tct_gate = None
            self.latest_tcn_gate = None
            self.latest_parallel_tcn_gate = None
            self.latest_graph_gate = None
            self.latest_temporal_alpha = None
            self.latest_parallel_branch_alphas = None
            self.latest_parallel_scale_weight = None
        else:
            h_fused = h_base
            h_gru_shape = None
            h_gru_proj_shape = None
            fusion_desc = "h_base"
            latest_gate = None
            self.latest_tct_gate = None
            self.latest_tcn_gate = None
            self.latest_parallel_tcn_gate = None
            self.latest_graph_gate = None
            self.latest_temporal_alpha = None
            self.latest_parallel_branch_alphas = None
            self.latest_parallel_scale_weight = None

        self.latest_gru_shapes = dict(self.pli_encoder.latest_shapes)
        if self.use_multi_pli and self.latest_branch_gates is not None:
            self.latest_gru_shapes.update({
                "use_multi_pli": True,
                "multi_pli_input": self.latest_multi_pli_input_shape,
                "multi_pli_branches": self.multi_pli_branches,
                "aux_residual_scale": self.aux_residual_scale,
                **self.latest_branch_gates,
            })
        else:
            self.latest_gru_shapes["use_multi_pli"] = False
        if self.use_tct_lite and self.tct_lite is not None:
            self.latest_gru_shapes.update(self.tct_lite.latest_shapes)
        if self.use_tcn_lite and self.tcn_lite is not None:
            self.latest_gru_shapes.update(self.tcn_lite.latest_shapes)
        if self.use_parallel_multiscale_tcn and self.parallel_multiscale_tcn is not None:
            self.latest_gru_shapes.update(self.parallel_multiscale_tcn.latest_shapes)
        if self.use_temporal_block_graph and self.temporal_block_graph is not None:
            self.latest_gru_shapes.update(self.temporal_block_graph.latest_shapes)
        self.latest_gru_shapes.update({
            "gru_enabled": self.use_gru,
            "tct_lite_enabled": self.use_tct_lite,
            "tcn_lite_enabled": self.use_tcn_lite,
            "parallel_multiscale_tcn_enabled": self.use_parallel_multiscale_tcn,
            "temporal_block_graph_enabled": self.use_temporal_block_graph,
            "h_gru": h_gru_shape,
            "h_gru_proj": h_gru_proj_shape,
            "h_tct": h_tct_shape,
            "h_tcn_seq": h_tcn_seq_shape,
            "h_tcn": h_tcn_shape,
            **parallel_branch_shapes,
            "graph_seq": graph_seq_shape,
            "h_graph": h_graph_shape,
            "temporal_alpha": temporal_alpha_shape,
            "h_fused": tuple(h_fused.shape),
            "h_fused_formula": fusion_desc,
            "tct_gate": self.tct_gate if self.use_tct_lite else None,
            "tcn_gate": self.tcn_gate if self.use_tcn_lite else None,
            "parallel_tcn_gate": self.parallel_tcn_gate if self.use_parallel_multiscale_tcn else None,
            "graph_gate": self.graph_gate if self.use_temporal_block_graph else None,
        })
        self.latest_gru_gate = latest_gate
        return h_fused

    def get_temporal_alpha_stats(self):
        alpha = getattr(self, "latest_temporal_alpha", None)
        if alpha is None:
            return None
        detached = alpha.detach()
        return {
            "temporal_alpha_mean": float(detached.mean().cpu().item()),
            "temporal_alpha_std": float(detached.std(unbiased=False).cpu().item()),
            "temporal_alpha_min": float(detached.min().cpu().item()),
            "temporal_alpha_max": float(detached.max().cpu().item()),
        }

    def get_parallel_tcn_stats(self):
        if not self.use_parallel_multiscale_tcn or self.parallel_multiscale_tcn is None:
            return None
        shapes = getattr(self.parallel_multiscale_tcn, "latest_shapes", None)
        if not shapes:
            return None
        keys = []
        stats = {}
        for dilation in shapes.get("parallel_tcn_dilations", []):
            for suffix in ("mean", "std", "min", "max"):
                key = f"branch_alpha_d{dilation}_{suffix}"
                if key in shapes:
                    stats[key] = shapes[key]
                    keys.append(key)
            scale_key = f"scale_weight_d{dilation}"
            if scale_key in shapes:
                stats[scale_key] = shapes[scale_key]
                keys.append(scale_key)
        return stats

    def get_multi_pli_gate_values(self):
        if not self.use_multi_pli:
            return None
        if self.multi_pli_fusion_type == "learnable_softmax":
            fusion_weights = torch.softmax(self.pli_fusion_logits, dim=0)
            w_delta = fusion_weights[0]
            w_task = fusion_weights[1]
            fusion_prior_loss = (w_task - self.fusion_prior_target_task) ** 2
            return {
                "gate_task": 0.0,
                "gate_rest": 0.0,
                "effective_task_scale": 0.0,
                "effective_rest_scale": 0.0,
                "fusion_w_delta": float(w_delta.detach().cpu().item()),
                "fusion_w_task": float(w_task.detach().cpu().item()),
                "fusion_prior_loss": float(fusion_prior_loss.detach().cpu().item()),
            }
        gate_task = torch.sigmoid(self.gate_task_logit)
        gate_rest = torch.sigmoid(self.gate_rest_logit)
        use_task = self.multi_pli_branches in ("delta_task", "delta_task_rest")
        use_rest = self.multi_pli_branches in ("delta_rest", "delta_task_rest")
        return {
            "gate_task": float(gate_task.detach().cpu().item()) if use_task else 0.0,
            "gate_rest": float(gate_rest.detach().cpu().item()) if use_rest else 0.0,
            "effective_task_scale": float((self.aux_residual_scale * gate_task).detach().cpu().item()) if use_task else 0.0,
            "effective_rest_scale": float((self.aux_residual_scale * gate_rest).detach().cpu().item()) if use_rest else 0.0,
            "fusion_w_delta": 0.0,
            "fusion_w_task": 0.0,
            "fusion_prior_loss": 0.0,
        }

    def fusion_prior_regularization(self):
        if not self.use_multi_pli or self.multi_pli_fusion_type != "learnable_softmax":
            return torch.tensor(0.0, device=next(self.parameters()).device)
        fusion_weights = torch.softmax(self.pli_fusion_logits, dim=0)
        return (fusion_weights[1] - self.fusion_prior_target_task) ** 2

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
