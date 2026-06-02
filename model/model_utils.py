from model.tcn_plus import MASA_TCN_Regressor
from model.gru_token_mixer import (
    PLIEncoderGRURegOnlyRegressor,
)


PLI_ENCODER_REG_ONLY_EXPERIMENTS = {
    "Exp_DeltaPLI_PLIEncoder_GRU_RegOnly_LearnableLoss",
    "Exp_DeltaPLI_PLIEncoder_RegOnly_LearnableLoss_NoGRU",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_GRU_RegOnly_LearnableLoss",
    "Exp_DeltaPLI_PLIEncoder_TemporalBandEdgeAttn_GRU_RegOnly_LearnableLoss",
    "Exp_DeltaPLI_PLIEncoder_TemporalBandEdgeAttnLite_GRU_RegOnly_LearnableLoss",
    "Exp_DeltaPLI_PLIEncoder_TemporalBandEdgeAttnLite_GRU_RegOnly_LearnableLoss_lr1e4_wd5e4",
    "Exp_DeltaPLI_PLIEncoder_TemporalBandEdgeAttnLite_GRU_RegOnly_LearnableLoss_lr1e4_wd5e4_LearnableFusion",
}


def get_model(features=None, num_chan=8, thickness=5, dropout=0.3, **kwargs):
    """Build the model used by the training loop.

    Current experiments are regression-only. Auxiliary classification routes are
    intentionally not constructed here.
    """
    hidden_channels = kwargs.get("hidden_channels", [64, 64])
    kernel_sizes = kwargs.get("kernel_sizes", [2, 4, 6])
    max_si_score = kwargs.get("max_si_score", 30)
    experiment_name = kwargs.get("experiment_name", "")
    use_channel_attention = kwargs.get("channel_attention", True)
    input_channels = kwargs.get("input_channels", 1)

    if experiment_name in PLI_ENCODER_REG_ONLY_EXPERIMENTS:
        return PLIEncoderGRURegOnlyRegressor(
            hidden_dim=hidden_channels[-1],
            dropout=dropout,
            max_si_score=max_si_score,
            gru_hidden_dim=kwargs.get("gru_hidden_dim", None),
            gru_num_layers=kwargs.get("gru_num_layers", 1),
            gru_bidirectional=kwargs.get("gru_bidirectional", False),
            gru_gate=kwargs.get("gru_gate", 0.05),
            use_gru=kwargs.get("use_gru", True),
            use_band_edge_attention=kwargs.get("use_band_edge_attention", False),
            use_temporal_band_edge_attention=kwargs.get("use_temporal_band_edge_attention", False),
            attn_scale=kwargs.get("attn_scale", 0.1),
            gamma_band=kwargs.get("gamma_band", 0.2),
            gamma_edge=kwargs.get("gamma_edge", 0.2),
            band_mlp_hidden_dim=kwargs.get("band_mlp_hidden_dim", 16),
            edge_mlp_hidden_dim=kwargs.get("edge_mlp_hidden_dim", 64),
            attention_dropout=kwargs.get("attention_dropout", 0.0),
            use_uncertainty_loss_weighting=kwargs.get("loss_weighting", "") == "homoscedastic_uncertainty",
            learnable_score_fusion=kwargs.get("learnable_score_fusion", False),
            init_score_fusion_alpha=kwargs.get("init_score_fusion_alpha", 0.7),
        )

    return MASA_TCN_Regressor(
        num_channels_list=hidden_channels,
        num_eeg_chan=num_chan,
        freq=thickness,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        max_si_score=max_si_score,
        use_channel_attention=use_channel_attention,
        input_channels=input_channels,
    )
