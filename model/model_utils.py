from model.tcn_plus import MASA_TCN_Regressor
from model.gru_token_mixer import (
    PLIEncoderGRURegOnlyRegressor,
)


PLI_ENCODER_REG_ONLY_EXPERIMENTS = {
    "Exp2034_DeltaPLI_PLIEncoder_RegOnly_LearnableLoss_NoGRU",
    "Exp2034_DeltaPLI_PLIEncoder_GRU_RegOnly_LearnableLoss",
    "Exp_DeltaPLI_PLIEncoder_GRU_RegOnly_LearnableLoss",
    "Exp_DeltaPLI_PLIEncoder_RegOnly_LearnableLoss_NoGRU",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_GRU_RegOnly_LearnableLoss",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_GRU_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1_Drop035",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1_Drop035_Gate010",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1_Drop035_Gate010_lr5e5",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_AntiOverfit_Drop035",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_AntiOverfit_Drop035_Gate010",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_AntiOverfit_Drop035_Gate010_lr5e5",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_Layer2",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_Heads8",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_Layer2_Heads8",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_Gate010_StdRatio025_Lambda002_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010_Drop040",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_ParallelMS_TCN_D124_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TemporalBlockGraph_D1_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TemporalBlockGraph_D12_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TemporalBlockGraph_D124_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TemporalBlockGraph_D1248_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_MultiPLI_DeltaTask_SharedEncoder_GatedResidual_TCNLite_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_MultiPLI_DeltaTask_SharedEncoder_LearnableSoftmaxFusion_TCNLite_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_MultiPLI_DeltaRest_SharedEncoder_GatedResidual_TCNLite_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1",
    "Exp_MultiPLI_DeltaTaskRest_SharedEncoder_GatedResidual_TCNLite_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1",
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
            attn_scale=kwargs.get("attn_scale", 0.1),
            attention_dropout=kwargs.get("attention_dropout", 0.0),
            use_tct_lite=kwargs.get("use_tct_lite", False),
            tct_gate=kwargs.get("tct_gate", 0.05),
            num_tct_layers=kwargs.get("num_tct_layers", 1),
            tct_num_heads=kwargs.get("tct_num_heads", 4),
            tct_ffn_dim=kwargs.get("tct_ffn_dim", 128),
            tct_dropout=kwargs.get("tct_dropout", 0.25),
            temporal_pos_embed=kwargs.get("temporal_pos_embed", True),
            temporal_attention_pooling=kwargs.get("temporal_attention_pooling", True),
            use_tcn_lite=kwargs.get("use_tcn_lite", False),
            tcn_gate=kwargs.get("tcn_gate", 0.05),
            tcn_channels=kwargs.get("tcn_channels", 64),
            tcn_kernel_size=kwargs.get("tcn_kernel_size", 3),
            tcn_dilations=kwargs.get("tcn_dilations", [1, 2, 4]),
            tcn_dropout=kwargs.get("tcn_dropout", 0.20),
            tcn_pooling=kwargs.get("tcn_pooling", "attention"),
            tcn_residual=kwargs.get("tcn_residual", True),
            use_parallel_multiscale_tcn=kwargs.get("use_parallel_multiscale_tcn", False),
            parallel_tcn_gate=kwargs.get("parallel_tcn_gate", kwargs.get("tcn_gate", 0.10)),
            parallel_tcn_dilations=kwargs.get("parallel_tcn_dilations", [1, 2, 4]),
            parallel_tcn_kernel_size=kwargs.get("parallel_tcn_kernel_size", 3),
            parallel_tcn_dropout=kwargs.get("parallel_tcn_dropout", kwargs.get("tcn_dropout", 0.20)),
            parallel_tcn_pooling=kwargs.get("parallel_tcn_pooling", "attention"),
            parallel_tcn_residual=kwargs.get("parallel_tcn_residual", True),
            use_temporal_block_graph=kwargs.get("use_temporal_block_graph", False),
            graph_gate=kwargs.get("graph_gate", 0.10),
            edge_distances=kwargs.get("edge_distances", [1]),
            graph_dropout=kwargs.get("graph_dropout", 0.20),
            graph_pooling=kwargs.get("graph_pooling", "attention"),
            graph_residual=kwargs.get("graph_residual", True),
            use_self_loop=kwargs.get("use_self_loop", True),
            use_uncertainty_loss_weighting=kwargs.get("loss_weighting", "") == "homoscedastic_uncertainty",
            learnable_score_fusion=kwargs.get("learnable_score_fusion", False),
            init_score_fusion_alpha=kwargs.get("init_score_fusion_alpha", 0.7),
            use_multi_pli=kwargs.get("use_multi_pli", False),
            multi_pli_branches=kwargs.get("multi_pli_branches", "delta_task"),
            multi_pli_fusion_type=kwargs.get("multi_pli_fusion_type", "gated_residual"),
            fusion_init_delta_weight=kwargs.get("fusion_init_delta_weight", 0.80),
            aux_residual_scale=kwargs.get("aux_residual_scale", 0.10),
            shared_pli_encoder=kwargs.get("shared_pli_encoder", True),
            log_branch_gates=kwargs.get("log_branch_gates", True),
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
