import argparse
import os
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from base.dataset import DataArranger
from base.experiment_utils import (
    ExperimentLogger,
    StreamToLogger,
    run_single_fold,
    setup_logger,
    summarize_fold_metrics,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_global_seed(seed):
    seed = int(seed)
    print(f"Random seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


EXPERIMENTS = {
    "Exp2034_DeltaPLI_PLIEncoder_RegOnly_LearnableLoss_NoGRU": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "gru_gate": 0.05,
        "learning_rate": 2e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 1e-4,
        "patience": 4,
        "early_stopping": 8,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "best_score_mode": "val_mae",
    },
    "Exp2034_DeltaPLI_PLIEncoder_GRU_RegOnly_LearnableLoss": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "gru_gate": 0.05,
        "learning_rate": 2e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 1e-4,
        "patience": 4,
        "early_stopping": 8,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": True,
        "best_score_mode": "val_mae",
    },
    "Exp_DeltaPLI_PLIEncoder_GRU_RegOnly_LearnableLoss": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "gru_gate": 0.05,
        "learning_rate": 2e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 1e-4,
        "patience": 4,
        "early_stopping": 8,
        "loss_weighting": "homoscedastic_uncertainty",
    },
    "Exp_DeltaPLI_PLIEncoder_RegOnly_LearnableLoss_NoGRU": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "gru_gate": 0.05,
        "learning_rate": 2e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 1e-4,
        "patience": 4,
        "early_stopping": 8,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_GRU_RegOnly_LearnableLoss": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "gru_gate": 0.05,
        "learning_rate": 2e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 1e-4,
        "patience": 4,
        "early_stopping": 8,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_GRU_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "gru_gate": 0.05,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.45,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.30,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.45,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.30,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "num_tct_layers": 1,
        "tct_num_heads": 4,
        "tct_ffn_dim": 128,
        "tct_dropout": 0.25,
        "tct_gate": 0.05,
        "temporal_pos_embed": True,
        "temporal_attention_pooling": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_GRU_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1_Drop035": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.20,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "num_tct_layers": 1,
        "tct_num_heads": 4,
        "tct_ffn_dim": 128,
        "tct_dropout": 0.15,
        "tct_gate": 0.05,
        "temporal_pos_embed": True,
        "temporal_attention_pooling": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1_Drop035_Gate010": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.20,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "num_tct_layers": 1,
        "tct_num_heads": 4,
        "tct_ffn_dim": 128,
        "tct_dropout": 0.15,
        "tct_gate": 0.10,
        "temporal_pos_embed": True,
        "temporal_attention_pooling": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1_Drop035_Gate010_lr5e5": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 5e-5,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 4,
        "early_stopping": 6,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.20,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "num_tct_layers": 1,
        "tct_num_heads": 4,
        "tct_ffn_dim": 128,
        "tct_dropout": 0.15,
        "tct_gate": 0.10,
        "temporal_pos_embed": True,
        "temporal_attention_pooling": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_AntiOverfit_Drop035": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.20,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "num_tct_layers": 1,
        "tct_num_heads": 4,
        "tct_ffn_dim": 128,
        "tct_dropout": 0.15,
        "tct_gate": 0.05,
        "temporal_pos_embed": True,
        "temporal_attention_pooling": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_AntiOverfit_Drop035_Gate010": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.20,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "num_tct_layers": 1,
        "tct_num_heads": 4,
        "tct_ffn_dim": 128,
        "tct_dropout": 0.15,
        "tct_gate": 0.10,
        "temporal_pos_embed": True,
        "temporal_attention_pooling": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_AntiOverfit_Drop035_Gate010_lr5e5": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 5e-5,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 4,
        "early_stopping": 6,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.20,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "num_tct_layers": 1,
        "tct_num_heads": 4,
        "tct_ffn_dim": 128,
        "tct_dropout": 0.15,
        "tct_gate": 0.10,
        "temporal_pos_embed": True,
        "temporal_attention_pooling": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_Layer2": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.20,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "num_tct_layers": 2,
        "tct_num_heads": 4,
        "tct_ffn_dim": 128,
        "tct_dropout": 0.15,
        "tct_gate": 0.05,
        "temporal_pos_embed": True,
        "temporal_attention_pooling": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_Heads8": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.20,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "num_tct_layers": 1,
        "tct_num_heads": 8,
        "tct_ffn_dim": 128,
        "tct_dropout": 0.15,
        "tct_gate": 0.05,
        "temporal_pos_embed": True,
        "temporal_attention_pooling": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_Layer2_Heads8": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.20,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "num_tct_layers": 2,
        "tct_num_heads": 8,
        "tct_ffn_dim": 128,
        "tct_dropout": 0.15,
        "tct_gate": 0.05,
        "temporal_pos_embed": True,
        "temporal_attention_pooling": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.45,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": False,
        "use_tcn_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.30,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "tcn_gate": 0.05,
        "tcn_channels": 64,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4],
        "tcn_dropout": 0.20,
        "tcn_pooling": "attention",
        "tcn_residual": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_GRU_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.45,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": False,
        "use_tcn_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.30,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "tcn_gate": 0.10,
        "tcn_channels": 64,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4],
        "tcn_dropout": 0.20,
        "tcn_pooling": "attention",
        "tcn_residual": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_MultiPLI_DeltaTask_SharedEncoder_GatedResidual_TCNLite_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.45,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": False,
        "use_tcn_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.30,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "tcn_gate": 0.10,
        "tcn_channels": 64,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4],
        "tcn_dropout": 0.20,
        "tcn_pooling": "attention",
        "tcn_residual": True,
        "input_mode": "delta_task",
        "use_multi_pli": True,
        "multi_pli_branches": "delta_task",
        "aux_residual_scale": 0.20,
        "shared_pli_encoder": True,
        "log_branch_gates": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
    },
    "Exp_MultiPLI_DeltaTask_SharedEncoder_LearnableSoftmaxFusion_TCNLite_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.45,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": False,
        "use_tcn_lite": True,
        "use_parallel_multiscale_tcn": False,
        "use_temporal_block_graph": False,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.30,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "tcn_gate": 0.10,
        "tcn_channels": 64,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4],
        "tcn_dropout": 0.20,
        "tcn_pooling": "attention",
        "tcn_residual": True,
        "input_mode": "delta_task",
        "use_multi_pli": True,
        "multi_pli_branches": "delta_task",
        "multi_pli_fusion_type": "learnable_softmax",
        "fusion_init_delta_weight": 0.80,
        "lambda_fusion_prior": 0.001,
        "aux_residual_scale": 0.20,
        "shared_pli_encoder": True,
        "log_branch_gates": True,
        "use_std_ratio_loss": False,
        "based_on": "Exp_MultiPLI_DeltaTask_SharedEncoder_GatedResidual_TCNLite_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_MultiPLI_DeltaRest_SharedEncoder_GatedResidual_TCNLite_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.45,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": False,
        "use_tcn_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.30,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "tcn_gate": 0.10,
        "tcn_channels": 64,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4],
        "tcn_dropout": 0.20,
        "tcn_pooling": "attention",
        "tcn_residual": True,
        "input_mode": "delta_rest",
        "use_multi_pli": True,
        "multi_pli_branches": "delta_rest",
        "aux_residual_scale": 0.10,
        "shared_pli_encoder": True,
        "log_branch_gates": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
    },
    "Exp_MultiPLI_DeltaTaskRest_SharedEncoder_GatedResidual_TCNLite_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.45,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": False,
        "use_tcn_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.30,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "tcn_gate": 0.10,
        "tcn_channels": 64,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4],
        "tcn_dropout": 0.20,
        "tcn_pooling": "attention",
        "tcn_residual": True,
        "input_mode": "delta_task_rest",
        "use_multi_pli": True,
        "multi_pli_branches": "delta_task_rest",
        "aux_residual_scale": 0.10,
        "shared_pli_encoder": True,
        "log_branch_gates": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_Gate010_StdRatio025_Lambda002_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.45,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": False,
        "use_tcn_lite": True,
        "use_parallel_multiscale_tcn": False,
        "use_temporal_block_graph": False,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.30,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "tcn_gate": 0.10,
        "tcn_channels": 64,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4],
        "tcn_dropout": 0.20,
        "tcn_pooling": "attention",
        "tcn_residual": True,
        "use_std_ratio_loss": True,
        "target_std_ratio": 0.25,
        "lambda_std": 0.02,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_ParallelMS_TCN_D124_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.45,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": False,
        "use_tcn_lite": False,
        "use_parallel_multiscale_tcn": True,
        "use_temporal_block_graph": False,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.30,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "tcn_gate": 0.10,
        "parallel_tcn_gate": 0.10,
        "parallel_tcn_dilations": [1, 2, 4],
        "parallel_tcn_kernel_size": 3,
        "parallel_tcn_channels": 64,
        "parallel_tcn_dropout": 0.20,
        "parallel_tcn_pooling": "attention",
        "parallel_tcn_residual": True,
        "scale_fusion": "softmax",
        "branch_pooling": "attention",
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010_Drop040": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.40,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False,
        "use_tct_lite": False,
        "use_tcn_lite": True,
        "use_band_edge_attention": True,
        "attn_scale": 0.1,
        "attention_dropout": 0.25,
        "lambda_attn": 0.01,
        "best_score_mode": "val_mae",
        "tcn_gate": 0.10,
        "tcn_channels": 64,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4],
        "tcn_dropout": 0.15,
        "tcn_pooling": "attention",
        "tcn_residual": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TemporalBlockGraph_D1_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"], "num_chan": 28, "thickness": 5,
        "huber_weight": 0.70, "rank_weight": 0.30,
        "learning_rate": 1e-4, "cnn1d_dropout": 0.45, "weight_decay": 5e-4,
        "patience": 3, "early_stopping": 5, "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False, "use_tct_lite": False, "use_tcn_lite": False,
        "use_temporal_block_graph": True, "use_band_edge_attention": True,
        "attn_scale": 0.1, "attention_dropout": 0.30, "lambda_attn": 0.01,
        "best_score_mode": "val_mae", "graph_gate": 0.10, "edge_distances": [1],
        "graph_dropout": 0.20, "graph_pooling": "attention", "graph_residual": True,
        "use_self_loop": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TemporalBlockGraph_D12_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"], "num_chan": 28, "thickness": 5,
        "huber_weight": 0.70, "rank_weight": 0.30,
        "learning_rate": 1e-4, "cnn1d_dropout": 0.45, "weight_decay": 5e-4,
        "patience": 3, "early_stopping": 5, "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False, "use_tct_lite": False, "use_tcn_lite": False,
        "use_temporal_block_graph": True, "use_band_edge_attention": True,
        "attn_scale": 0.1, "attention_dropout": 0.30, "lambda_attn": 0.01,
        "best_score_mode": "val_mae", "graph_gate": 0.10, "edge_distances": [1, 2],
        "graph_dropout": 0.20, "graph_pooling": "attention", "graph_residual": True,
        "use_self_loop": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TemporalBlockGraph_D124_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"], "num_chan": 28, "thickness": 5,
        "huber_weight": 0.70, "rank_weight": 0.30,
        "learning_rate": 1e-4, "cnn1d_dropout": 0.45, "weight_decay": 5e-4,
        "patience": 3, "early_stopping": 5, "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False, "use_tct_lite": False, "use_tcn_lite": False,
        "use_temporal_block_graph": True, "use_band_edge_attention": True,
        "attn_scale": 0.1, "attention_dropout": 0.30, "lambda_attn": 0.01,
        "best_score_mode": "val_mae", "graph_gate": 0.10, "edge_distances": [1, 2, 4],
        "graph_dropout": 0.20, "graph_pooling": "attention", "graph_residual": True,
        "use_self_loop": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
    },
    "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TemporalBlockGraph_D1248_Gate010_RegOnly_LearnableLoss_AntiOverfit_v1": {
        "features": ["eeg_PLI"], "num_chan": 28, "thickness": 5,
        "huber_weight": 0.70, "rank_weight": 0.30,
        "learning_rate": 1e-4, "cnn1d_dropout": 0.45, "weight_decay": 5e-4,
        "patience": 3, "early_stopping": 5, "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": False, "use_tct_lite": False, "use_tcn_lite": False,
        "use_temporal_block_graph": True, "use_band_edge_attention": True,
        "attn_scale": 0.1, "attention_dropout": 0.30, "lambda_attn": 0.01,
        "best_score_mode": "val_mae", "graph_gate": 0.10, "edge_distances": [1, 2, 4, 8],
        "graph_dropout": 0.20, "graph_pooling": "attention", "graph_residual": True,
        "use_self_loop": True,
        "based_on": "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
    },
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


def parse_int_list(value):
    if isinstance(value, list):
        return value
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="MASA EEG-SI regression")
    parser.add_argument("-dataset_path", "--dataset_path", default=r"C:\Users\云瑾\Desktop\data", type=str)
    parser.add_argument("-save_path", "--save_path", default=r"C:\Users\云瑾\Desktop\MASA", type=str)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-learning_rate", "--learning_rate", "--lr", default=5e-4, type=float)
    parser.add_argument("-min_learning_rate", "--min_learning_rate", default=1e-6, type=float)
    parser.add_argument("-cnn1d_dropout", "--dropout", dest="cnn1d_dropout", default=0.3, type=float)
    parser.add_argument("-patience", "--patience", default=5, type=int)
    parser.add_argument("-early_stopping", "--early_stopping", default=20, type=int)
    parser.add_argument("-weight_decay", "--weight_decay", default=1e-5, type=float)
    parser.add_argument("-factor", default=0.5, type=float)
    parser.add_argument("-model_type", default="masa_tcn", choices=["masa_tcn"])
    parser.add_argument("-channel_attention", default=True, type=str_to_bool)
    parser.add_argument("-huber_weight", default=0.70, type=float)
    parser.add_argument("-rank_weight", default=0.30, type=float)
    parser.add_argument("-rank_min_label_gap", default=0.4, type=float)
    parser.add_argument("-rank_min_label_gap_real", default=None, type=float)
    parser.add_argument("-gru_gate", default=0.05, type=float)
    parser.add_argument(
        "-input_mode",
        "--input_mode",
        default="delta_pli",
        choices=["delta_pli", "delta_task", "delta_rest", "delta_task_rest"],
    )
    parser.add_argument("-gru_num_layers", default=1, type=int)
    parser.add_argument("-gru_bidirectional", default=False, type=str_to_bool)
    parser.add_argument("-use_gru", default=True, type=str_to_bool)
    parser.add_argument("-use_band_edge_attention", default=False, type=str_to_bool)
    parser.add_argument("-attn_scale", "--attn_scale", default=0.1, type=float)
    parser.add_argument("-gamma_band", "--gamma_band", default=0.2, type=float)
    parser.add_argument("-gamma_edge", "--gamma_edge", default=0.2, type=float)
    parser.add_argument("-band_mlp_hidden_dim", "--band_mlp_hidden_dim", default=16, type=int)
    parser.add_argument("-edge_mlp_hidden_dim", "--edge_mlp_hidden_dim", default=64, type=int)
    parser.add_argument("-attention_dropout", "--attention_dropout", default=0.0, type=float)
    parser.add_argument("-lambda_attn", "--lambda_attn", default=0.0, type=float)
    parser.add_argument("-use_std_ratio_loss", "--use_std_ratio_loss", default=False, type=str_to_bool)
    parser.add_argument("-target_std_ratio", "--target_std_ratio", default=0.25, type=float)
    parser.add_argument("-lambda_std", "--lambda_std", default=0.0, type=float)
    parser.add_argument("-use_tct_lite", "--use_tct_lite", default=False, type=str_to_bool)
    parser.add_argument("-tct_gate", "--tct_gate", default=0.05, type=float)
    parser.add_argument("-num_tct_layers", "--num_tct_layers", default=1, type=int)
    parser.add_argument("-tct_num_heads", "--tct_num_heads", default=4, type=int)
    parser.add_argument("-tct_ffn_dim", "--tct_ffn_dim", default=128, type=int)
    parser.add_argument("-tct_dropout", "--tct_dropout", default=0.25, type=float)
    parser.add_argument("-temporal_pos_embed", "--temporal_pos_embed", default=True, type=str_to_bool)
    parser.add_argument("-temporal_attention_pooling", "--temporal_attention_pooling", default=True, type=str_to_bool)
    parser.add_argument("-use_tcn_lite", "--use_tcn_lite", default=False, type=str_to_bool)
    parser.add_argument("-tcn_gate", "--tcn_gate", default=0.05, type=float)
    parser.add_argument("-tcn_channels", "--tcn_channels", default=64, type=int)
    parser.add_argument("-tcn_kernel_size", "--tcn_kernel_size", default=3, type=int)
    parser.add_argument("-tcn_dilations", "--tcn_dilations", default=[1, 2, 4], type=parse_int_list)
    parser.add_argument("-tcn_dropout", "--tcn_dropout", default=0.20, type=float)
    parser.add_argument("-tcn_pooling", "--tcn_pooling", default="attention", choices=["attention", "mean"])
    parser.add_argument("-tcn_residual", "--tcn_residual", default=True, type=str_to_bool)
    parser.add_argument("-use_parallel_multiscale_tcn", "--use_parallel_multiscale_tcn", default=False, type=str_to_bool)
    parser.add_argument("-parallel_tcn_gate", "--parallel_tcn_gate", default=0.10, type=float)
    parser.add_argument("-parallel_tcn_dilations", "--parallel_tcn_dilations", default=[1, 2, 4], type=parse_int_list)
    parser.add_argument("-parallel_tcn_kernel_size", "--parallel_tcn_kernel_size", default=3, type=int)
    parser.add_argument("-parallel_tcn_channels", "--parallel_tcn_channels", default=64, type=int)
    parser.add_argument("-parallel_tcn_dropout", "--parallel_tcn_dropout", default=0.20, type=float)
    parser.add_argument("-parallel_tcn_pooling", "--parallel_tcn_pooling", default="attention", choices=["attention", "mean"])
    parser.add_argument("-parallel_tcn_residual", "--parallel_tcn_residual", default=True, type=str_to_bool)
    parser.add_argument("-scale_fusion", "--scale_fusion", default="softmax", choices=["softmax"])
    parser.add_argument("-branch_pooling", "--branch_pooling", default="attention", choices=["attention", "mean"])
    parser.add_argument("-use_temporal_block_graph", "--use_temporal_block_graph", default=False, type=str_to_bool)
    parser.add_argument("-graph_gate", "--graph_gate", default=0.10, type=float)
    parser.add_argument("-edge_distances", "--edge_distances", default=[1], type=parse_int_list)
    parser.add_argument("-graph_dropout", "--graph_dropout", default=0.20, type=float)
    parser.add_argument("-graph_pooling", "--graph_pooling", default="attention", choices=["attention", "mean"])
    parser.add_argument("-graph_residual", "--graph_residual", default=True, type=str_to_bool)
    parser.add_argument("-use_self_loop", "--use_self_loop", default=True, type=str_to_bool)
    parser.add_argument("-learnable_score_fusion", "--learnable_score_fusion", default=False, type=str_to_bool)
    parser.add_argument("-init_score_fusion_alpha", "--init_score_fusion_alpha", default=0.7, type=float)
    parser.add_argument("-use_multi_pli", "--use_multi_pli", default=False, type=str_to_bool)
    parser.add_argument(
        "-multi_pli_branches",
        "--multi_pli_branches",
        default="delta_task",
        choices=["delta_task", "delta_rest", "delta_task_rest"],
    )
    parser.add_argument(
        "-multi_pli_fusion_type",
        "--multi_pli_fusion_type",
        default="gated_residual",
        choices=["gated_residual", "learnable_softmax"],
    )
    parser.add_argument("-fusion_init_delta_weight", "--fusion_init_delta_weight", default=0.80, type=float)
    parser.add_argument("-lambda_fusion_prior", "--lambda_fusion_prior", default=0.0, type=float)
    parser.add_argument("-aux_residual_scale", "--aux_residual_scale", default=0.10, type=float)
    parser.add_argument("-shared_pli_encoder", "--shared_pli_encoder", default=True, type=str_to_bool)
    parser.add_argument("-log_branch_gates", "--log_branch_gates", default=True, type=str_to_bool)
    parser.add_argument("-best_score_mode", "--best_score_mode", default="default", choices=["default", "corr_v2", "val_mae"])
    parser.add_argument("-run_sanity_check", "--run_sanity_check", default=False, type=str_to_bool)
    parser.add_argument("-seed", "--seed", default=2026, type=int)
    parser.add_argument("-random_seed", "--random_seed", default=None, type=int)
    parser.add_argument("-fold_split_seed", "--fold_split_seed", default=2026, type=int)
    parser.add_argument("-use_fixed_split", "--use_fixed_split", default=True, type=str_to_bool)
    parser.add_argument(
        "-fixed_split_path",
        "--fixed_split_path",
        default=os.path.join("splits", "fixed_5fold_seed2026.json"),
        type=str,
    )
    parser.add_argument(
        "-loss_weighting",
        default="fixed",
        choices=["fixed", "homoscedastic_uncertainty"],
    )
    parser.add_argument(
        "-experiment_name",
        "--experiment_name",
        default="Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010",
        choices=["all_core"] + list(EXPERIMENTS.keys()),
    )
    return parser


def apply_experiment_config(args, exp_name, cfg, defaults):
    args.huber_weight = cfg["huber_weight"]
    args.rank_weight = cfg["rank_weight"]
    args.gru_gate = cfg.get("gru_gate", defaults["gru_gate"])
    args.learning_rate = cfg.get("learning_rate", defaults["learning_rate"])
    args.cnn1d_dropout = cfg.get("cnn1d_dropout", defaults["cnn1d_dropout"])
    args.patience = cfg.get("patience", defaults["patience"])
    args.early_stopping = cfg.get("early_stopping", defaults["early_stopping"])
    args.weight_decay = cfg.get("weight_decay", defaults["weight_decay"])
    args.input_mode = cfg.get("input_mode", args.input_mode)
    args.loss_weighting = cfg.get("loss_weighting", defaults["loss_weighting"])
    args.use_gru = cfg.get("use_gru", defaults["use_gru"])
    args.use_band_edge_attention = cfg.get(
        "use_band_edge_attention",
        defaults["use_band_edge_attention"],
    )
    args.attn_scale = cfg.get("attn_scale", defaults["attn_scale"])
    args.gamma_band = cfg.get("gamma_band", defaults["gamma_band"])
    args.gamma_edge = cfg.get("gamma_edge", defaults["gamma_edge"])
    args.band_mlp_hidden_dim = cfg.get("band_mlp_hidden_dim", defaults["band_mlp_hidden_dim"])
    args.edge_mlp_hidden_dim = cfg.get("edge_mlp_hidden_dim", defaults["edge_mlp_hidden_dim"])
    args.attention_dropout = cfg.get("attention_dropout", defaults["attention_dropout"])
    args.lambda_attn = cfg.get("lambda_attn", defaults["lambda_attn"])
    args.use_std_ratio_loss = cfg.get("use_std_ratio_loss", defaults["use_std_ratio_loss"])
    args.target_std_ratio = cfg.get("target_std_ratio", defaults["target_std_ratio"])
    args.lambda_std = cfg.get("lambda_std", defaults["lambda_std"])
    args.learnable_score_fusion = cfg.get(
        "learnable_score_fusion",
        defaults["learnable_score_fusion"],
    )
    args.init_score_fusion_alpha = cfg.get(
        "init_score_fusion_alpha",
        defaults["init_score_fusion_alpha"],
    )
    args.use_tct_lite = cfg.get("use_tct_lite", defaults["use_tct_lite"])
    args.tct_gate = cfg.get("tct_gate", defaults["tct_gate"])
    args.num_tct_layers = cfg.get("num_tct_layers", defaults["num_tct_layers"])
    args.tct_num_heads = cfg.get("tct_num_heads", defaults["tct_num_heads"])
    args.tct_ffn_dim = cfg.get("tct_ffn_dim", defaults["tct_ffn_dim"])
    args.tct_dropout = cfg.get("tct_dropout", defaults["tct_dropout"])
    args.temporal_pos_embed = cfg.get("temporal_pos_embed", defaults["temporal_pos_embed"])
    args.temporal_attention_pooling = cfg.get(
        "temporal_attention_pooling",
        defaults["temporal_attention_pooling"],
    )
    args.use_tcn_lite = cfg.get("use_tcn_lite", defaults["use_tcn_lite"])
    args.tcn_gate = cfg.get("tcn_gate", defaults["tcn_gate"])
    args.tcn_channels = cfg.get("tcn_channels", defaults["tcn_channels"])
    args.tcn_kernel_size = cfg.get("tcn_kernel_size", defaults["tcn_kernel_size"])
    args.tcn_dilations = cfg.get("tcn_dilations", defaults["tcn_dilations"])
    args.tcn_dropout = cfg.get("tcn_dropout", defaults["tcn_dropout"])
    args.tcn_pooling = cfg.get("tcn_pooling", defaults["tcn_pooling"])
    args.tcn_residual = cfg.get("tcn_residual", defaults["tcn_residual"])
    args.use_parallel_multiscale_tcn = cfg.get(
        "use_parallel_multiscale_tcn",
        defaults["use_parallel_multiscale_tcn"],
    )
    args.parallel_tcn_gate = cfg.get("parallel_tcn_gate", defaults["parallel_tcn_gate"])
    args.parallel_tcn_dilations = cfg.get(
        "parallel_tcn_dilations",
        defaults["parallel_tcn_dilations"],
    )
    args.parallel_tcn_kernel_size = cfg.get(
        "parallel_tcn_kernel_size",
        defaults["parallel_tcn_kernel_size"],
    )
    args.parallel_tcn_channels = cfg.get(
        "parallel_tcn_channels",
        defaults["parallel_tcn_channels"],
    )
    args.parallel_tcn_dropout = cfg.get(
        "parallel_tcn_dropout",
        defaults["parallel_tcn_dropout"],
    )
    args.parallel_tcn_pooling = cfg.get("parallel_tcn_pooling", defaults["parallel_tcn_pooling"])
    args.parallel_tcn_residual = cfg.get("parallel_tcn_residual", defaults["parallel_tcn_residual"])
    args.scale_fusion = cfg.get("scale_fusion", defaults["scale_fusion"])
    args.branch_pooling = cfg.get("branch_pooling", defaults["branch_pooling"])
    args.use_temporal_block_graph = cfg.get(
        "use_temporal_block_graph",
        defaults["use_temporal_block_graph"],
    )
    args.graph_gate = cfg.get("graph_gate", defaults["graph_gate"])
    args.edge_distances = cfg.get("edge_distances", defaults["edge_distances"])
    args.graph_dropout = cfg.get("graph_dropout", defaults["graph_dropout"])
    args.graph_pooling = cfg.get("graph_pooling", defaults["graph_pooling"])
    args.graph_residual = cfg.get("graph_residual", defaults["graph_residual"])
    args.use_self_loop = cfg.get("use_self_loop", defaults["use_self_loop"])
    args.use_multi_pli = cfg.get("use_multi_pli", defaults["use_multi_pli"])
    args.multi_pli_branches = cfg.get("multi_pli_branches", defaults["multi_pli_branches"])
    args.multi_pli_fusion_type = cfg.get("multi_pli_fusion_type", defaults["multi_pli_fusion_type"])
    args.fusion_init_delta_weight = cfg.get("fusion_init_delta_weight", defaults["fusion_init_delta_weight"])
    args.lambda_fusion_prior = cfg.get("lambda_fusion_prior", defaults["lambda_fusion_prior"])
    args.aux_residual_scale = cfg.get("aux_residual_scale", defaults["aux_residual_scale"])
    args.shared_pli_encoder = cfg.get("shared_pli_encoder", defaults["shared_pli_encoder"])
    args.log_branch_gates = cfg.get("log_branch_gates", defaults["log_branch_gates"])
    args.based_on = cfg.get("based_on", defaults.get("based_on", ""))
    args.best_score_mode = cfg.get("best_score_mode", "default")
    args.rank_min_label_gap_real = None
    args.rank_min_label_gap = 0.4

    print(f"Selected experiment_name = {exp_name}")
    print(f"Experiment name = {exp_name}")
    print(f"Fold split seed = {args.fold_split_seed}")
    print(f"Train seed = {args.seed}")
    print(f"Using fixed split = {args.use_fixed_split}")
    print(f"Fixed split path = {args.fixed_split_path}")
    print("mode = regression_only")
    print("PLI-oriented encoder enabled = True")
    print("MASA-TCN enabled = False")
    print(f"GRU Token Mixer enabled = {args.use_gru}")
    print(f"Temporal Context Transformer Lite enabled = {args.use_tct_lite}")
    print(f"TCN-Lite enabled = {args.use_tcn_lite}")
    print(f"Parallel Multi-scale TCN-Lite enabled = {args.use_parallel_multiscale_tcn}")
    print(f"Temporal Block Graph enabled = {args.use_temporal_block_graph}")
    print("Regression-only training = True")
    print("Auxiliary classification heads = removed")
    print("BinaryHardCls enabled = False")
    print("Risk head enabled = False")
    print("Loss = learnable Huber + Rank uncertainty weighting")
    if args.use_tct_lite:
        print(f"Based on = {getattr(args, 'based_on', '')}")
        print("Model structure changed = False")
        print("Only hyperparameters changed = True")
    if args.use_band_edge_attention:
        print("Backbone = PLIEncoderBandEdgeAttention")
        print(f"attn_scale = {args.attn_scale}")
        print(f"Attention dropout = {args.attention_dropout}")
        if args.lambda_attn > 0:
            print("Attention regularization enabled = True")
            print(f"lambda_attn = {args.lambda_attn}")
    else:
        print("Backbone = PLIEncoder")
    if args.use_tct_lite:
        print("Based on = Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_GRU_RegOnly_LearnableLoss_AntiOverfit_v1")
        print("GRU Token Mixer enabled = False")
        print("Temporal Context Transformer Lite enabled = True")
        print(f"num_tct_layers = {args.num_tct_layers}")
        print(f"tct_num_heads = {args.tct_num_heads}")
        print(f"tct_ffn_dim = {args.tct_ffn_dim}")
        print(f"tct_dropout = {args.tct_dropout}")
        print(f"tct_gate = {args.tct_gate}")
        print(f"temporal_pos_embed = {args.temporal_pos_embed}")
        print(f"temporal_attention_pooling = {args.temporal_attention_pooling}")
    if args.use_tcn_lite:
        print(f"Based on = {getattr(args, 'based_on', '')}")
        if args.use_std_ratio_loss:
            print("Model family = Serial TCN-Lite Gate010 + StdRatio loss")
            print("Temporal module changed = False")
            print("Loss changed = True")
            print("Only added std_ratio calibration loss = True")
        else:
            print("Model structure changed = False")
            print("Only hyperparameters changed = True")
        print("GRU Token Mixer enabled = False")
        print("Temporal Context Transformer Lite enabled = False")
        print("Parallel Multi-scale TCN enabled = False")
        print("Temporal Block Graph enabled = False")
        print("TCN-Lite enabled = True")
        print(f"tcn_channels = {args.tcn_channels}")
        print(f"tcn_kernel_size = {args.tcn_kernel_size}")
        print(f"tcn_dilations = {args.tcn_dilations}")
        print(f"tcn_dropout = {args.tcn_dropout}")
        print(f"tcn_gate = {args.tcn_gate}")
        print(f"tcn_pooling = {args.tcn_pooling}")
        print(f"tcn_residual = {args.tcn_residual}")
        if args.use_std_ratio_loss:
            print("Loss = learnable Huber + Rank uncertainty weighting + std_ratio loss")
            print(f"use_std_ratio_loss = {args.use_std_ratio_loss}")
            print(f"target_std_ratio = {args.target_std_ratio}")
            print(f"lambda_std = {args.lambda_std}")
    if args.use_parallel_multiscale_tcn:
        print(f"Based on = {getattr(args, 'based_on', '')}")
        print("Model family = Parallel Multi-scale TCN-Lite")
        print("Ablation factor = serial TCN vs parallel multi-scale TCN")
        print("Temporal module changed only = True")
        print("GRU Token Mixer enabled = False")
        print("Temporal Context Transformer Lite enabled = False")
        print("Serial TCN-Lite enabled = False")
        print("Parallel Multi-scale TCN-Lite enabled = True")
        print(f"parallel_tcn_dilations = {args.parallel_tcn_dilations}")
        print(f"parallel_tcn_kernel_size = {args.parallel_tcn_kernel_size}")
        print(f"parallel_tcn_channels = {args.parallel_tcn_channels}")
        print(f"parallel_tcn_dropout = {args.parallel_tcn_dropout}")
        print(f"tcn_gate = {args.parallel_tcn_gate}")
        print(f"scale_fusion = {args.scale_fusion} learnable weights")
        print(f"branch_pooling = {args.branch_pooling}")
    if args.use_temporal_block_graph:
        print(f"Based on = {getattr(args, 'based_on', '')}")
        print("Model family = Temporal Block Graph")
        print("Ablation factor = edge_distances")
        print("Model structure changed from baseline temporal module only = True")
        print("GRU Token Mixer enabled = False")
        print("Temporal Context Transformer Lite enabled = False")
        print("TCN-Lite enabled = False")
        print("Temporal Block Graph enabled = True")
        print(f"graph_dropout = {args.graph_dropout}")
        print(f"graph_gate = {args.graph_gate}")
        print(f"edge_distances = {args.edge_distances}")
        print("graph_hidden_dim = 64")
        print("graph_num_layers = 1")
        print(f"graph_pooling = {args.graph_pooling}")
        print(f"graph_residual = {args.graph_residual}")
        print(f"use_self_loop = {args.use_self_loop}")
        print("use_fixed_temporal_graph = True")
    if args.use_multi_pli:
        print("Multi-PLI shared-encoder fusion enabled = True")
        if args.multi_pli_fusion_type == "learnable_softmax":
            print("Learnable softmax fusion enabled = True")
        else:
            print("Gated residual fusion enabled = True")
        print(f"multi_pli_branches = {args.multi_pli_branches}")
        print(f"multi_pli_fusion_type = {args.multi_pli_fusion_type}")
        print(f"input_mode = {args.input_mode}")
        print(f"aux_residual_scale = {args.aux_residual_scale}")
        print(f"fusion_init_delta_weight = {args.fusion_init_delta_weight}")
        print(f"lambda_fusion_prior = {args.lambda_fusion_prior}")
        print(f"shared_pli_encoder = {args.shared_pli_encoder}")
        print(f"log_branch_gates = {args.log_branch_gates}")
        print("Delta branch = eeg_PLI.npy")
        if args.multi_pli_branches in ("delta_task", "delta_task_rest"):
            print("Task branch = Delta + Rest baseline")
        if args.multi_pli_branches in ("delta_rest", "delta_task_rest"):
            print("Rest branch = eeg_PLI_base.npy broadcast to 40 temporal slices")
    if args.learnable_score_fusion:
        print("Learnable direct/ordinal fusion enabled = True")
        print("fusion_logit init = 0.8473")
        print("Score fusion learnable = True")
        print("expected_score = alpha * direct_score + (1-alpha) * ordinal_score")
        print(f"init_score_fusion_alpha = {args.init_score_fusion_alpha}")
    print(f"BestScore mode = {args.best_score_mode}")
    if args.best_score_mode == "val_mae":
        print("Checkpoint selection metric = validation MAE")
    print(f"learning_rate = {args.learning_rate}")
    print(f"lr = {args.learning_rate}")
    print(f"weight_decay = {args.weight_decay}")
    print(f"dropout = {args.cnn1d_dropout}")
    print(f"patience = {args.patience}")
    print(f"early_stopping = {args.early_stopping}")
    print(f"gru_gate = {args.gru_gate}")
    print(f"use_std_ratio_loss = {args.use_std_ratio_loss}")
    print(f"target_std_ratio = {args.target_std_ratio}")
    print(f"lambda_std = {args.lambda_std}")
    if args.loss_weighting == "homoscedastic_uncertainty":
        loss_text = "uncertainty-weighted huber/rank"
    else:
        loss_text = f"huber={args.huber_weight}, rank={args.rank_weight}"
    print(
        "Loss weighting: {loss_text}, lr={lr}, dropout={dropout}, "
        "patience={patience}, early_stopping={early_stopping}, "
        "weight_decay={weight_decay}, gru_gate={gru_gate}, input_mode={input_mode}".format(
            loss_text=loss_text,
            lr=args.learning_rate,
            dropout=args.cnn1d_dropout,
            patience=args.patience,
            early_stopping=args.early_stopping,
            weight_decay=args.weight_decay,
            gru_gate=args.gru_gate,
            input_mode=args.input_mode,
        )
    )



if __name__ == "__main__":
    print(f"Current PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    parser = build_parser()
    args = parser.parse_args()
    if args.random_seed is not None:
        args.seed = int(args.random_seed)
    set_global_seed(args.seed)
    print(f"Fold split seed = {args.fold_split_seed}")
    print(f"Train seed = {args.seed}")
    print(f"Using fixed split = {args.use_fixed_split}")
    print(f"Fixed split path = {args.fixed_split_path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    defaults = {
        "learning_rate": args.learning_rate,
        "cnn1d_dropout": args.cnn1d_dropout,
        "patience": args.patience,
        "early_stopping": args.early_stopping,
        "weight_decay": args.weight_decay,
        "gru_gate": args.gru_gate,
        "input_mode": args.input_mode,
        "loss_weighting": args.loss_weighting,
        "use_gru": args.use_gru,
        "use_band_edge_attention": args.use_band_edge_attention,
        "attn_scale": args.attn_scale,
        "gamma_band": args.gamma_band,
        "gamma_edge": args.gamma_edge,
        "band_mlp_hidden_dim": args.band_mlp_hidden_dim,
        "edge_mlp_hidden_dim": args.edge_mlp_hidden_dim,
        "attention_dropout": args.attention_dropout,
        "lambda_attn": args.lambda_attn,
        "use_std_ratio_loss": args.use_std_ratio_loss,
        "target_std_ratio": args.target_std_ratio,
        "lambda_std": args.lambda_std,
        "use_tct_lite": args.use_tct_lite,
        "tct_gate": args.tct_gate,
        "num_tct_layers": args.num_tct_layers,
        "tct_num_heads": args.tct_num_heads,
        "tct_ffn_dim": args.tct_ffn_dim,
        "tct_dropout": args.tct_dropout,
        "temporal_pos_embed": args.temporal_pos_embed,
        "temporal_attention_pooling": args.temporal_attention_pooling,
        "use_tcn_lite": args.use_tcn_lite,
        "tcn_gate": args.tcn_gate,
        "tcn_channels": args.tcn_channels,
        "tcn_kernel_size": args.tcn_kernel_size,
        "tcn_dilations": args.tcn_dilations,
        "tcn_dropout": args.tcn_dropout,
        "tcn_pooling": args.tcn_pooling,
        "tcn_residual": args.tcn_residual,
        "use_parallel_multiscale_tcn": args.use_parallel_multiscale_tcn,
        "parallel_tcn_gate": args.parallel_tcn_gate,
        "parallel_tcn_dilations": args.parallel_tcn_dilations,
        "parallel_tcn_kernel_size": args.parallel_tcn_kernel_size,
        "parallel_tcn_channels": args.parallel_tcn_channels,
        "parallel_tcn_dropout": args.parallel_tcn_dropout,
        "parallel_tcn_pooling": args.parallel_tcn_pooling,
        "parallel_tcn_residual": args.parallel_tcn_residual,
        "scale_fusion": args.scale_fusion,
        "branch_pooling": args.branch_pooling,
        "use_temporal_block_graph": args.use_temporal_block_graph,
        "graph_gate": args.graph_gate,
        "edge_distances": args.edge_distances,
        "graph_dropout": args.graph_dropout,
        "graph_pooling": args.graph_pooling,
        "graph_residual": args.graph_residual,
        "use_self_loop": args.use_self_loop,
        "based_on": getattr(args, "based_on", ""),
        "learnable_score_fusion": args.learnable_score_fusion,
        "init_score_fusion_alpha": args.init_score_fusion_alpha,
        "best_score_mode": args.best_score_mode,
        "run_sanity_check": args.run_sanity_check,
        "use_multi_pli": args.use_multi_pli,
        "multi_pli_branches": args.multi_pli_branches,
        "multi_pli_fusion_type": args.multi_pli_fusion_type,
        "fusion_init_delta_weight": args.fusion_init_delta_weight,
        "lambda_fusion_prior": args.lambda_fusion_prior,
        "aux_residual_scale": args.aux_residual_scale,
        "shared_pli_encoder": args.shared_pli_encoder,
        "log_branch_gates": args.log_branch_gates,
    }

    args_dict = vars(args)
    args_dict.update({
        "device": device,
        "criterion": nn.MSELoss(),
        "verbose": True,
        "milestone": [],
        "load_best_at_each_epoch": True,
        "emotion": "SI",
        "metrics": ["mae", "rmse", "r2", "pcc", "ccc"],
        "save_plot": False,
        "scheduler": "plateau",
        "min_epoch": 5,
        "max_epoch": 50,
    })

    if args.experiment_name == "all_core":
        tasks_to_run = list(EXPERIMENTS.items())
    else:
        tasks_to_run = [(args.experiment_name, EXPERIMENTS[args.experiment_name])]

    save_root = str(args.save_path)
    arranger = DataArranger(
        dataset_path=args.dataset_path,
        seed=args.fold_split_seed,
        use_fixed_split=args.use_fixed_split,
        fixed_split_path=args.fixed_split_path,
    )

    for exp_name, cfg in tasks_to_run:
        apply_experiment_config(args, exp_name, cfg, defaults)

        exp_root = os.path.join(save_root, exp_name)
        os.makedirs(exp_root, exist_ok=True)

        logger = setup_logger(exp_root, exp_name)
        sys.stdout = StreamToLogger(logger)
        print(f"Selected experiment_name = {exp_name}")
        apply_experiment_config(args, exp_name, cfg, defaults)

        print("\n" + "=" * 60)
        print(f"Start experiment: {exp_name} | features: {cfg['features']}")
        print("=" * 60)

        fold_metrics = []
        for fold in range(5):
            print(f"\n>>> Fold {fold + 1}/5 <<<")
            fold_data = arranger.get_fold(fold)

            metrics = run_single_fold(
                fold=fold,
                fold_data=fold_data,
                exp_name=exp_name,
                features=cfg["features"],
                num_chan=cfg["num_chan"],
                thickness=cfg["thickness"],
                args=args,
                save_root=save_root,
                device=device,
            )
            fold_metrics.append(metrics)

            print(
                f"Fold {fold + 1} Test -> "
                f"MAE: {metrics['mae']:.4f}, "
                f"RMSE: {metrics['rmse']:.4f}, "
                f"R2: {metrics['r2']:.4f}, "
                f"PCC: {metrics['pcc']:.4f}, "
                f"CCC: {metrics['ccc']:.4f}"
            )

        summarize_fold_metrics(exp_name, fold_metrics)
        if (
            exp_name.startswith("Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_AntiOverfit_v1")
            or exp_name.startswith("Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_AntiOverfit")
            or exp_name.startswith("Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_Layer2")
            or exp_name.startswith("Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCTLite_RegOnly_LearnableLoss_Heads8")
            or exp_name.startswith("Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite")
            or exp_name.startswith("Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_ParallelMS_TCN")
            or exp_name.startswith("Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TemporalBlockGraph")
            or exp_name.startswith("Exp_MultiPLI_")
        ):
            results_dir = os.path.join(save_root, "results")
            os.makedirs(results_dir, exist_ok=True)
            row = {
                "experiment_name": exp_name,
                "based_on": getattr(args, "based_on", ""),
                "fold_split_seed": args.fold_split_seed,
                "train_seed": args.seed,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "dropout": args.cnn1d_dropout,
                "attention_dropout": args.attention_dropout,
                "tct_dropout": args.tct_dropout,
                "tct_gate": args.tct_gate,
                "tcn_dropout": args.tcn_dropout,
                "tcn_gate": args.tcn_gate,
                "tcn_channels": args.tcn_channels,
                "tcn_kernel_size": args.tcn_kernel_size,
                "tcn_dilations": ",".join(str(item) for item in args.tcn_dilations),
                "tcn_pooling": args.tcn_pooling,
                "tcn_residual": args.tcn_residual,
                "temporal_module": "ParallelMS_TCN" if args.use_parallel_multiscale_tcn else (
                    "TCN-Lite" if args.use_tcn_lite else ("TemporalBlockGraph" if args.use_temporal_block_graph else "")
                ),
                "parallel_tcn_dilations": ",".join(str(item) for item in args.parallel_tcn_dilations),
                "parallel_tcn_gate": args.parallel_tcn_gate,
                "parallel_tcn_dropout": args.parallel_tcn_dropout,
                "parallel_tcn_kernel_size": args.parallel_tcn_kernel_size,
                "parallel_tcn_channels": args.parallel_tcn_channels,
                "scale_fusion": args.scale_fusion,
                "branch_pooling": args.branch_pooling,
                "edge_distances": ",".join(str(item) for item in args.edge_distances),
                "graph_dropout": args.graph_dropout,
                "graph_gate": args.graph_gate,
                "graph_pooling": args.graph_pooling,
                "graph_residual": args.graph_residual,
                "use_self_loop": args.use_self_loop,
                "lambda_attn": args.lambda_attn,
                "loss_type": (
                    "learnable_huber_rank_uncertainty_plus_std_ratio"
                    if args.use_std_ratio_loss
                    else "learnable_huber_rank_uncertainty"
                ),
                "use_std_ratio_loss": args.use_std_ratio_loss,
                "target_std_ratio": args.target_std_ratio,
                "lambda_std": args.lambda_std,
                "use_multi_pli": args.use_multi_pli,
                "multi_pli_branches": args.multi_pli_branches,
                "multi_pli_fusion_type": args.multi_pli_fusion_type,
                "fusion_init_delta_weight": args.fusion_init_delta_weight,
                "lambda_fusion_prior": args.lambda_fusion_prior,
                "aux_residual_scale": args.aux_residual_scale,
                "shared_pli_encoder": args.shared_pli_encoder,
            }
            summary_metric_names = [
                "mae", "rmse", "r2", "pcc", "ccc",
                "pred_std", "label_std", "pred_std_ratio",
                "baseline_mae", "model_minus_baseline_mae",
                "temporal_alpha_std",
                "std_loss", "weighted_std_loss",
            ]
            if args.use_multi_pli and args.multi_pli_fusion_type == "gated_residual":
                summary_metric_names.extend([
                    "gate_task", "gate_rest",
                    "effective_task_scale", "effective_rest_scale",
                ])
            if args.use_multi_pli and args.multi_pli_fusion_type == "learnable_softmax":
                summary_metric_names.extend([
                    "fusion_w_delta", "fusion_w_task",
                    "fusion_prior_loss", "weighted_fusion_prior_loss",
                ])
            for metric_name in summary_metric_names:
                values = [m[metric_name] for m in fold_metrics if metric_name in m]
                if values:
                    row[f"{metric_name}_mean"] = float(np.mean(values))
                    row[f"{metric_name}_std"] = float(np.std(values))
                    if metric_name in ("mae", "rmse", "r2", "pcc", "ccc"):
                        row[f"{metric_name.upper()}_mean"] = row[f"{metric_name}_mean"]
                        row[f"{metric_name.upper()}_std"] = row[f"{metric_name}_std"]
            if "baseline_mae_mean" in row:
                row["mean_baseline_MAE_mean"] = row["baseline_mae_mean"]
            if "model_minus_baseline_mae_mean" in row:
                row["model_minus_baseline_MAE_mean"] = row["model_minus_baseline_mae_mean"]
            if args.use_multi_pli and args.multi_pli_fusion_type == "gated_residual":
                for key in ("gate_task", "gate_rest", "effective_task_scale", "effective_rest_scale"):
                    mean_key = f"{key}_mean"
                    if mean_key in row:
                        row[key] = row[mean_key]
            if args.use_multi_pli and args.multi_pli_fusion_type == "learnable_softmax":
                for key in ("fusion_w_delta", "fusion_w_task"):
                    mean_key = f"{key}_mean"
                    std_key = f"{key}_std"
                    if mean_key in row:
                        row[key] = row[mean_key]
                        pretty = "Delta" if key == "fusion_w_delta" else "Task"
                        print(f"Learned {pretty} weight: {row[mean_key]:.6f} ± {row.get(std_key, 0.0):.6f}")
                if "fusion_w_task_mean" in row:
                    print(f"Task branch learned contribution > 0.20: {row['fusion_w_task_mean'] > 0.20}")
                    print(f"Task branch learned contribution < 0.10: {row['fusion_w_task_mean'] < 0.10}")
            graph_meta = {}
            for key in ("num_edges_without_self_loop", "num_edges_with_self_loop", "A_norm_density"):
                values = [m[key] for m in fold_metrics if key in m]
                if values:
                    graph_meta[key] = float(np.mean(values))
            row.update(graph_meta)
            if "temporal_alpha_std_mean" in row:
                row["graph_alpha_std_mean"] = row["temporal_alpha_std_mean"]
            if args.use_parallel_multiscale_tcn:
                for dilation in getattr(args, "parallel_tcn_dilations", []):
                    for metric_name in (f"scale_weight_d{dilation}", f"branch_alpha_d{dilation}_std"):
                        values = [m[metric_name] for m in fold_metrics if metric_name in m]
                        if values:
                            row[f"{metric_name}_mean"] = float(np.mean(values))
                            row[f"{metric_name}_std"] = float(np.std(values))
            summary_path = os.path.join(results_dir, f"{exp_name}_summary.csv")
            pd.DataFrame([row]).to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"TCT-Lite summary saved: {summary_path}")
        sys.stdout = sys.__stdout__

