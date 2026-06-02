import argparse
import os
import sys
import random
import numpy as np

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
    "Exp_DeltaPLI_PLIEncoder_TemporalBandEdgeAttn_GRU_RegOnly_LearnableLoss": {
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
        "use_temporal_band_edge_attention": True,
        "gamma_band": 0.2,
        "gamma_edge": 0.2,
    },
    "Exp_DeltaPLI_PLIEncoder_TemporalBandEdgeAttnLite_GRU_RegOnly_LearnableLoss": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "gru_gate": 0.05,
        "learning_rate": 2e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 1e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": True,
        "use_temporal_band_edge_attention": True,
        "gamma_band": 0.05,
        "gamma_edge": 0.05,
        "band_mlp_hidden_dim": 8,
        "edge_mlp_hidden_dim": 32,
        "attention_dropout": 0.2,
        "lambda_attn": 1e-3,
        "best_score_mode": "val_mae",
    },
    "Exp_DeltaPLI_PLIEncoder_TemporalBandEdgeAttnLite_GRU_RegOnly_LearnableLoss_lr1e4_wd5e4": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "gru_gate": 0.05,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": True,
        "use_temporal_band_edge_attention": True,
        "gamma_band": 0.05,
        "gamma_edge": 0.05,
        "band_mlp_hidden_dim": 8,
        "edge_mlp_hidden_dim": 32,
        "attention_dropout": 0.2,
        "lambda_attn": 1e-3,
        "best_score_mode": "val_mae",
    },
    "Exp_DeltaPLI_PLIEncoder_TemporalBandEdgeAttnLite_GRU_RegOnly_LearnableLoss_lr1e4_wd5e4_LearnableFusion": {
        "features": ["eeg_PLI"],
        "num_chan": 28,
        "thickness": 5,
        "huber_weight": 0.70,
        "rank_weight": 0.30,
        "gru_gate": 0.05,
        "learning_rate": 1e-4,
        "cnn1d_dropout": 0.35,
        "weight_decay": 5e-4,
        "patience": 3,
        "early_stopping": 5,
        "loss_weighting": "homoscedastic_uncertainty",
        "use_gru": True,
        "use_temporal_band_edge_attention": True,
        "gamma_band": 0.05,
        "gamma_edge": 0.05,
        "band_mlp_hidden_dim": 8,
        "edge_mlp_hidden_dim": 32,
        "attention_dropout": 0.2,
        "lambda_attn": 1e-3,
        "best_score_mode": "val_mae",
        "learnable_score_fusion": True,
        "init_score_fusion_alpha": 0.7,
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
    parser.add_argument("-dataset_path", default=r"C:\Users\云瑾\Desktop\data", type=str)
    parser.add_argument("-save_path", default=r"C:\Users\云瑾\Desktop\MASA", type=str)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-learning_rate", "--learning_rate", "--lr", default=5e-4, type=float)
    parser.add_argument("-min_learning_rate", "--min_learning_rate", default=1e-6, type=float)
    parser.add_argument("-cnn1d_dropout", default=0.3, type=float)
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
        choices=["delta_pli"],
    )
    parser.add_argument("-gru_num_layers", default=1, type=int)
    parser.add_argument("-gru_bidirectional", default=False, type=str_to_bool)
    parser.add_argument("-use_gru", default=True, type=str_to_bool)
    parser.add_argument("-use_band_edge_attention", default=False, type=str_to_bool)
    parser.add_argument("-use_temporal_band_edge_attention", "--use_temporal_band_edge_attention", default=False, type=str_to_bool)
    parser.add_argument("-attn_scale", "--attn_scale", default=0.1, type=float)
    parser.add_argument("-gamma_band", "--gamma_band", default=0.2, type=float)
    parser.add_argument("-gamma_edge", "--gamma_edge", default=0.2, type=float)
    parser.add_argument("-band_mlp_hidden_dim", "--band_mlp_hidden_dim", default=16, type=int)
    parser.add_argument("-edge_mlp_hidden_dim", "--edge_mlp_hidden_dim", default=64, type=int)
    parser.add_argument("-attention_dropout", "--attention_dropout", default=0.0, type=float)
    parser.add_argument("-lambda_attn", "--lambda_attn", default=0.0, type=float)
    parser.add_argument("-learnable_score_fusion", "--learnable_score_fusion", default=False, type=str_to_bool)
    parser.add_argument("-init_score_fusion_alpha", "--init_score_fusion_alpha", default=0.7, type=float)
    parser.add_argument("-best_score_mode", "--best_score_mode", default="default", choices=["default", "corr_v2", "val_mae"])
    parser.add_argument("-seed", "--seed", default=2026, type=int)
    parser.add_argument("-random_seed", "--random_seed", default=None, type=int)
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
        default="Exp_DeltaPLI_PLIEncoder_GRU_RegOnly_LearnableLoss",
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
    args.use_temporal_band_edge_attention = cfg.get(
        "use_temporal_band_edge_attention",
        defaults["use_temporal_band_edge_attention"],
    )
    args.attn_scale = cfg.get("attn_scale", defaults["attn_scale"])
    args.gamma_band = cfg.get("gamma_band", defaults["gamma_band"])
    args.gamma_edge = cfg.get("gamma_edge", defaults["gamma_edge"])
    args.band_mlp_hidden_dim = cfg.get("band_mlp_hidden_dim", defaults["band_mlp_hidden_dim"])
    args.edge_mlp_hidden_dim = cfg.get("edge_mlp_hidden_dim", defaults["edge_mlp_hidden_dim"])
    args.attention_dropout = cfg.get("attention_dropout", defaults["attention_dropout"])
    args.lambda_attn = cfg.get("lambda_attn", defaults["lambda_attn"])
    args.learnable_score_fusion = cfg.get(
        "learnable_score_fusion",
        defaults["learnable_score_fusion"],
    )
    args.init_score_fusion_alpha = cfg.get(
        "init_score_fusion_alpha",
        defaults["init_score_fusion_alpha"],
    )
    args.best_score_mode = cfg.get("best_score_mode", "default")
    args.rank_min_label_gap_real = None
    args.rank_min_label_gap = 0.4

    print(f"Selected experiment_name = {exp_name}")
    print(f"Random seed = {args.seed}")
    print(f"use_fixed_split = {args.use_fixed_split}")
    print(f"fixed_split_path = {args.fixed_split_path}")
    print("mode = regression_only")
    print("PLI-oriented encoder enabled = True")
    print("MASA-TCN enabled = False")
    print(f"GRU Token Mixer enabled = {args.use_gru}")
    print("Regression-only training = True")
    print("Auxiliary classification heads = removed")
    if args.use_temporal_band_edge_attention:
        print("Backbone = PLIEncoderTemporalBandEdgeAttention")
        print(f"gamma_band = {args.gamma_band}")
        print(f"gamma_edge = {args.gamma_edge}")
        print(f"BandMLP hidden_dim = {args.band_mlp_hidden_dim}")
        print(f"EdgeMLP hidden_dim = {args.edge_mlp_hidden_dim}")
        print(f"Attention dropout = {args.attention_dropout}")
        if args.lambda_attn > 0:
            print("Attention regularization enabled = True")
            print(f"lambda_attn = {args.lambda_attn}")
    elif args.use_band_edge_attention:
        print("Backbone = PLIEncoderBandEdgeAttention")
        print(f"attn_scale = {args.attn_scale}")
    else:
        print("Backbone = PLIEncoder")
    if args.learnable_score_fusion:
        print("Score fusion learnable = True")
        print("expected_score = alpha * direct_score + (1-alpha) * ordinal_score")
        print(f"init_score_fusion_alpha = {args.init_score_fusion_alpha}")
    print(f"BestScore mode = {args.best_score_mode}")
    if args.best_score_mode == "val_mae":
        print("Checkpoint selection metric = validation MAE")
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
        "use_temporal_band_edge_attention": args.use_temporal_band_edge_attention,
        "attn_scale": args.attn_scale,
        "gamma_band": args.gamma_band,
        "gamma_edge": args.gamma_edge,
        "band_mlp_hidden_dim": args.band_mlp_hidden_dim,
        "edge_mlp_hidden_dim": args.edge_mlp_hidden_dim,
        "attention_dropout": args.attention_dropout,
        "lambda_attn": args.lambda_attn,
        "learnable_score_fusion": args.learnable_score_fusion,
        "init_score_fusion_alpha": args.init_score_fusion_alpha,
        "best_score_mode": args.best_score_mode,
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
        seed=args.seed,
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
        sys.stdout = sys.__stdout__
