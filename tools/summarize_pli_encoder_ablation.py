import argparse
import os
import re

import pandas as pd


DEFAULT_EXPERIMENTS = [
    "Exp2034_DeltaPLI_PLIEncoder_RegOnly_LearnableLoss_NoGRU",
    "Exp2034_DeltaPLI_PLIEncoder_GRU_RegOnly_LearnableLoss",
    "Exp2034_DeltaPLI_PLIEncoder_TemporalBandEdgeAttnLite_GRU_RegOnly_LearnableLoss",
    "Exp2034_DeltaPLI_PLIEncoder_TemporalBandEdgeAttnLite_GRU_RegOnly_LearnableLoss_lr1e4_wd5e4",
    "Exp2034_DeltaPLI_PLIEncoder_TemporalBandEdgeAttnLite_GRU_RegOnly_LearnableLoss_lr1e4_wd5e4_LearnableFusion",
]


METRIC_RE = re.compile(
    r"^(MAE|RMSE|R2|R²|PCC|CCC):\s*([-+0-9.eE]+)\s*[±卤]\s*([-+0-9.eE]+)",
    re.MULTILINE,
)


def read_text(path):
    for encoding in ("utf-8", "gbk"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def newest_log(exp_dir, exp_name):
    if not os.path.isdir(exp_dir):
        return None
    candidates = [
        os.path.join(exp_dir, name)
        for name in os.listdir(exp_dir)
        if name.endswith(".log") and name.startswith(exp_name)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def parse_final_metrics(text):
    metrics = {}
    for match in METRIC_RE.finditer(text):
        name = match.group(1).replace("R²", "R2").lower()
        metrics[f"{name}_mean"] = float(match.group(2))
        metrics[f"{name}_std"] = float(match.group(3))
    required = ["mae_mean", "rmse_mean", "r2_mean", "pcc_mean", "ccc_mean"]
    return metrics if all(key in metrics for key in required) else None


def experiment_metadata(exp_name):
    use_gru = "NoGRU" not in exp_name
    use_attn = "TemporalBandEdgeAttnLite" in exp_name
    learnable_fusion = "LearnableFusion" in exp_name
    lr = 1e-4 if "lr1e4" in exp_name else 2e-4
    weight_decay = 5e-4 if "wd5e4" in exp_name else 1e-4
    return {
        "encoder": "PLIEncoderTemporalBandEdgeAttentionLite" if use_attn else "PLIEncoder",
        "use_gru": use_gru,
        "use_temporal_band_edge_attention_lite": use_attn,
        "learnable_fusion": learnable_fusion,
        "lr": lr,
        "weight_decay": weight_decay,
        "gamma_band": 0.05 if use_attn else "",
        "gamma_edge": 0.05 if use_attn else "",
        "lambda_attn": 1e-3 if use_attn else "",
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize fixed-split PLIEncoder ablation logs.")
    parser.add_argument("--root", default=".", type=str)
    parser.add_argument(
        "--out_csv",
        default=os.path.join("results", "ablation_pli_encoder_seed2034_train2026_summary.csv"),
        type=str,
    )
    parser.add_argument(
        "--out_md",
        default=os.path.join("results", "ablation_pli_encoder_seed2034_train2026_summary.md"),
        type=str,
    )
    return parser


def main():
    args = build_parser().parse_args()
    rows = []
    missing = []

    for exp_name in DEFAULT_EXPERIMENTS:
        log_path = newest_log(os.path.join(args.root, exp_name), exp_name)
        if log_path is None:
            missing.append(f"{exp_name}: no log found")
            continue
        text = read_text(log_path)
        metrics = parse_final_metrics(text)
        if metrics is None:
            missing.append(f"{exp_name}: final metrics not found in {log_path}")
            continue

        row = {
            "experiment_name": exp_name,
            "fold_split_seed": 2034,
            "train_seed": 2026,
            "use_fixed_split": True,
            "fixed_split_path": os.path.join("splits", "fixed_5fold_seed2034.json"),
            **experiment_metadata(exp_name),
            **metrics,
            "pred_std_mean": "",
            "label_std_mean": "",
            "log_file": log_path,
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["mae_mean", "rmse_mean", "pcc_mean"], ascending=[True, True, False])
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    df.to_markdown(args.out_md, index=False)

    if missing:
        skipped_path = os.path.join(os.path.dirname(args.out_csv), "ablation_pli_encoder_missing_logs.txt")
        with open(skipped_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
        print(f"Missing or skipped logs written to {skipped_path}")

    print(f"summary_csv = {args.out_csv}")
    print(f"summary_md = {args.out_md}")
    if not df.empty:
        print(df[["experiment_name", "mae_mean", "rmse_mean", "r2_mean", "pcc_mean", "ccc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
