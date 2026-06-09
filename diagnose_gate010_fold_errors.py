import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT = "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010"
DEFAULT_OUTPUT = Path("results") / "error_diagnosis_gate010"


def pick_column(df, candidates, required=True):
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}. Available: {list(df.columns)}")
    return None


def safe_float_array(values):
    return pd.to_numeric(values, errors="coerce").astype(float).to_numpy()


def pearson_corr(preds, labels):
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(preds) < 2 or np.std(preds) < 1e-12 or np.std(labels) < 1e-12:
        return np.nan
    return float(np.corrcoef(preds, labels)[0, 1])


def concordance_corr(preds, labels):
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(preds) == 0:
        return np.nan
    mean_p = float(np.mean(preds))
    mean_l = float(np.mean(labels))
    var_p = float(np.var(preds))
    var_l = float(np.var(labels))
    cov = float(np.mean((preds - mean_p) * (labels - mean_l)))
    denom = var_p + var_l + (mean_p - mean_l) ** 2
    if abs(denom) < 1e-12:
        return np.nan
    return float((2.0 * cov) / denom)


def regression_metrics(preds, labels):
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    err = preds - labels
    mae = float(np.mean(np.abs(err))) if len(err) else np.nan
    rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((labels - np.mean(labels)) ** 2)) if len(labels) else np.nan
    r2 = np.nan if not np.isfinite(ss_tot) or ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "PCC": pearson_corr(preds, labels),
        "CCC": concordance_corr(preds, labels),
    }


def collapse_level(pred_std_ratio):
    if not np.isfinite(pred_std_ratio):
        return "unknown"
    if pred_std_ratio < 0.10:
        return "severe"
    if pred_std_ratio < 0.25:
        return "mild_or_moderate"
    return "acceptable"


def label_bin(label):
    if label <= 12:
        return "Low"
    if label <= 20:
        return "Mid"
    return "High"


def infer_fold_id(path, fallback):
    for part in reversed(path.parts):
        match = re.search(r"fold[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)) + 1
    return int(fallback)


def find_prediction_files(experiment_dir):
    files = sorted(Path(experiment_dir).rglob("test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No test_predictions.csv found under {experiment_dir}")
    return files


def load_fold_predictions(csv_path, fold_id):
    df = pd.read_csv(csv_path)
    label_col = pick_column(df, ["label", "y_true", "target", "true_score"])
    pred_col = pick_column(df, ["prediction", "pred", "y_pred", "expected_score"])
    id_col = pick_column(df, ["subject_id", "sample_id", "trial_id", "id"], required=False)
    baseline_col = pick_column(df, ["baseline_pred", "train_mean", "mean_baseline", "baseline"], required=False)

    out = pd.DataFrame({
        "fold": fold_id,
        "sample_id": df[id_col].astype(str) if id_col is not None else [f"fold{fold_id}_sample{i}" for i in range(len(df))],
        "label": safe_float_array(df[label_col]),
        "prediction": safe_float_array(df[pred_col]),
        "source_file": str(csv_path),
    })
    out = out.dropna(subset=["label", "prediction"]).reset_index(drop=True)
    if baseline_col is not None:
        baseline_values = safe_float_array(df.loc[out.index, baseline_col])
        if len(baseline_values) == len(out) and np.isfinite(baseline_values).any():
            out["baseline_pred"] = baseline_values
    out["error"] = out["prediction"] - out["label"]
    out["abs_error"] = np.abs(out["error"])
    out["label_bin"] = out["label"].apply(label_bin)
    return out


def fold_metrics(fold_df):
    labels = fold_df["label"].to_numpy(dtype=float)
    preds = fold_df["prediction"].to_numpy(dtype=float)
    metrics = regression_metrics(preds, labels)

    if "baseline_pred" in fold_df.columns and np.isfinite(fold_df["baseline_pred"]).any():
        baseline_pred = float(np.nanmean(fold_df["baseline_pred"]))
        baseline_type = "train_mean_from_predictions"
    else:
        baseline_pred = float(np.mean(labels))
        baseline_type = "test_mean_reference"

    baseline_mae = float(np.mean(np.abs(labels - baseline_pred)))
    pred_std = float(np.std(preds))
    label_std = float(np.std(labels))
    pred_std_ratio = float(pred_std / (label_std + 1e-6))
    return {
        "fold": int(fold_df["fold"].iloc[0]),
        "n_test": int(len(fold_df)),
        **metrics,
        "pred_mean": float(np.mean(preds)),
        "pred_std": pred_std,
        "label_mean": float(np.mean(labels)),
        "label_std": label_std,
        "pred_std_ratio": pred_std_ratio,
        "mean_baseline_MAE": baseline_mae,
        "model_minus_baseline_MAE": float(metrics["MAE"] - baseline_mae),
        "baseline_type": baseline_type,
        "collapse_level": collapse_level(pred_std_ratio),
    }


def bin_error_metrics(df):
    labels = df["label"].to_numpy(dtype=float)
    preds = df["prediction"].to_numpy(dtype=float)
    err = preds - labels
    metrics = regression_metrics(preds, labels)
    return {
        "n_samples": int(len(df)),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "mean_error": float(np.mean(err)) if len(err) else np.nan,
        "median_abs_error": float(np.median(np.abs(err))) if len(err) else np.nan,
        "pred_mean": float(np.mean(preds)) if len(preds) else np.nan,
        "label_mean": float(np.mean(labels)) if len(labels) else np.nan,
        "underestimate_rate": float(np.mean(preds < labels)) if len(preds) else np.nan,
        "overestimate_rate": float(np.mean(preds > labels)) if len(preds) else np.nan,
    }


def write_csv(df, output_dir, filename):
    path = Path(output_dir) / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def make_plots(fold_df, bin_df, fold_bin_df, all_df, output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    folds = fold_df["fold"].astype(str).tolist()
    mean_mae = float(fold_df["MAE"].mean())

    plt.figure(figsize=(8, 4))
    plt.bar(folds, fold_df["MAE"])
    plt.axhline(mean_mae, color="red", linestyle="--", label=f"Mean MAE={mean_mae:.3f}")
    plt.xlabel("Fold")
    plt.ylabel("MAE")
    plt.title("Fold MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fold_mae_bar.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    colors = ["#d95f02" if v > 0 else "#1b9e77" for v in fold_df["model_minus_baseline_MAE"]]
    plt.bar(folds, fold_df["model_minus_baseline_MAE"], color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Fold")
    plt.ylabel("Model MAE - Mean Baseline MAE")
    plt.title("Model vs Mean Baseline")
    plt.tight_layout()
    plt.savefig(output_dir / "fold_model_minus_baseline_bar.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(all_df["label"], all_df["prediction"], alpha=0.7)
    min_v = float(min(all_df["label"].min(), all_df["prediction"].min()))
    max_v = float(max(all_df["label"].max(), all_df["prediction"].max()))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", label="y=x")
    plt.xlabel("Label")
    plt.ylabel("Prediction")
    plt.title("Prediction vs Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pred_vs_label_scatter_all.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(all_df["label"], all_df["error"], alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Label")
    plt.ylabel("Prediction - Label")
    plt.title("Residual vs Label")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_label_all.png", dpi=150)
    plt.close()

    ordered_bins = ["Low", "Mid", "High"]
    bin_plot = bin_df.set_index("label_bin").reindex(ordered_bins)
    plt.figure(figsize=(6, 4))
    plt.bar(ordered_bins, bin_plot["MAE"])
    plt.xlabel("Label Bin")
    plt.ylabel("MAE")
    plt.title("Label Bin MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "label_bin_mae_bar.png", dpi=150)
    plt.close()

    heat = fold_bin_df.pivot(index="fold", columns="label_bin", values="MAE").reindex(columns=ordered_bins)
    plt.figure(figsize=(7, 4))
    im = plt.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, label="MAE")
    plt.xticks(np.arange(len(ordered_bins)), ordered_bins)
    plt.yticks(np.arange(len(heat.index)), heat.index.astype(str))
    plt.xlabel("Label Bin")
    plt.ylabel("Fold")
    plt.title("Fold by Label Bin MAE")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            if np.isfinite(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "fold_by_label_bin_mae_heatmap.png", dpi=150)
    plt.close()


def build_report(fold_df, bin_df, fold_bin_df, top30_df):
    lines = []
    mean_mae = float(fold_df["MAE"].mean())
    worst_mae = fold_df.loc[fold_df["MAE"].idxmax()]
    worst_baseline = fold_df.loc[fold_df["model_minus_baseline_MAE"].idxmax()]
    worst_pcc = fold_df.loc[fold_df["PCC"].idxmin()]
    lowest_ratio = fold_df.loc[fold_df["pred_std_ratio"].idxmin()]
    worst_bin = bin_df.loc[bin_df["MAE"].idxmax()]

    high_row = bin_df[bin_df["label_bin"] == "High"]
    low_row = bin_df[bin_df["label_bin"] == "Low"]
    high_under = bool(not high_row.empty and high_row.iloc[0]["underestimate_rate"] > 0.6)
    low_over = bool(not low_row.empty and low_row.iloc[0]["overestimate_rate"] > 0.6)
    failed_baseline = fold_df[fold_df["model_minus_baseline_MAE"] > 0]["fold"].tolist()
    passed_baseline = fold_df[fold_df["model_minus_baseline_MAE"] <= 0]["fold"].tolist()
    collapse_folds = fold_df[fold_df["collapse_level"] != "acceptable"][["fold", "collapse_level", "pred_std_ratio"]]
    top_fold_counts = top30_df["fold"].value_counts().to_dict()
    top_bin_counts = top30_df["label_bin"].value_counts().to_dict()

    lines.append("Gate010 Fold-Level Error Diagnosis")
    lines.append("=" * 60)
    lines.append(f"Current 5-fold mean MAE: {mean_mae:.4f}")
    lines.append(f"Worst fold by MAE: fold {int(worst_mae['fold'])}, MAE={worst_mae['MAE']:.4f}")
    lines.append(
        f"Worst fold vs baseline: fold {int(worst_baseline['fold'])}, "
        f"model_minus_baseline_MAE={worst_baseline['model_minus_baseline_MAE']:.4f}"
    )
    lines.append(f"Lowest PCC fold: fold {int(worst_pcc['fold'])}, PCC={worst_pcc['PCC']:.4f}")
    lines.append(
        f"Lowest pred_std_ratio fold: fold {int(lowest_ratio['fold'])}, "
        f"ratio={lowest_ratio['pred_std_ratio']:.4f}, collapse={lowest_ratio['collapse_level']}"
    )
    lines.append(f"Folds outperforming mean baseline: {passed_baseline}")
    lines.append(f"Folds not outperforming mean baseline: {failed_baseline}")
    lines.append(f"Worst label bin: {worst_bin['label_bin']}, MAE={worst_bin['MAE']:.4f}")
    lines.append(f"High-score systematic underestimation: {high_under}")
    lines.append(f"Low-score systematic overestimation: {low_over}")
    lines.append(f"Top-30 large errors by fold: {top_fold_counts}")
    lines.append(f"Top-30 large errors by label bin: {top_bin_counts}")
    if collapse_folds.empty:
        lines.append("Prediction distribution collapse: no fold below pred_std_ratio 0.25.")
    else:
        lines.append("Prediction distribution collapse detected:")
        for _, row in collapse_folds.iterrows():
            lines.append(f"  fold {int(row['fold'])}: {row['collapse_level']} ({row['pred_std_ratio']:.4f})")

    lines.append("")
    lines.append("Next-step suggestions:")
    if high_under:
        lines.append("- High-score underestimation is visible; try validation linear calibration.")
    if failed_baseline:
        lines.append("- Some folds do not beat mean baseline; inspect their label and subject distributions.")
    if not collapse_folds.empty:
        lines.append("- Prediction range is compressed; prefer lightweight calibration before changing architecture again.")
    if worst_bin["MAE"] > bin_df["MAE"].mean() * 1.15:
        lines.append("- Errors concentrate in specific label bins; consider segment calibration or sample weighting.")
    if not (high_under or failed_baseline or not collapse_folds.empty):
        lines.append("- No single obvious failure mode dominates; compare fold distribution and top-error subjects.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Diagnose Gate010 fold-level prediction errors.")
    parser.add_argument(
        "--experiment_dir",
        default=DEFAULT_EXPERIMENT,
        type=str,
        help="Experiment directory containing fold*/test_predictions.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT),
        type=str,
        help="Directory for diagnosis CSVs, plots, and report.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = find_prediction_files(experiment_dir)
    fold_frames = []
    for idx, csv_path in enumerate(prediction_files, start=1):
        fold_id = infer_fold_id(csv_path, idx)
        fold_frames.append(load_fold_predictions(csv_path, fold_id))

    all_df = pd.concat(fold_frames, ignore_index=True)
    all_df = all_df.sort_values(["fold", "sample_id"]).reset_index(drop=True)
    write_csv(all_df, output_dir, "all_test_predictions_with_errors.csv")

    fold_rows = [fold_metrics(df) for _, df in all_df.groupby("fold")]
    fold_df = pd.DataFrame(fold_rows).sort_values("fold")
    write_csv(fold_df, output_dir, "fold_level_metrics.csv")

    bin_rows = []
    for bin_name, group in all_df.groupby("label_bin"):
        row = {"label_bin": bin_name}
        row.update(bin_error_metrics(group))
        bin_rows.append(row)
    bin_df = pd.DataFrame(bin_rows)
    bin_order = {"Low": 0, "Mid": 1, "High": 2}
    bin_df = bin_df.sort_values("label_bin", key=lambda s: s.map(bin_order))
    write_csv(bin_df, output_dir, "label_bin_error_analysis.csv")

    fold_bin_rows = []
    for (fold, bin_name), group in all_df.groupby(["fold", "label_bin"]):
        row = {"fold": int(fold), "label_bin": bin_name}
        row.update(bin_error_metrics(group))
        fold_bin_rows.append(row)
    fold_bin_df = pd.DataFrame(fold_bin_rows).sort_values(["fold", "label_bin"], key=lambda s: s.map(bin_order) if s.name == "label_bin" else s)
    write_csv(fold_bin_df, output_dir, "fold_by_label_bin_error.csv")

    top30_df = all_df.sort_values("abs_error", ascending=False).head(30).copy()
    top30_df.insert(0, "rank", np.arange(1, len(top30_df) + 1))
    write_csv(top30_df[["rank", "fold", "sample_id", "label", "prediction", "error", "abs_error", "label_bin"]], output_dir, "top30_abs_error_samples.csv")

    top10_fold_df = (
        all_df.sort_values(["fold", "abs_error"], ascending=[True, False])
        .groupby("fold")
        .head(10)
        .copy()
    )
    top10_fold_df["rank_in_fold"] = top10_fold_df.groupby("fold")["abs_error"].rank(method="first", ascending=False).astype(int)
    write_csv(
        top10_fold_df[["fold", "rank_in_fold", "sample_id", "label", "prediction", "error", "abs_error", "label_bin"]],
        output_dir,
        "top10_abs_error_samples_by_fold.csv",
    )

    make_plots(fold_df, bin_df, fold_bin_df, all_df, output_dir)
    report = build_report(fold_df, bin_df, fold_bin_df, top30_df)
    report_path = output_dir / "error_diagnosis_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\nFold-level diagnosis completed.")
    print(f"Output dir: {output_dir}")
    print("\nKey findings:")
    print(f"MAE highest fold: {int(fold_df.loc[fold_df['MAE'].idxmax(), 'fold'])}")
    print(f"model_minus_baseline_MAE largest fold: {int(fold_df.loc[fold_df['model_minus_baseline_MAE'].idxmax(), 'fold'])}")
    print(f"PCC lowest fold: {int(fold_df.loc[fold_df['PCC'].idxmin(), 'fold'])}")
    print(f"pred_std_ratio lowest fold: {int(fold_df.loc[fold_df['pred_std_ratio'].idxmin(), 'fold'])}")
    for _, row in fold_df.iterrows():
        print(f"fold {int(row['fold'])}: pred_std_ratio={row['pred_std_ratio']:.4f}, collapse_level={row['collapse_level']}")

    worst_bin = bin_df.loc[bin_df["MAE"].idxmax()]
    high_row = bin_df[bin_df["label_bin"] == "High"]
    low_row = bin_df[bin_df["label_bin"] == "Low"]
    print(f"Worst label bin by MAE: {worst_bin['label_bin']} ({worst_bin['MAE']:.4f})")
    if not high_row.empty:
        print(f"High bin systematic underestimation: {high_row.iloc[0]['underestimate_rate'] > 0.6}")
    if not low_row.empty:
        print(f"Low bin systematic overestimation: {low_row.iloc[0]['overestimate_rate'] > 0.6}")


if __name__ == "__main__":
    main()
