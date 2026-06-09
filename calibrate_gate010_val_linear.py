import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT = "Exp_DeltaPLI_PLIEncoder_BandEdgeAttn_TCNLite_RegOnly_LearnableLoss_AntiOverfit_v1_Gate010"
DEFAULT_OUTPUT = Path("results") / "calibration_gate010_val_linear"
LABEL_CLIP_MIN = 0.0
LABEL_CLIP_MAX = 30.0


def resolve_experiment_dir(path_text):
    path = Path(path_text)
    if path.exists():
        return path
    alt = Path.cwd() / path.name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Experiment dir not found: {path_text}")


def pick_column(df, candidates, required=True):
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried {candidates}, got {list(df.columns)}")
    return None


def numeric(values):
    return pd.to_numeric(values, errors="coerce").astype(float).to_numpy()


def infer_fold_id(path, fallback):
    for part in reversed(path.parts):
        match = re.search(r"fold[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)) + 1
    return int(fallback)


def find_fold_pairs(experiment_dir):
    val_files = sorted(Path(experiment_dir).rglob("val_predictions.csv"))
    if not val_files:
        raise FileNotFoundError(f"No val_predictions.csv found under {experiment_dir}")
    pairs = []
    for idx, val_path in enumerate(val_files, start=1):
        test_path = val_path.parent / "test_predictions.csv"
        if not test_path.exists():
            continue
        pairs.append((infer_fold_id(val_path, idx), val_path, test_path))
    if not pairs:
        raise FileNotFoundError(f"No fold with both val_predictions.csv and test_predictions.csv under {experiment_dir}")
    return sorted(pairs, key=lambda item: item[0])


def load_predictions(path, fold_id):
    df = pd.read_csv(path)
    label_col = pick_column(df, ["label", "y_true", "target", "true_score", "si_score"])
    pred_col = pick_column(df, ["prediction", "pred", "y_pred", "expected_score", "pred_score"])
    id_col = pick_column(df, ["sample_id", "subject_id", "trial_id", "id", "file_id"], required=False)
    out = pd.DataFrame({
        "fold": fold_id,
        "sample_id": df[id_col].astype(str) if id_col is not None else [f"fold{fold_id}_sample{i}" for i in range(len(df))],
        "label": numeric(df[label_col]),
        "prediction": numeric(df[pred_col]),
    })
    out = out.dropna(subset=["label", "prediction"]).reset_index(drop=True)
    return out


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
    pred_std = float(np.std(preds)) if len(preds) else np.nan
    label_std = float(np.std(labels)) if len(labels) else np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "PCC": pearson_corr(preds, labels),
        "CCC": concordance_corr(preds, labels),
        "pred_mean": float(np.mean(preds)) if len(preds) else np.nan,
        "pred_std": pred_std,
        "label_mean": float(np.mean(labels)) if len(labels) else np.nan,
        "label_std": label_std,
        "pred_std_ratio": float(pred_std / (label_std + 1e-6)) if np.isfinite(label_std) else np.nan,
    }


def label_bin(label):
    if label <= 12:
        return "Low"
    if label <= 20:
        return "Mid"
    return "High"


def fit_calibrator(val_df):
    x = val_df["prediction"].to_numpy(dtype=float)
    y = val_df["label"].to_numpy(dtype=float)
    if np.std(x) < 1e-6:
        return 1.0, 0.0, 1.0, 0.0, "skipped_val_pred_std_too_small"
    a_raw, b_raw = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    a_used = float(np.clip(a_raw, 0.5, 3.0))
    status = "fitted"
    if abs(a_used - a_raw) > 1e-12:
        status = "fitted_slope_clipped"
    return float(a_raw), float(b_raw), a_used, float(b_raw), status


def bin_metrics(df, pred_col):
    labels = df["label"].to_numpy(dtype=float)
    preds = df[pred_col].to_numpy(dtype=float)
    err = preds - labels
    m = regression_metrics(preds, labels)
    return {
        "MAE": m["MAE"],
        "RMSE": m["RMSE"],
        "mean_error": float(np.mean(err)) if len(err) else np.nan,
        "median_abs_error": float(np.median(np.abs(err))) if len(err) else np.nan,
        "pred_mean": float(np.mean(preds)) if len(preds) else np.nan,
        "label_mean": float(np.mean(labels)) if len(labels) else np.nan,
        "underestimate_rate": float(np.mean(preds < labels)) if len(preds) else np.nan,
        "overestimate_rate": float(np.mean(preds > labels)) if len(preds) else np.nan,
    }


def make_plots(fold_metrics_df, bin_effect_df, all_df, output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    folds = fold_metrics_df["fold"].astype(str).tolist()
    x = np.arange(len(folds))
    width = 0.38

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, fold_metrics_df["raw_MAE"], width, label="Raw")
    plt.bar(x + width / 2, fold_metrics_df["calibrated_MAE"], width, label="Calibrated")
    plt.xticks(x, folds)
    plt.xlabel("Fold")
    plt.ylabel("MAE")
    plt.title("Fold MAE Raw vs Calibrated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fold_mae_raw_vs_calibrated.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, fold_metrics_df["raw_pred_std_ratio"], width, label="Raw")
    plt.bar(x + width / 2, fold_metrics_df["calibrated_pred_std_ratio"], width, label="Calibrated")
    plt.xticks(x, folds)
    plt.xlabel("Fold")
    plt.ylabel("pred_std / label_std")
    plt.title("Fold Prediction Std Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fold_pred_std_ratio_raw_vs_calibrated.png", dpi=150)
    plt.close()

    for pred_col, filename, title in [
        ("prediction_raw", "pred_vs_label_raw.png", "Raw Prediction vs Label"),
        ("prediction_calibrated", "pred_vs_label_calibrated.png", "Calibrated Prediction vs Label"),
    ]:
        plt.figure(figsize=(5, 5))
        plt.scatter(all_df["label"], all_df[pred_col], alpha=0.7)
        min_v = float(min(all_df["label"].min(), all_df[pred_col].min()))
        max_v = float(max(all_df["label"].max(), all_df[pred_col].max()))
        plt.plot([min_v, max_v], [min_v, max_v], "r--")
        plt.xlabel("Label")
        plt.ylabel("Prediction")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()

    for err_col, filename, title in [
        ("error_raw", "residual_vs_label_raw.png", "Raw Residual vs Label"),
        ("error_calibrated", "residual_vs_label_calibrated.png", "Calibrated Residual vs Label"),
    ]:
        plt.figure(figsize=(7, 4))
        plt.scatter(all_df["label"], all_df[err_col], alpha=0.7)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Label")
        plt.ylabel("Prediction - Label")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()

    ordered = ["Low", "Mid", "High"]
    plot_df = bin_effect_df.set_index("label_bin").reindex(ordered)
    x = np.arange(len(ordered))
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, plot_df["raw_MAE"], width, label="Raw")
    plt.bar(x + width / 2, plot_df["calibrated_MAE"], width, label="Calibrated")
    plt.xticks(x, ordered)
    plt.xlabel("Label Bin")
    plt.ylabel("MAE")
    plt.title("Label Bin MAE Raw vs Calibrated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "label_bin_mae_raw_vs_calibrated.png", dpi=150)
    plt.close()


def build_report(fold_df, summary_df, bin_df):
    raw_mae = float(summary_df.loc[0, "raw_MAE_mean"])
    cal_mae = float(summary_df.loc[0, "calibrated_MAE_mean"])
    improved = fold_df[fold_df["MAE_delta"] < 0]["fold"].tolist()
    worsened = fold_df[fold_df["MAE_delta"] > 0]["fold"].tolist()
    fold2 = fold_df[fold_df["fold"] == 2]
    low = bin_df[bin_df["label_bin"] == "Low"]
    high = bin_df[bin_df["label_bin"] == "High"]
    clipped = fold_df[fold_df["calibration_status"].str.contains("clipped", na=False)]["fold"].tolist()

    lines = []
    lines.append("Gate010 Fold-Wise Validation Linear Calibration")
    lines.append("=" * 60)
    for metric in ["MAE", "RMSE", "R2", "PCC", "CCC"]:
        lines.append(
            f"{metric}: raw={summary_df.loc[0, f'raw_{metric}_mean']:.4f}, "
            f"calibrated={summary_df.loc[0, f'calibrated_{metric}_mean']:.4f}"
        )
    lines.append(f"MAE < 4.0 after calibration: {cal_mae < 4.0}")
    if not fold2.empty:
        row = fold2.iloc[0]
        lines.append(f"Fold 2 MAE: raw={row['raw_MAE']:.4f}, calibrated={row['calibrated_MAE']:.4f}, delta={row['MAE_delta']:.4f}")
    lines.append(f"Improved folds by MAE: {improved}")
    lines.append(f"Worsened folds by MAE: {worsened}")
    if not low.empty:
        row = low.iloc[0]
        lines.append(
            f"Low bin MAE: raw={row['raw_MAE']:.4f}, calibrated={row['calibrated_MAE']:.4f}; "
            f"overestimate raw={row['raw_overestimate_rate']:.4f}, calibrated={row['calibrated_overestimate_rate']:.4f}"
        )
    if not high.empty:
        row = high.iloc[0]
        lines.append(
            f"High bin MAE: raw={row['raw_MAE']:.4f}, calibrated={row['calibrated_MAE']:.4f}; "
            f"underestimate raw={row['raw_underestimate_rate']:.4f}, calibrated={row['calibrated_underestimate_rate']:.4f}"
        )
    lines.append(
        f"pred_std_ratio: raw={summary_df.loc[0, 'raw_pred_std_ratio_mean']:.4f}, "
        f"calibrated={summary_df.loc[0, 'calibrated_pred_std_ratio_mean']:.4f}"
    )
    lines.append(f"Folds with slope clipping: {clipped}")
    lines.append("")
    lines.append("Recommendation:")
    if cal_mae < 4.0:
        lines.append("- Calibration reaches MAE < 4.0; Gate010 + validation linear calibration can be considered final post-processing.")
    elif cal_mae < raw_mae:
        lines.append("- Calibration improves MAE but does not reach 4.0; keep it as a supportive result.")
    else:
        lines.append("- Calibration worsens MAE; keep raw Gate010 as the main result.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Fold-wise validation linear calibration for Gate010.")
    parser.add_argument(
        "--experiment_dir",
        default=DEFAULT_EXPERIMENT,
        type=str,
        help="Gate010 experiment directory containing fold*/val_predictions.csv and fold*/test_predictions.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT),
        type=str,
        help="Output directory for calibration files.",
    )
    args = parser.parse_args()

    experiment_dir = resolve_experiment_dir(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    all_rows = []
    for fold_id, val_path, test_path in find_fold_pairs(experiment_dir):
        val_df = load_predictions(val_path, fold_id)
        test_df = load_predictions(test_path, fold_id)
        a_raw, b_raw, a_used, b_used, status = fit_calibrator(val_df)

        raw_pred = test_df["prediction"].to_numpy(dtype=float)
        labels = test_df["label"].to_numpy(dtype=float)
        cal_pred = np.clip(a_used * raw_pred + b_used, LABEL_CLIP_MIN, LABEL_CLIP_MAX)

        raw_metrics = regression_metrics(raw_pred, labels)
        cal_metrics = regression_metrics(cal_pred, labels)
        row = {
            "fold": fold_id,
            "n_val": len(val_df),
            "n_test": len(test_df),
            "a_raw": a_raw,
            "b_raw": b_raw,
            "a_used": a_used,
            "b_used": b_used,
            "calibration_status": status,
        }
        for metric in ["MAE", "RMSE", "R2", "PCC", "CCC"]:
            row[f"raw_{metric}"] = raw_metrics[metric]
            row[f"calibrated_{metric}"] = cal_metrics[metric]
            row[f"{metric}_delta"] = cal_metrics[metric] - raw_metrics[metric]
        row["raw_pred_std_ratio"] = raw_metrics["pred_std_ratio"]
        row["calibrated_pred_std_ratio"] = cal_metrics["pred_std_ratio"]
        row["label_std"] = raw_metrics["label_std"]
        row["raw_pred_mean"] = raw_metrics["pred_mean"]
        row["raw_pred_std"] = raw_metrics["pred_std"]
        row["calibrated_pred_mean"] = cal_metrics["pred_mean"]
        row["calibrated_pred_std"] = cal_metrics["pred_std"]
        fold_rows.append(row)

        out = pd.DataFrame({
            "sample_id": test_df["sample_id"],
            "fold": fold_id,
            "label": labels,
            "prediction_raw": raw_pred,
            "prediction_calibrated": cal_pred,
        })
        out["error_raw"] = out["prediction_raw"] - out["label"]
        out["error_calibrated"] = out["prediction_calibrated"] - out["label"]
        out["abs_error_raw"] = np.abs(out["error_raw"])
        out["abs_error_calibrated"] = np.abs(out["error_calibrated"])
        out["label_bin"] = out["label"].apply(label_bin)
        fold_csv = output_dir / f"fold_{fold_id}_test_predictions_calibrated.csv"
        out.to_csv(fold_csv, index=False, encoding="utf-8-sig")
        all_rows.append(out)

    fold_df = pd.DataFrame(fold_rows).sort_values("fold")
    fold_metrics_path = output_dir / "fold_calibration_metrics.csv"
    fold_df.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(output_dir / "all_test_predictions_calibrated.csv", index=False, encoding="utf-8-sig")

    summary = {}
    for metric in ["MAE", "RMSE", "R2", "PCC", "CCC"]:
        summary[f"raw_{metric}_mean"] = float(fold_df[f"raw_{metric}"].mean())
        summary[f"raw_{metric}_std"] = float(fold_df[f"raw_{metric}"].std(ddof=0))
        summary[f"calibrated_{metric}_mean"] = float(fold_df[f"calibrated_{metric}"].mean())
        summary[f"calibrated_{metric}_std"] = float(fold_df[f"calibrated_{metric}"].std(ddof=0))
    summary["raw_pred_std_ratio_mean"] = float(fold_df["raw_pred_std_ratio"].mean())
    summary["calibrated_pred_std_ratio_mean"] = float(fold_df["calibrated_pred_std_ratio"].mean())
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "calibration_summary.csv", index=False, encoding="utf-8-sig")

    bin_rows = []
    for bin_name, group in all_df.groupby("label_bin"):
        raw = bin_metrics(group, "prediction_raw")
        cal = bin_metrics(group, "prediction_calibrated")
        row = {
            "label_bin": bin_name,
            "n_samples": len(group),
            "raw_MAE": raw["MAE"],
            "calibrated_MAE": cal["MAE"],
            "MAE_delta": cal["MAE"] - raw["MAE"],
            "raw_mean_error": raw["mean_error"],
            "calibrated_mean_error": cal["mean_error"],
            "raw_underestimate_rate": raw["underestimate_rate"],
            "calibrated_underestimate_rate": cal["underestimate_rate"],
            "raw_overestimate_rate": raw["overestimate_rate"],
            "calibrated_overestimate_rate": cal["overestimate_rate"],
            "raw_RMSE": raw["RMSE"],
            "calibrated_RMSE": cal["RMSE"],
            "raw_median_abs_error": raw["median_abs_error"],
            "calibrated_median_abs_error": cal["median_abs_error"],
            "raw_pred_mean": raw["pred_mean"],
            "calibrated_pred_mean": cal["pred_mean"],
            "label_mean": raw["label_mean"],
        }
        bin_rows.append(row)
    order = {"Low": 0, "Mid": 1, "High": 2}
    bin_df = pd.DataFrame(bin_rows).sort_values("label_bin", key=lambda s: s.map(order))
    bin_df.to_csv(output_dir / "label_bin_calibration_effect.csv", index=False, encoding="utf-8-sig")

    make_plots(fold_df, bin_df, all_df, output_dir)
    report = build_report(fold_df, summary_df, bin_df)
    (output_dir / "calibration_report.txt").write_text(report, encoding="utf-8")

    print("\nValidation linear calibration completed.")
    print(f"Output dir: {output_dir}")
    print(f"Raw 5-fold MAE: {summary['raw_MAE_mean']:.4f} ± {summary['raw_MAE_std']:.4f}")
    print(f"Calibrated 5-fold MAE: {summary['calibrated_MAE_mean']:.4f} ± {summary['calibrated_MAE_std']:.4f}")
    print(f"MAE < 4.0 after calibration: {summary['calibrated_MAE_mean'] < 4.0}")
    fold2 = fold_df[fold_df["fold"] == 2]
    if not fold2.empty:
        print(f"Fold 2 MAE delta: {fold2.iloc[0]['MAE_delta']:.4f}")
    low = bin_df[bin_df["label_bin"] == "Low"]
    high = bin_df[bin_df["label_bin"] == "High"]
    if not low.empty:
        print(f"Low bin MAE delta: {low.iloc[0]['MAE_delta']:.4f}")
    if not high.empty:
        print(f"High bin MAE delta: {high.iloc[0]['MAE_delta']:.4f}")
    print(f"Improved folds: {fold_df[fold_df['MAE_delta'] < 0]['fold'].tolist()}")
    print(f"Worsened folds: {fold_df[fold_df['MAE_delta'] > 0]['fold'].tolist()}")


if __name__ == "__main__":
    main()
