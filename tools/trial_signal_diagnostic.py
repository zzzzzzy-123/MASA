import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.dataset import DataArranger
from base.experiment_utils import compute_regression_metrics


def load_trial_features(folder_path, feature_names, trial_idx):
    features = []
    for feature_name in feature_names:
        data = np.load(Path(folder_path) / f"{feature_name}.npy").astype(np.float32)
        if data.ndim != 2 or data.shape[0] <= trial_idx:
            raise ValueError(f"{feature_name} has shape {data.shape}, cannot read trial {trial_idx}")
        features.append(data[trial_idx])
    return np.concatenate(features, axis=0)


def build_matrix(items, feature_names, trial_idx):
    x_rows = []
    y_rows = []
    for folder_path, label, _ in items:
        x_rows.append(load_trial_features(folder_path, feature_names, trial_idx))
        y_rows.append(float(label))
    return np.stack(x_rows, axis=0), np.asarray(y_rows, dtype=np.float32)


def standardize(train_x, *others):
    mean = np.mean(train_x, axis=0, keepdims=True)
    std = np.std(train_x, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return [(x - mean) / std for x in (train_x, *others)]


def fit_ridge(train_x, train_y, alpha):
    x_aug = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=train_x.dtype)], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + alpha * eye, x_aug.T @ train_y)


def predict_ridge(x, weights):
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def pick_alpha(train_x, train_y, val_x, val_y, alphas):
    best_alpha = alphas[0]
    best_score = float("inf")
    for alpha in alphas:
        weights = fit_ridge(train_x, train_y, alpha)
        preds = predict_ridge(val_x, weights)
        score = compute_regression_metrics(preds, val_y)["rmse"]
        if score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_logistic(train_x, train_y, lr=0.05, weight_decay=1e-3, epochs=600):
    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    bias = 0.0
    for _ in range(epochs):
        probs = sigmoid(train_x @ weights + bias)
        error = probs - train_y
        grad_w = train_x.T @ error / len(train_y) + weight_decay * weights
        grad_b = float(np.mean(error))
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def auc_score(y_true, scores):
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1
    n_pos = int(np.sum(pos))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    rank_sum = float(np.sum(ranks[pos]))
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def high_low_auc(train_x, train_y_raw, test_x, test_y_raw):
    low_cut, high_cut = np.quantile(train_y_raw, [1 / 3, 2 / 3])
    train_mask = (train_y_raw <= low_cut) | (train_y_raw >= high_cut)
    test_mask = (test_y_raw <= low_cut) | (test_y_raw >= high_cut)
    if np.sum(train_mask) < 8 or np.sum(test_mask) < 4:
        return np.nan

    train_y = (train_y_raw[train_mask] >= high_cut).astype(np.float32)
    test_y = (test_y_raw[test_mask] >= high_cut).astype(np.float32)
    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        return np.nan

    weights, bias = fit_logistic(train_x[train_mask], train_y)
    scores = sigmoid(test_x[test_mask] @ weights + bias)
    return auc_score(test_y, scores)


def summarize(values):
    values = np.asarray(values, dtype=np.float32)
    return float(np.nanmean(values)), float(np.nanstd(values))


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether individual animation trials contain SI-predictive signal."
    )
    parser.add_argument("-dataset_path", default=r"C:\Users\云瑾\Desktop\data")
    parser.add_argument("--features", nargs="+", default=["eeg_PLI"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_trials", type=int, default=40)
    parser.add_argument("--output", default="trial_signal_diagnostic.csv")
    args = parser.parse_args()

    arranger = DataArranger(dataset_path=args.dataset_path, seed=args.seed)
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    rows = []

    for trial_idx in range(args.num_trials):
        fold_rows = []
        for fold in range(5):
            fold_data = arranger.get_fold(fold)
            train_x, train_y = build_matrix(fold_data["train"], args.features, trial_idx)
            val_x, val_y = build_matrix(fold_data["validate"], args.features, trial_idx)
            test_x, test_y = build_matrix(fold_data["test"], args.features, trial_idx)

            train_x, val_x, test_x = standardize(train_x, val_x, test_x)
            alpha = pick_alpha(train_x, train_y, val_x, val_y, alphas)
            weights = fit_ridge(train_x, train_y, alpha)
            preds = predict_ridge(test_x, weights)
            metrics = compute_regression_metrics(preds, test_y)
            auc = high_low_auc(train_x, train_y, test_x, test_y)

            fold_rows.append({
                "trial": trial_idx,
                "fold": fold,
                "alpha": alpha,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "pcc": metrics["pcc"],
                "ccc": metrics["ccc"],
                "pred_std": float(np.std(preds)),
                "auc_high_low": auc,
            })

        rows.extend(fold_rows)
        pcc_mean, pcc_std = summarize([row["pcc"] for row in fold_rows])
        auc_mean, auc_std = summarize([row["auc_high_low"] for row in fold_rows])
        mae_mean, mae_std = summarize([row["mae"] for row in fold_rows])
        print(
            f"trial{trial_idx:02d}: "
            f"PCC={pcc_mean:.4f}+/-{pcc_std:.4f} "
            f"AUC={auc_mean:.4f}+/-{auc_std:.4f} "
            f"MAE={mae_mean:.4f}+/-{mae_std:.4f}"
        )

    detail_df = pd.DataFrame(rows)
    summary_df = detail_df.groupby("trial", as_index=False).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"),
        pcc_mean=("pcc", "mean"),
        pcc_std=("pcc", "std"),
        ccc_mean=("ccc", "mean"),
        pred_std_mean=("pred_std", "mean"),
        auc_mean=("auc_high_low", "mean"),
        auc_std=("auc_high_low", "std"),
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    detail_path = output_path.with_name(output_path.stem + "_folds" + output_path.suffix)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved summary: {output_path}")
    print(f"Saved fold details: {detail_path}")
    print("\nTop trials by PCC:")
    print(summary_df.sort_values("pcc_mean", ascending=False).head(10).to_string(index=False))
    print("\nTop trials by high-low AUC:")
    print(summary_df.sort_values("auc_mean", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
