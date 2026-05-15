import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.dataset import DataArranger
from base.experiment_utils import compute_regression_metrics


def load_summary_features(folder_path, feature_names):
    summaries = []
    for feature_name in feature_names:
        data = np.load(Path(folder_path) / f"{feature_name}.npy").astype(np.float32)
        summaries.extend([
            np.mean(data, axis=0),
            np.std(data, axis=0),
            np.median(data, axis=0),
            np.percentile(data, 25, axis=0),
            np.percentile(data, 75, axis=0),
        ])
    return np.concatenate(summaries, axis=0)


def build_matrix(items, feature_names):
    x_rows = []
    y_rows = []
    ids = []
    for folder_path, label, trial_id in items:
        x_rows.append(load_summary_features(folder_path, feature_names))
        y_rows.append(float(label))
        ids.append(trial_id)
    return np.stack(x_rows, axis=0), np.asarray(y_rows, dtype=np.float32), ids


def standardize(train_x, *others):
    mean = np.mean(train_x, axis=0, keepdims=True)
    std = np.std(train_x, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return [(x - mean) / std for x in (train_x, *others)]


def fit_ridge(train_x, train_y, alpha):
    x_aug = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=train_x.dtype)], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + alpha * eye, x_aug.T @ train_y)
    return weights


def predict_ridge(x, weights):
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def pick_alpha(train_x, train_y, val_x, val_y, alphas):
    best_alpha = alphas[0]
    best_rmse = float("inf")
    for alpha in alphas:
        weights = fit_ridge(train_x, train_y, alpha)
        preds = predict_ridge(val_x, weights)
        rmse = compute_regression_metrics(preds, val_y)["rmse"]
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    return best_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", default=r"C:\Users\浜戠懢\Desktop\data")
    parser.add_argument("--features", nargs="+", default=["eeg_PLI"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arranger = DataArranger(dataset_path=args.dataset_path, seed=args.seed)
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    fold_metrics = []

    for fold in range(5):
        fold_data = arranger.get_fold(fold)
        train_x, train_y, _ = build_matrix(fold_data["train"], args.features)
        val_x, val_y, _ = build_matrix(fold_data["validate"], args.features)
        test_x, test_y, _ = build_matrix(fold_data["test"], args.features)

        train_x, val_x, test_x = standardize(train_x, val_x, test_x)
        alpha = pick_alpha(train_x, train_y, val_x, val_y, alphas)
        weights = fit_ridge(train_x, train_y, alpha)
        preds = predict_ridge(test_x, weights)
        metrics = compute_regression_metrics(preds, test_y)
        fold_metrics.append(metrics)

        print(
            f"fold{fold}: alpha={alpha:g} "
            f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
            f"R2={metrics['r2']:.4f} PCC={metrics['pcc']:.4f} CCC={metrics['ccc']:.4f}"
        )

    print("\nRidge summary")
    for name in ["mae", "rmse", "r2", "pcc", "ccc"]:
        values = [m[name] for m in fold_metrics]
        print(f"{name.upper()}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


if __name__ == "__main__":
    main()
