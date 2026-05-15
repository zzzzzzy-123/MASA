import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.dataset import DataArranger


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
    for folder_path, label, _ in items:
        x_rows.append(load_summary_features(folder_path, feature_names))
        y_rows.append(float(label))
    return np.stack(x_rows, axis=0), np.asarray(y_rows, dtype=np.float32)


def keep_extremes(x, y, low_q=0.3, high_q=0.7):
    low_cut, high_cut = np.quantile(y, [low_q, high_q])
    mask = (y <= low_cut) | (y >= high_cut)
    binary_y = (y[mask] >= high_cut).astype(np.float32)
    return x[mask], binary_y


def standardize(train_x, *others):
    mean = np.mean(train_x, axis=0, keepdims=True)
    std = np.std(train_x, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return [(x - mean) / std for x in (train_x, *others)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_logistic(train_x, train_y, lr=0.05, weight_decay=1e-3, epochs=800):
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
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1
    n_pos = int(np.sum(pos))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = float(np.sum(ranks[pos]))
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", default=r"C:\Users\浜戠懢\Desktop\data")
    parser.add_argument("--features", nargs="+", default=["eeg_PLI"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arranger = DataArranger(dataset_path=args.dataset_path, seed=args.seed)
    aucs = []

    for fold in range(5):
        fold_data = arranger.get_fold(fold)
        train_x, train_y_raw = build_matrix(fold_data["train"], args.features)
        test_x, test_y_raw = build_matrix(fold_data["test"], args.features)

        train_x, train_y = keep_extremes(train_x, train_y_raw)
        test_x, test_y = keep_extremes(test_x, test_y_raw)
        train_x, test_x = standardize(train_x, test_x)

        weights, bias = fit_logistic(train_x, train_y)
        scores = sigmoid(test_x @ weights + bias)
        auc = auc_score(test_y, scores)
        aucs.append(auc)

        print(
            f"fold{fold}: n_train={len(train_y)} n_test={len(test_y)} "
            f"AUC={auc:.4f}"
        )

    print(f"AUC mean +/- std: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")


if __name__ == "__main__":
    main()
