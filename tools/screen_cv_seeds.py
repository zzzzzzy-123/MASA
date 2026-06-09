import argparse
import json
import os
import random

import numpy as np
import pandas as pd


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def find_column(df, candidates, contains=None):
    stripped = {str(col).strip(): col for col in df.columns}
    for name in candidates:
        if name in stripped:
            return stripped[name]
    if contains:
        for col in df.columns:
            if contains.lower() in str(col).strip().lower():
                return col
    return None


def load_trials(label_file, compacted_dir=None):
    df = pd.read_excel(label_file)
    df.columns = df.columns.map(lambda x: str(x).strip())

    label_col = find_column(df, ["SI", "si"], contains="SI")
    if label_col is None:
        raise ValueError(f"Could not find SI label column in {label_file}")

    id_col = find_column(
        df,
        ["序号", "搴忓彿", "id", "ID", "subject_id", "Subject", "被试编号"],
    )

    label_by_id = {}
    for row_idx, row in df.iterrows():
        label = row[label_col]
        if pd.isna(label):
            continue
        if id_col is not None and not pd.isna(row[id_col]):
            try:
                subject_no = str(int(float(row[id_col])))
            except Exception:
                subject_no = str(row[id_col]).strip()
            trial_id = f"P{subject_no}-T1"
        else:
            trial_id = f"sample_{row_idx:04d}"
        label_by_id[trial_id] = float(label)

    if compacted_dir and os.path.isdir(compacted_dir):
        trial_ids = sorted(
            folder
            for folder in os.listdir(compacted_dir)
            if os.path.isdir(os.path.join(compacted_dir, folder))
            and any(name.endswith(".npy") for name in os.listdir(os.path.join(compacted_dir, folder)))
            and folder in label_by_id
        )
    else:
        trial_ids = sorted(label_by_id.keys())

    trials = [
        {
            "sample_index": idx,
            "trial_id": trial_id,
            "label": float(label_by_id[trial_id]),
        }
        for idx, trial_id in enumerate(trial_ids)
    ]
    if not trials:
        raise ValueError("No labeled samples were found for seed screening.")
    return trials


def stratified_fold_indices(labels, num_folds, seed):
    labels = np.asarray(labels, dtype=np.float32)
    bins = np.quantile(labels, [1 / 3, 2 / 3])
    strata = np.digitize(labels, bins)

    folds = [[] for _ in range(num_folds)]
    rng = random.Random(int(seed))
    for stratum in sorted(set(strata.tolist())):
        indices = [idx for idx, value in enumerate(strata) if value == stratum]
        rng.shuffle(indices)
        for offset, idx in enumerate(indices):
            folds[offset % num_folds].append(idx)
    return folds


def stratified_val_indices(labels, val_ratio, seed):
    labels = np.asarray(labels, dtype=np.float32)
    bins = np.quantile(labels, [1 / 3, 2 / 3])
    strata = np.digitize(labels, bins)

    rng = random.Random(int(seed))
    val_indices = []
    for stratum in sorted(set(strata.tolist())):
        indices = [idx for idx, value in enumerate(strata) if value == stratum]
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_ratio)))
        val_indices.extend(indices[:n_val])
    return set(val_indices)


def make_split(trials, seed, num_folds):
    labels = [item["label"] for item in trials]
    test_folds = stratified_fold_indices(labels, num_folds, seed)

    folds = []
    all_test_ids = []
    for fold_idx in range(num_folds):
        test_indices = set(test_folds[fold_idx])
        train_val_pairs = [
            (idx, item)
            for idx, item in enumerate(trials)
            if idx not in test_indices
        ]
        train_val_labels = [item["label"] for _, item in train_val_pairs]
        val_local_indices = stratified_val_indices(
            train_val_labels,
            val_ratio=0.2,
            seed=int(seed) + fold_idx,
        )

        train_ids = [
            item["trial_id"]
            for local_idx, (_, item) in enumerate(train_val_pairs)
            if local_idx not in val_local_indices
        ]
        val_ids = [
            item["trial_id"]
            for local_idx, (_, item) in enumerate(train_val_pairs)
            if local_idx in val_local_indices
        ]
        test_ids = [trials[idx]["trial_id"] for idx in sorted(test_indices)]
        all_test_ids.extend(test_ids)

        folds.append(
            {
                "fold": fold_idx + 1,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            }
        )

    all_ids = [item["trial_id"] for item in trials]
    if sorted(all_test_ids) != sorted(all_ids):
        raise ValueError("Split validation failed: test folds do not cover all samples exactly once.")
    return folds


def label_bucket_counts(labels, low_threshold, high_threshold):
    labels = np.asarray(labels, dtype=np.float32)
    low = int(np.sum(labels <= low_threshold))
    high = int(np.sum(labels >= high_threshold))
    mid = int(len(labels) - low - high)
    total = max(1, len(labels))
    return {
        "low_count": low,
        "mid_count": mid,
        "high_count": high,
        "low_ratio": low / total,
        "mid_ratio": mid / total,
        "high_ratio": high / total,
    }


def fold_stats_for_seed(trials, folds):
    label_by_id = {item["trial_id"]: item["label"] for item in trials}
    all_labels = np.asarray([item["label"] for item in trials], dtype=np.float32)
    low_threshold, high_threshold = np.quantile(all_labels, [0.30, 0.70])

    rows = []
    for fold in folds:
        test_labels = np.asarray([label_by_id[item_id] for item_id in fold["test_ids"]], dtype=np.float32)
        counts = label_bucket_counts(test_labels, low_threshold, high_threshold)
        rows.append(
            {
                "fold": int(fold["fold"]),
                "train_size": len(fold["train_ids"]),
                "val_size": len(fold["val_ids"]),
                "test_size": len(fold["test_ids"]),
                "test_label_mean": float(np.mean(test_labels)),
                "test_label_std": float(np.std(test_labels)),
                "test_label_min": float(np.min(test_labels)),
                "test_label_max": float(np.max(test_labels)),
                **counts,
            }
        )
    return rows


def score_seed(fold_rows):
    test_sizes = np.asarray([row["test_size"] for row in fold_rows], dtype=np.float32)
    means = np.asarray([row["test_label_mean"] for row in fold_rows], dtype=np.float32)
    stds = np.asarray([row["test_label_std"] for row in fold_rows], dtype=np.float32)
    low_ratios = np.asarray([row["low_ratio"] for row in fold_rows], dtype=np.float32)
    mid_ratios = np.asarray([row["mid_ratio"] for row in fold_rows], dtype=np.float32)
    high_ratios = np.asarray([row["high_ratio"] for row in fold_rows], dtype=np.float32)
    ranges = np.asarray(
        [row["test_label_max"] - row["test_label_min"] for row in fold_rows],
        dtype=np.float32,
    )

    missing_extreme_folds = sum(
        1 for row in fold_rows if row["low_count"] == 0 or row["high_count"] == 0
    )
    missing_extreme_penalty = 10.0 * missing_extreme_folds

    # Penalize unusually narrow label ranges relative to the median fold range.
    median_range = float(np.median(ranges))
    narrow_range_folds = int(np.sum(ranges < 0.6 * median_range)) if median_range > 1e-8 else 0
    range_penalty = 1.0 * narrow_range_folds

    score = (
        float(np.std(means))
        + 0.5 * float(np.std(stds))
        + 2.0 * float(np.std(low_ratios))
        + 2.0 * float(np.std(mid_ratios))
        + 2.0 * float(np.std(high_ratios))
        + 0.1 * float(np.std(test_sizes))
        + missing_extreme_penalty
        + range_penalty
    )
    return {
        "split_balance_score": score,
        "mean_of_test_label_means": float(np.mean(means)),
        "std_of_test_label_means": float(np.std(means)),
        "mean_of_test_label_stds": float(np.mean(stds)),
        "std_of_test_label_stds": float(np.std(stds)),
        "std_low_ratio": float(np.std(low_ratios)),
        "std_mid_ratio": float(np.std(mid_ratios)),
        "std_high_ratio": float(np.std(high_ratios)),
        "missing_extreme_folds": int(missing_extreme_folds),
        "test_size_std": float(np.std(test_sizes)),
        "narrow_range_folds": int(narrow_range_folds),
    }


def save_best_split(output_dir, seed, num_folds, trials, folds):
    split_path = os.path.join(output_dir, f"fixed_5fold_seed{seed}_balanced.json")
    split = {
        "seed": int(seed),
        "num_folds": int(num_folds),
        "subject_wise_split": True,
        "sample_count": len(trials),
        "selection_method": "label_distribution_balance_only",
        "note": "This split was selected by label distribution balance, not by test performance.",
        "folds": folds,
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    return split_path


def build_parser():
    parser = argparse.ArgumentParser(description="Screen fixed 5-fold CV seeds by label distribution balance only.")
    parser.add_argument("--seed_start", default=2020, type=int)
    parser.add_argument("--seed_end", default=2050, type=int)
    parser.add_argument("--num_folds", default=5, type=int)
    parser.add_argument("--dataset_path", default=r"C:\Users\云瑾\Desktop\data", type=str)
    parser.add_argument("--label_file", default=None, type=str)
    parser.add_argument("--compacted_dir", default=None, type=str)
    parser.add_argument("--output_dir", default=os.path.join("splits", "seed_screening"), type=str)
    parser.add_argument("--save_best", default=True, type=parse_bool)
    return parser


def main():
    args = build_parser().parse_args()
    label_file = args.label_file or os.path.join(args.dataset_path, "minor_scale_2_gai.xlsx")
    compacted_dir = args.compacted_dir or os.path.join(
        args.dataset_path,
        "Data_Processed",
        "compacted_EEG",
    )
    os.makedirs(args.output_dir, exist_ok=True)

    print("Seed screening uses label distribution balance only.")
    print("No test performance is used for seed selection.")
    print(f"label_file = {label_file}")
    print(f"compacted_dir = {compacted_dir}")
    print(f"candidate seeds = {args.seed_start}..{args.seed_end}")

    trials = load_trials(label_file, compacted_dir)
    print(f"sample_count = {len(trials)}")

    summary_rows = []
    folds_by_seed = {}
    for seed in range(args.seed_start, args.seed_end + 1):
        folds = make_split(trials, seed=seed, num_folds=args.num_folds)
        fold_rows = fold_stats_for_seed(trials, folds)
        score_info = score_seed(fold_rows)

        fold_csv = os.path.join(args.output_dir, f"seed_{seed}_fold_stats.csv")
        pd.DataFrame(fold_rows).to_csv(fold_csv, index=False, encoding="utf-8-sig")

        summary_rows.append({"seed": seed, **score_info})
        folds_by_seed[seed] = folds

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["split_balance_score", "std_of_test_label_means", "std_low_ratio"],
        ascending=[True, True, True],
    )
    summary_csv = os.path.join(args.output_dir, "seed_screening_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    best_seed = int(summary_df.iloc[0]["seed"])
    best_split_path = None
    if args.save_best:
        best_split_path = save_best_split(
            args.output_dir,
            seed=best_seed,
            num_folds=args.num_folds,
            trials=trials,
            folds=folds_by_seed[best_seed],
        )

    columns = [
        "seed",
        "split_balance_score",
        "std_of_test_label_means",
        "std_of_test_label_stds",
        "std_low_ratio",
        "std_mid_ratio",
        "std_high_ratio",
        "missing_extreme_folds",
    ]
    print("\nTop 10 seeds:")
    print(summary_df[columns].head(10).to_string(index=False))
    print("\nBest seed selected by label distribution balance only.")
    print("No test performance was used for seed selection.")
    print(f"summary_csv = {summary_csv}")
    if best_split_path:
        print(f"best_fixed_split = {best_split_path}")
        print("Use it with:")
        print(
            "python main.py --experiment_name Exp_DeltaPLI_PLIEncoder_TemporalBandEdgeAttnLite_GRU_RegOnly_LearnableLoss_lr1e4_wd5e4 "
            f"--use_fixed_split True --fixed_split_path {best_split_path}"
        )


if __name__ == "__main__":
    main()
