import argparse
import csv
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    center = mean(values)
    return math.sqrt(mean([(value - center) ** 2 for value in values])) if values else 0.0


def metric_row(labels, preds):
    errors = [pred - label for pred, label in zip(preds, labels)]
    abs_errors = [abs(error) for error in errors]
    ss_res = sum(error * error for error in errors)
    label_mean = mean(labels)
    ss_tot = sum((label - label_mean) ** 2 for label in labels)

    pred_mean = mean(preds)
    pred_std = std(preds)
    label_std = std(labels)
    if pred_std > 1e-8 and label_std > 1e-8:
        cov = mean([(pred - pred_mean) * (label - label_mean) for pred, label in zip(preds, labels)])
        pcc = cov / (pred_std * label_std)
        ccc = (2 * cov) / (
            pred_std ** 2 + label_std ** 2 + (pred_mean - label_mean) ** 2 + 1e-8
        )
    else:
        pcc = 0.0
        ccc = 0.0

    return {
        "mae": mean(abs_errors),
        "rmse": math.sqrt(mean([error * error for error in errors])),
        "r2": 1 - ss_res / (ss_tot + 1e-8),
        "pcc": pcc,
        "ccc": ccc,
        "label_mean": label_mean,
        "pred_mean": pred_mean,
        "label_std": label_std,
        "pred_std": pred_std,
    }


def read_predictions(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = [float(row["label"]) for row in rows]
    preds = [float(row["pred"]) for row in rows]
    baseline = [float(row["baseline_pred"]) for row in rows]
    return labels, preds, baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--file", default="test_predictions.csv")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    out_path = exp_dir / f"{Path(args.file).stem}_summary.csv"
    fieldnames = [
        "fold",
        "kind",
        "mae",
        "rmse",
        "r2",
        "pcc",
        "ccc",
        "label_mean",
        "pred_mean",
        "label_std",
        "pred_std",
    ]

    rows = []
    for path in sorted(exp_dir.glob(f"fold*/{args.file}")):
        labels, preds, baseline = read_predictions(path)
        model_row = metric_row(labels, preds)
        model_row.update({"fold": path.parent.name, "kind": "model"})
        rows.append(model_row)

        baseline_row = metric_row(labels, baseline)
        baseline_row.update({"fold": path.parent.name, "kind": "baseline"})
        rows.append(baseline_row)

    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {out_path}")
    for kind in ["model", "baseline"]:
        subset = [row for row in rows if row["kind"] == kind]
        print(kind)
        for name in ["mae", "rmse", "r2", "pcc", "ccc", "label_std", "pred_std"]:
            values = [row[name] for row in subset]
            print(f"  {name}: {mean(values):.4f} +/- {std(values):.4f}")


if __name__ == "__main__":
    main()
