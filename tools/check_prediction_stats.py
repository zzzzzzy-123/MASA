import argparse
import csv
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if not values:
        return 0.0
    center = mean(values)
    return math.sqrt(mean([(value - center) ** 2 for value in values]))


def pcc(xs, ys):
    if len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    sx = std(xs)
    sy = std(ys)
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    return mean([(x - mx) * (y - my) for x, y in zip(xs, ys)]) / (sx * sy)


def read_prediction_file(path):
    labels = []
    preds = []
    abs_errors = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.append(float(row["label"]))
            preds.append(float(row["pred"]))
            abs_errors.append(float(row["abs_error"]))
    return labels, preds, abs_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", nargs="?", default="Exp_Only_PLI")
    parser.add_argument("--file", default="test_predictions.csv")
    args = parser.parse_args()

    for path in sorted(Path(args.exp_dir).glob(f"fold*/{args.file}")):
        labels, preds, abs_errors = read_prediction_file(path)
        print(
            f"{path.parent.name}: n={len(labels)} "
            f"label_std={std(labels):.3f} pred_std={std(preds):.3f} "
            f"pcc={pcc(preds, labels):.3f} mae={mean(abs_errors):.3f}"
        )


if __name__ == "__main__":
    main()
