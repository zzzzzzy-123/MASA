import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_predictions(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = [float(row["label"]) for row in rows]
    preds = [float(row["pred"]) for row in rows]
    return labels, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--file", default="test_predictions.csv")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = exp_dir / "scatter"
    out_dir.mkdir(exist_ok=True)

    for path in sorted(exp_dir.glob(f"fold*/{args.file}")):
        labels, preds = read_predictions(path)
        lo = min(labels + preds)
        hi = max(labels + preds)

        plt.figure(figsize=(5, 5))
        plt.scatter(labels, preds, alpha=0.75, edgecolors="none")
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title(path.parent.name)
        plt.tight_layout()

        out_path = out_dir / f"{path.parent.name}_{Path(args.file).stem}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
