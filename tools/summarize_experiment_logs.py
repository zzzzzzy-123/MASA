import argparse
import csv
import math
import re
from pathlib import Path


FINAL_HEADER_RE = re.compile(
    r"[\[【]\s*(?P<name>Exp_.+?)\s*[\]】].*?(?:5\s*折|5折).*?(?:最终战报|交叉验证)",
    re.IGNORECASE,
)
FILENAME_TS_RE = re.compile(r"^(?P<name>.+?)_\d{8}_\d{6}\.log$", re.IGNORECASE)
FINAL_METRIC_RE = re.compile(
    r"(?P<metric>MAE|RMSE|R2|R²|PCC|CCC)\s*:\s*"
    r"(?P<mean>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*"
    r"(?:±|\+/-|\+-)\s*"
    r"(?P<std>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))",
    re.IGNORECASE,
)
FOLD_RE = re.compile(
    r"(?:Fold\s*(?P<fold_en>\d+)\s*Test|第\s*(?P<fold_cn>\d+)\s*折\s*Test)\s*->\s*"
    r"MAE\s*:\s*(?P<mae>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*,\s*"
    r"RMSE\s*:\s*(?P<rmse>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*,\s*"
    r"(?:R2|R²)\s*:\s*(?P<r2>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*,\s*"
    r"PCC\s*:\s*(?P<pcc>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*,\s*"
    r"CCC\s*:\s*(?P<ccc>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))",
    re.IGNORECASE,
)
PRED_DIST_RE = re.compile(
    r"Test pred min/max/mean/std\s*:\s*"
    r"(?P<min>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*/\s*"
    r"(?P<max>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*/\s*"
    r"(?P<mean>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*/\s*"
    r"(?P<std>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))",
    re.IGNORECASE,
)
LABEL_DIST_RE = re.compile(
    r"Test label min/max/mean/std\s*:\s*"
    r"(?P<min>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*/\s*"
    r"(?P<max>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*/\s*"
    r"(?P<mean>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*/\s*"
    r"(?P<std>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))",
    re.IGNORECASE,
)
RISK_RE = re.compile(
    r"Risk Head\s*->\s*acc\s*:\s*(?P<acc>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*,\s*"
    r"f1\s*:\s*(?P<f1>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*,\s*"
    r"low_pred_mean\s*:\s*(?P<low>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*,\s*"
    r"high_pred_mean\s*:\s*(?P<high>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))\s*,\s*"
    r"sep\s*:\s*(?P<sep>[-+]?(?:nan|inf|\d+(?:\.\d+)?|\.\d+))",
    re.IGNORECASE,
)


SUMMARY_FIELDS = [
    "experiment_name",
    "log_file",
    "mae_mean",
    "mae_std",
    "rmse_mean",
    "rmse_std",
    "r2_mean",
    "r2_std",
    "pcc_mean",
    "pcc_std",
    "ccc_mean",
    "ccc_std",
    "pred_std",
    "risk_acc",
    "risk_f1",
    "sep",
]

FOLD_FIELDS = [
    "experiment_name",
    "log_file",
    "fold",
    "mae",
    "rmse",
    "r2",
    "pcc",
    "ccc",
    "pred_std",
    "label_std",
    "risk_acc",
    "risk_f1",
    "low_pred_mean",
    "high_pred_mean",
    "sep",
]


def parse_number(value):
    value = str(value).strip().lower()
    if value in {"nan", "+nan", "-nan"}:
        return math.nan
    if value in {"inf", "+inf"}:
        return math.inf
    if value == "-inf":
        return -math.inf
    return float(value)


def sort_value(row, key, reverse=False):
    value = row.get(key, "")
    if value == "" or value is None:
        return -math.inf if reverse else math.inf
    if isinstance(value, float) and math.isnan(value):
        return -math.inf if reverse else math.inf
    return value


def mean_or_blank(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return ""
    return sum(clean) / len(clean)


def read_text(path):
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def infer_experiment_name(path, text):
    matches = list(FINAL_HEADER_RE.finditer(text))
    if matches:
        return matches[-1].group("name").strip()
    filename_match = FILENAME_TS_RE.match(path.name)
    if filename_match:
        return filename_match.group("name")
    return path.stem


def parse_log(path):
    text = read_text(path)
    experiment_name = infer_experiment_name(path, text)

    header_matches = list(FINAL_HEADER_RE.finditer(text))
    final_text = text[header_matches[-1].end():] if header_matches else ""

    metrics = {}
    for match in FINAL_METRIC_RE.finditer(final_text):
        metric = match.group("metric").lower().replace("²", "2")
        metrics[f"{metric}_mean"] = parse_number(match.group("mean"))
        metrics[f"{metric}_std"] = parse_number(match.group("std"))

    required = [
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "r2_std",
        "pcc_mean",
        "pcc_std",
        "ccc_mean",
        "ccc_std",
    ]
    if not all(key in metrics for key in required):
        missing = ", ".join(key for key in required if key not in metrics)
        return None, [], f"missing final metrics: {missing}"

    fold_rows = []
    fold_context = {}
    current_fold = None

    for line in text.splitlines():
        fold_match = FOLD_RE.search(line)
        if fold_match:
            current_fold = int(fold_match.group("fold_en") or fold_match.group("fold_cn"))
            fold_row = {
                "experiment_name": experiment_name,
                "log_file": str(path),
                "fold": current_fold,
                "mae": parse_number(fold_match.group("mae")),
                "rmse": parse_number(fold_match.group("rmse")),
                "r2": parse_number(fold_match.group("r2")),
                "pcc": parse_number(fold_match.group("pcc")),
                "ccc": parse_number(fold_match.group("ccc")),
            }
            fold_row.update(fold_context.get(current_fold, {}))
            fold_rows.append(fold_row)
            continue

        pred_match = PRED_DIST_RE.search(line)
        if pred_match and current_fold is not None:
            fold_context.setdefault(current_fold, {})["pred_std"] = parse_number(pred_match.group("std"))
            continue

        label_match = LABEL_DIST_RE.search(line)
        if label_match and current_fold is not None:
            fold_context.setdefault(current_fold, {})["label_std"] = parse_number(label_match.group("std"))
            continue

        risk_match = RISK_RE.search(line)
        if risk_match and current_fold is not None:
            fold_context.setdefault(current_fold, {}).update(
                {
                    "risk_acc": parse_number(risk_match.group("acc")),
                    "risk_f1": parse_number(risk_match.group("f1")),
                    "low_pred_mean": parse_number(risk_match.group("low")),
                    "high_pred_mean": parse_number(risk_match.group("high")),
                    "sep": parse_number(risk_match.group("sep")),
                }
            )

    # Patch context that appeared before the fold result line.
    for row in fold_rows:
        row.update(fold_context.get(row["fold"], {}))

    summary = {
        "experiment_name": experiment_name,
        "log_file": str(path),
        **metrics,
        "pred_std": mean_or_blank([row.get("pred_std") for row in fold_rows]),
        "risk_acc": mean_or_blank([row.get("risk_acc") for row in fold_rows]),
        "risk_f1": mean_or_blank([row.get("risk_f1") for row in fold_rows]),
        "sep": mean_or_blank([row.get("sep") for row in fold_rows]),
    }

    return summary, fold_rows, None


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(path, summary_rows, fold_rows):
    try:
        import pandas as pd
    except ImportError:
        print("未安装 pandas，跳过 Excel 输出。CSV 已正常生成。")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(fold_rows).to_excel(writer, sheet_name="folds", index=False)


def print_top(title, rows, key, reverse=False):
    print(f"\n{title}")
    ranked = sorted(rows, key=lambda row: sort_value(row, key, reverse=reverse), reverse=reverse)
    for idx, row in enumerate(ranked[:10], start=1):
        mae = row.get("mae_mean", "")
        pcc = row.get("pcc_mean", "")
        ccc = row.get("ccc_mean", "")
        print(
            f"{idx:2d}. {row['experiment_name']} | "
            f"MAE={mae} | PCC={pcc} | CCC={ccc} | {Path(row['log_file']).name}"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize MASA experiment log files.")
    parser.add_argument("--log_dir", required=True, help="Directory to recursively scan for .log files.")
    parser.add_argument("--out_csv", required=True, help="Output summary CSV path.")
    parser.add_argument("--out_xlsx", default=None, help="Optional Excel output path.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_files = sorted(log_dir.rglob("*.log"))

    summary_rows = []
    fold_rows = []
    skipped = []

    for log_file in log_files:
        summary, folds, reason = parse_log(log_file)
        if reason:
            skipped.append((str(log_file), reason))
            continue
        summary_rows.append(summary)
        fold_rows.extend(folds)

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            sort_value(row, "mae_mean"),
            sort_value(row, "rmse_mean"),
            -sort_value(row, "pcc_mean", reverse=True),
            -sort_value(row, "ccc_mean", reverse=True),
        ),
    )

    out_csv = Path(args.out_csv)
    out_folds = out_csv.with_name(out_csv.stem.replace("summary", "folds") + out_csv.suffix)
    skipped_path = out_csv.with_name("skipped_logs.txt")

    write_csv(out_csv, summary_rows, SUMMARY_FIELDS)
    write_csv(out_folds, fold_rows, FOLD_FIELDS)

    if skipped:
        skipped_path.write_text(
            "\n".join(f"{path}\t{reason}" for path, reason in skipped),
            encoding="utf-8",
        )
    else:
        skipped_path.write_text("", encoding="utf-8")

    if args.out_xlsx:
        write_xlsx(args.out_xlsx, summary_rows, fold_rows)

    print(f"扫描日志: {len(log_files)}")
    print(f"成功汇总: {len(summary_rows)}")
    print(f"跳过日志: {len(skipped)} -> {skipped_path}")
    print(f"主汇总表: {out_csv}")
    print(f"每折明细: {out_folds}")

    print_top("按 MAE 最低排序 Top 10", summary_rows, "mae_mean", reverse=False)
    print_top("按 PCC 最高排序 Top 10", summary_rows, "pcc_mean", reverse=True)
    print_top("按 CCC 最高排序 Top 10", summary_rows, "ccc_mean", reverse=True)


if __name__ == "__main__":
    main()
