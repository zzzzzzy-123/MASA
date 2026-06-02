import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    'fold',
    'train_label_mean', 'train_label_std', 'train_label_min', 'train_label_max',
    'val_label_mean', 'val_label_std', 'val_label_min', 'val_label_max',
    'test_label_mean', 'test_label_std', 'test_label_min', 'test_label_max',
    'test_low_count', 'test_mid_count', 'test_high_count',
    'test_pred_mean', 'test_pred_std',
    'low_pred_mean', 'high_pred_mean', 'sep',
    'test_mae', 'test_rmse', 'test_r2', 'test_pcc', 'test_ccc',
]


def resolve_input_path(input_path):
    path = Path(input_path)
    if path.is_dir():
        path = path / 'fold_diagnostics.csv'
    if not path.exists():
        raise FileNotFoundError(f"Cannot find fold diagnostics csv: {path}")
    return path


def read_diagnostics(path):
    df = pd.read_csv(path)
    missing = [col for col in SUMMARY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df['fold'] = df['fold'].astype(int)
    for col in SUMMARY_COLUMNS:
        if col != 'fold':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('fold')


def save_summary(df, out_dir):
    out_path = out_dir / 'fold_diagnostics_summary.csv'
    df[SUMMARY_COLUMNS].to_csv(out_path, index=False, encoding='utf-8-sig')
    return out_path


def print_fold_label(fold_idx):
    return f"fold {fold_idx} / 第 {fold_idx + 1} 折"


def print_findings(df):
    overall_test_mean = float(df['test_label_mean'].mean())
    mean_shift = (df['test_label_mean'] - overall_test_mean).abs()
    max_shift_row = df.loc[mean_shift.idxmax()]
    min_std_row = df.loc[df['test_label_std'].idxmin()]

    group_min = df[['test_low_count', 'test_high_count']].min(axis=1)
    min_group_row = df.loc[group_min.idxmin()]

    print("\nFold diagnostics summary")
    print(f"- 整体 test_label_mean 平均值: {overall_test_mean:.4f}")
    print(
        f"- test_label_mean 与整体平均差异最大: {print_fold_label(int(max_shift_row['fold']))}, "
        f"test_label_mean={max_shift_row['test_label_mean']:.4f}, "
        f"diff={abs(max_shift_row['test_label_mean'] - overall_test_mean):.4f}"
    )
    print(
        f"- test_label_std 最小: {print_fold_label(int(min_std_row['fold']))}, "
        f"test_label_std={min_std_row['test_label_std']:.4f}"
    )
    print(
        f"- test low/high 数量最少: {print_fold_label(int(min_group_row['fold']))}, "
        f"low={int(min_group_row['test_low_count'])}, high={int(min_group_row['test_high_count'])}"
    )

    neg_sep = df[df['sep'] < 0]
    if len(neg_sep) > 0:
        print("- sep 为负的 fold:")
        for _, row in neg_sep.iterrows():
            print(
                f"  {print_fold_label(int(row['fold']))}: "
                f"sep={row['sep']:.4f}, low_pred_mean={row['low_pred_mean']:.4f}, "
                f"high_pred_mean={row['high_pred_mean']:.4f}"
            )
    else:
        print("- 没有 sep 为负的 fold。")

    neg_corr = df[(df['test_pcc'] < 0) | (df['test_ccc'] < 0)]
    if len(neg_corr) > 0:
        print("- PCC/CCC 为负的 fold:")
        for _, row in neg_corr.iterrows():
            print(
                f"  {print_fold_label(int(row['fold']))}: "
                f"PCC={row['test_pcc']:.4f}, CCC={row['test_ccc']:.4f}"
            )
    else:
        print("- 没有 PCC/CCC 为负的 fold。")

    suspicious = set()
    for _, row in neg_sep.iterrows():
        suspicious.add(int(row['fold']))
    for _, row in neg_corr.iterrows():
        suspicious.add(int(row['fold']))
    suspicious.add(int(max_shift_row['fold']))
    suspicious.add(int(min_group_row['fold']))

    if 4 in suspicious:
        row = df[df['fold'] == 4].iloc[0]
        reasons = []
        if abs(row['test_label_mean'] - overall_test_mean) >= float(mean_shift.mean()):
            reasons.append("test label 分布偏移")
        if row['test_low_count'] == group_min.min() or row['test_high_count'] == group_min.min():
            reasons.append("low/high 样本数量不足")
        if row['sep'] < 0:
            reasons.append("预测方向反转(high_pred_mean < low_pred_mean)")
        if row['test_pcc'] < 0 or row['test_ccc'] < 0:
            reasons.append("PCC/CCC 为负，排序或一致性异常")
        if len(reasons) == 0:
            reasons.append("该折被多个诊断规则标记，建议重点查看样本分布")
        print(f"- 第 5 折异常提示: {'；'.join(reasons)}。")

    fold4 = df[df['fold'] == 3]
    if len(fold4) > 0:
        row = fold4.iloc[0]
        reasons = []
        if row['sep'] < 0:
            reasons.append("预测方向反转")
        if row['test_pcc'] < 0 or row['test_ccc'] < 0:
            reasons.append("PCC/CCC 为负")
        if row['test_low_count'] <= df['test_low_count'].quantile(0.25) or row['test_high_count'] <= df['test_high_count'].quantile(0.25):
            reasons.append("low/high 数量相对偏少")
        if reasons:
            print(f"- 第 4 折异常提示: {'；'.join(reasons)}。")


def ensure_plot_backend():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save_plots(df, out_dir):
    plt = ensure_plot_backend()
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(df))
    fold_labels = [str(int(fold) + 1) for fold in df['fold']]

    fig, ax = plt.subplots(figsize=(10, 5))
    for split, offset in [('train', -0.18), ('val', 0.0), ('test', 0.18)]:
        ax.errorbar(
            x + offset,
            df[f'{split}_label_mean'],
            yerr=df[f'{split}_label_std'],
            fmt='o-',
            capsize=4,
            label=split,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Label mean +/- std')
    ax.set_title('Fold Label Distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / 'fold_label_distribution.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    ax.bar(x - width, df['test_low_count'], width=width, label='low')
    ax.bar(x, df['test_mid_count'], width=width, label='mid')
    ax.bar(x + width, df['test_high_count'], width=width, label='high')
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Count')
    ax.set_title('Fold Test Group Counts')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / 'fold_test_group_counts.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(x, df['test_mae'], 'o-', label='MAE')
    axes[0].plot(x, df['test_rmse'], 'o-', label='RMSE')
    axes[0].set_ylabel('Error')
    axes[0].legend()

    axes[1].plot(x, df['test_r2'], 'o-', label='R2')
    axes[1].axhline(0, color='gray', linewidth=1, linestyle='--')
    axes[1].set_ylabel('R2')
    axes[1].legend()

    axes[2].plot(x, df['test_pcc'], 'o-', label='PCC')
    axes[2].plot(x, df['test_ccc'], 'o-', label='CCC')
    axes[2].axhline(0, color='gray', linewidth=1, linestyle='--')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(fold_labels)
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('Correlation')
    axes[2].legend()

    fig.suptitle('Fold Metrics')
    fig.tight_layout()
    fig.savefig(plots_dir / 'fold_metrics.png', dpi=160)
    plt.close(fig)

    return [
        plots_dir / 'fold_label_distribution.png',
        plots_dir / 'fold_test_group_counts.png',
        plots_dir / 'fold_metrics.png',
    ]


def main():
    parser = argparse.ArgumentParser(description='Analyze MASA fold diagnostics.')
    parser.add_argument(
        '--input_csv',
        required=True,
        help='Path to fold_diagnostics.csv, or an experiment directory containing it.',
    )
    parser.add_argument(
        '--out_dir',
        default=None,
        help='Output directory. Defaults to the input CSV parent directory.',
    )
    args = parser.parse_args()

    input_csv = resolve_input_path(args.input_csv)
    out_dir = Path(args.out_dir) if args.out_dir else input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_diagnostics(input_csv)
    summary_path = save_summary(df, out_dir)
    plot_paths = save_plots(df, out_dir)

    print(f"读取: {input_csv}")
    print(f"汇总 CSV: {summary_path}")
    for path in plot_paths:
        print(f"图已保存: {path}")
    print_findings(df)


if __name__ == '__main__':
    main()
