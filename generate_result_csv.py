import pandas as pd
from pathlib import Path
import numpy as np
import ast
import os
import sys
import argparse


def csv_generator(path_to_read_result, num_folds=5, **kwargs):  # 🌟 默认改为 5 折

    fold_wise_csv_filename = "training_logs.csv"
    csv_filename = os.path.join(path_to_read_result, "result.csv")

    metrics = ["rmse", "pcc", "ccc"]
    validate_result = np.zeros((len(metrics), num_folds + 2))
    test_result = np.zeros((len(metrics), num_folds + 2))

    for path in Path(path_to_read_result).rglob(fold_wise_csv_filename):
        # 安全提取 Fold 编号
        try:
            fold = int(path.parts[-2].split('_')[-2])
        except ValueError:
            continue  # 如果文件夹命名不是标准格式，跳过防报错

        print(f"正在处理: {path}")
        df = pd.read_csv(path, skiprows=4)
        last_row = df.iloc[-1, :]

        best_epoch = str(df["best_epoch"].iloc[-2])
        result_of_best_epoch = df.loc[df['epoch'] == best_epoch]

        test_rmse = float(last_row.iloc[2])
        test_pcc = float(last_row.iloc[4])
        test_ccc = float(last_row.iloc[7])

        val_rmse = float(result_of_best_epoch["val_rmse"].values[0])
        val_pcc = result_of_best_epoch["val_pcc_v"].values[0]
        val_ccc = float(result_of_best_epoch["val_ccc"].values[0])

        test_result[0, fold] = test_rmse
        test_result[1, fold] = test_pcc
        test_result[2, fold] = test_ccc

        validate_result[0, fold] = val_rmse
        validate_result[1, fold] = val_pcc
        validate_result[2, fold] = val_ccc

    test_result[:, -2] = np.mean(test_result[:, :-2], axis=1)
    test_result[:, -1] = np.std(test_result[:, :-2], axis=1)

    validate_result[:, -2] = np.mean(validate_result[:, :-2], axis=1)
    validate_result[:, -1] = np.std(validate_result[:, :-2], axis=1)

    # ==== 🌟 护法魔法：动态生成表头，再也不会维度报错了！ ====
    col_names = [str(i) for i in range(num_folds)] + ['mean', 'std']

    result_csv = pd.DataFrame(validate_result, columns=col_names, index=metrics).rename_axis("val")
    result_csv.to_csv(csv_filename)

    result_csv_test = pd.DataFrame(test_result, columns=col_names, index=metrics).rename_axis("test")
    result_csv_test.to_csv(csv_filename, mode='a')

    print(f"\n🎉 战报汇总完毕！总成绩单已保存在: {csv_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate Training Results')
    parser.add_argument(
        '-path_to_read_result',
        default=r'C:\Users\云瑾\Desktop\MASA\save',  # 🌟 换成你自己的 save 文件夹路径！
        type=str, help='The root directory of the saved models.')

    # 🌟 在命令行里也改成 5 折
    parser.add_argument('-num_folds', default=5, type=int, help='Total number of folds run')

    args = parser.parse_args()
    csv_generator(args.path_to_read_result, num_folds=args.num_folds)