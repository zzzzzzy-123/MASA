import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

from base.dataset import DataArranger
from base.experiment_utils import setup_logger, StreamToLogger, ExperimentLogger, run_single_fold, summarize_fold_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    print(f"当前 PyTorch 版本: {torch.__version__}, 是否检测到显卡: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='MASA-TCN 自动化实验')
    parser.add_argument('-dataset_path', default=r'C:\Users\云瑾\Desktop\data', type=str)
    parser.add_argument('-save_path', default=r'C:\Users\云瑾\Desktop\MASA', type=str)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-learning_rate', default=5e-4, type=float)
    parser.add_argument('-min_learning_rate', default=1e-6, type=float)
    parser.add_argument('-cnn1d_dropout', default=0.3, type=float)
    parser.add_argument('-patience', default=5, type=int)
    parser.add_argument('-factor', default=0.5, type=float)

    args = parser.parse_args()
    args_dict = vars(args)

    args_dict.update({
        'device': device,
        'criterion': nn.MSELoss(),
        'verbose': True, 
        'milestone': [],
        'load_best_at_each_epoch': True,
        'emotion': 'SI',
        'metrics': ["mae", "rmse", "r2", "pcc", "ccc"],
        'save_plot': False,
        'scheduler': 'plateau',
        'min_epoch': 5,
        'max_epoch': 50,
    })

    experiments_group_1 = [
        ("Exp_Only_DE", ['eeg_DE'], 8, 5),
        ("Exp_Only_RP", ['eeg_RP'], 8, 5),
        ("Exp_Only_FE", ['eeg_FE'], 8, 5),
        ("Exp_Only_PLI", ['eeg_PLI'], 28, 5),
        ("Exp_Power_Fusion", ['eeg_DE', 'eeg_RP', 'eeg_FE'], 8, 15),
    ]

    tasks_to_run = [("Exp_Only_PLI", ['eeg_PLI'], 28, 5)]
    save_root = str(args.save_path)

    arranger = DataArranger(dataset_path=args.dataset_path)

    for exp_name, features, num_chan, thickness in tasks_to_run:
        exp_root = os.path.join(save_root, exp_name)
        os.makedirs(exp_root, exist_ok=True)

        logger = setup_logger(exp_root, exp_name)
        sys.stdout = StreamToLogger(logger)

        print("\n" + "=" * 60)
        print(f"启动实验: {exp_name} | 特征: {features}")
        print("=" * 60)

        fold_metrics = []

        for fold in range(5):
            print(f"\n>>> Fold {fold + 1}/5 <<<")

            fold_data = arranger.get_fold(fold)

            metrics = run_single_fold(
                fold=fold,
                fold_data=fold_data,
                exp_name=exp_name,
                features=features,
                num_chan=num_chan,
                thickness=thickness,
                args=args,
                save_root=save_root,
                device=device
            )

            fold_metrics.append(metrics)

            print(
                f"Fold {fold + 1} Test -> "
                f"MAE: {metrics['mae']:.4f}, "
                f"RMSE: {metrics['rmse']:.4f}, "
                f"R2: {metrics['r2']:.4f}, "
                f"PCC: {metrics['pcc']:.4f}, "
                f"CCC: {metrics['ccc']:.4f}"
            )

        summarize_fold_metrics(exp_name, fold_metrics)

        sys.stdout = sys.__stdout__
