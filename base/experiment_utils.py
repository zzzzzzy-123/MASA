import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from base.dataset import EEG_Fusion_Dataset, compute_label_scaler, compute_train_scaler
from base.trainer import Trainer
from model.model_utils import get_model


def setup_logger(save_path, exp_name):
    import logging, sys
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{exp_name}_{timestamp}.log"
    log_file_path = os.path.join(save_path, log_filename)

    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_format = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(file_format)
    logger.addHandler(stream_handler)

    return logger


class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


class ExperimentLogger:
    def __init__(self, save_path, args_dict):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.csv_path = os.path.join(self.save_path, "training_log.csv")
        self.log_data = []
        self._save_config(args_dict)

    def _save_config(self, args_dict):
        clean_dict = {}
        for k, v in args_dict.items():
            if isinstance(v, (int, float, str, bool, list, tuple, dict)):
                clean_dict[k] = v
            else:
                clean_dict[k] = str(v)

        config_path = os.path.join(self.save_path, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(clean_dict, f, indent=4, ensure_ascii=False)

    def save_log_to_csv(self, epoch, mean_train_record=None, mean_validate_record=None, test_record=None):
        row = {'epoch': epoch}

        if mean_train_record:
            for k, v in mean_train_record.items():
                row[f'train_{k}'] = v[0] if isinstance(v, list) else v

        if mean_validate_record:
            for k, v in mean_validate_record.items():
                row[f'val_{k}'] = v[0] if isinstance(v, list) else v

        if test_record:
            for k, v in test_record.items():
                row[f'test_{k}'] = v[0] if isinstance(v, list) else v

        self.log_data.append(row)
        df = pd.DataFrame(self.log_data)
        df.to_csv(self.csv_path, index=False, encoding='utf-8')

    def save_checkpoint(self, trainer, parameter_controller, save_path):
        pass


def extract_scalar_pcc(pcc_value):
    if isinstance(pcc_value, list):
        return float(pcc_value[0])
    if isinstance(pcc_value, tuple):
        return float(pcc_value[0])
    return float(pcc_value)


def compute_regression_metrics(preds, labels):
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    mae = float(np.mean(np.abs(preds - labels)))
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))

    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    if len(preds) > 1 and np.std(preds) > 1e-8 and np.std(labels) > 1e-8:
        pcc, p_value = pearsonr(preds, labels)
        pcc = float(pcc)
        p_value = float(p_value)

        mean_p = np.mean(preds)
        mean_l = np.mean(labels)
        var_p = np.var(preds)
        var_l = np.var(labels)
        cov = np.mean((preds - mean_p) * (labels - mean_l))
        ccc = float((2 * cov) / (var_p + var_l + (mean_p - mean_l) ** 2 + 1e-8))
    else:
        pcc = 0.0
        p_value = 1.0
        ccc = 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pcc': pcc,
        'p_value': p_value,
        'ccc': ccc
    }


def collect_predictions(model, dataloader, device, label_scaler=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_trial_ids = []

    with torch.no_grad():
        for X, Y, trial_ids in dataloader:
            X = X.to(device)

            output = model(X)

            if isinstance(output, tuple):
                if len(output) == 3:
                    direct_score, ordinal_logits, _ = output
                else:
                    ordinal_logits, direct_score = output
                expected_score = direct_score
                if label_scaler is not None:
                    ordinal_score_real = torch.sum(torch.sigmoid(ordinal_logits), dim=1, keepdim=True)
                    ordinal_score = (ordinal_score_real - label_scaler['mean']) / (label_scaler['std'] + 1e-8)
                    expected_score = 0.7 * direct_score + 0.3 * ordinal_score
            else:
                expected_score = output

            preds = expected_score.detach().cpu().numpy().reshape(-1)
            labels = Y.detach().cpu().numpy().reshape(-1)

            if label_scaler is not None:
                preds = preds * label_scaler['std'] + label_scaler['mean']
                labels = labels * label_scaler['std'] + label_scaler['mean']

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_trial_ids.extend(list(trial_ids))

    return np.array(all_preds), np.array(all_labels), all_trial_ids


def fit_validation_calibrator(preds, labels):
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)

    pred_std = float(np.std(preds))
    label_std = float(np.std(labels))
    if len(preds) < 3 or pred_std < 0.05 or label_std < 1e-6:
        return 1.0, 0.0, False

    pcc_value = compute_regression_metrics(preds, labels)['pcc']
    if pcc_value <= 0.05:
        return 1.0, 0.0, False

    cov = float(np.mean((preds - np.mean(preds)) * (labels - np.mean(labels))))
    slope = cov / (float(np.var(preds)) + 1e-8)
    slope = float(np.clip(slope, 0.0, 20.0))
    intercept = float(np.mean(labels) - slope * np.mean(preds))

    return slope, intercept, True


def apply_calibrator(preds, slope, intercept):
    return np.asarray(preds).reshape(-1) * slope + intercept


def print_prediction_distribution(name, preds, labels):
    print(
        f"{name} pred min/max/mean/std: "
        f"{np.min(preds):.4f} / {np.max(preds):.4f} / {np.mean(preds):.4f} / {np.std(preds):.4f}"
    )
    print(
        f"{name} label min/max/mean/std: "
        f"{np.min(labels):.4f} / {np.max(labels):.4f} / {np.mean(labels):.4f} / {np.std(labels):.4f}"
    )


def compute_mean_baseline_metrics(fold_data):
    train_labels = np.asarray([item[1] for item in fold_data['train']], dtype=np.float32)
    test_labels = np.asarray([item[1] for item in fold_data['test']], dtype=np.float32)
    baseline_value = float(np.mean(train_labels))
    baseline_preds = np.full_like(test_labels, baseline_value, dtype=np.float32)

    return baseline_value, compute_regression_metrics(baseline_preds, test_labels)


def save_prediction_table(save_path, trial_ids, labels, preds, baseline_value, filename):
    os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame({
        'trial_id': trial_ids,
        'label': labels.reshape(-1),
        'pred': preds.reshape(-1),
    })
    df['abs_error'] = np.abs(df['pred'] - df['label'])
    df['baseline_pred'] = baseline_value
    df['baseline_abs_error'] = np.abs(df['baseline_pred'] - df['label'])

    csv_path = os.path.join(save_path, filename)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return csv_path


def print_split_info(fold_data):
    train_ids = [x[2] if len(x) == 3 and isinstance(x[2], str) else x[1].get('trial_id') for x in fold_data['train']]
    val_ids = [x[2] if len(x) == 3 and isinstance(x[2], str) else x[1].get('trial_id') for x in fold_data['validate']]
    test_ids = [x[2] if len(x) == 3 and isinstance(x[2], str) else x[1].get('trial_id') for x in fold_data['test']]

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    print(f"训练集样本数: {len(train_ids)}")
    print(f"验证集样本数: {len(val_ids)}")
    print(f"测试集样本数: {len(test_ids)}")

    assert train_set.isdisjoint(val_set), "训练集和验证集存在被试重叠"
    assert train_set.isdisjoint(test_set), "训练集和测试集存在被试重叠"
    assert val_set.isdisjoint(test_set), "验证集和测试集存在被试重叠"

    print("被试级划分检查通过：train / validate / test 无重叠。")


def run_single_fold(
    fold,
    fold_data,
    exp_name,
    features,
    num_chan,
    thickness,
    args,
    save_root,
    device
):
    print_split_info(fold_data)

    train_scaler = compute_train_scaler(fold_data['train'], features)
    label_scaler = compute_label_scaler(fold_data['train'])
    print(
        f"SI label scaler(train only): mean={label_scaler['mean']:.4f}, "
        f"std={label_scaler['std']:.4f}"
    )

    dataloaders = {
        'train': DataLoader(
            EEG_Fusion_Dataset(fold_data['train'], features, scaler=train_scaler, label_scaler=label_scaler),
            batch_size=args.batch_size,
            shuffle=True
        ),
        'validate': DataLoader(
            EEG_Fusion_Dataset(fold_data['validate'], features, scaler=train_scaler, label_scaler=label_scaler),
            batch_size=args.batch_size,
            shuffle=False
        ),
        'test': DataLoader(
            EEG_Fusion_Dataset(fold_data['test'], features, scaler=train_scaler, label_scaler=label_scaler),
            batch_size=args.batch_size,
            shuffle=False
        )
    }

    model = get_model(
        features=features,
        num_chan=num_chan,
        thickness=thickness,
        dropout=args.cnn1d_dropout
    ).to(device)

    args_dict = vars(args).copy()
    args_dict.update({
        'device': device,
        'criterion': torch.nn.MSELoss(),
        'verbose': True,
        'milestone': [],
        'load_best_at_each_epoch': True,
        'emotion': 'SI',
        'metrics': ["mae", "rmse", "r2", "pcc", "ccc"],
        'save_plot': False,
        'scheduler': 'plateau',
        'model': model,
        'model_name': exp_name,
        'save_path': os.path.join(save_root, exp_name, f"fold{fold}"),
        'fold': fold,
        'label_mean': label_scaler['mean'],
        'label_std': label_scaler['std'],
        'risk_low': label_scaler['risk_low'],
        'risk_high': label_scaler['risk_high'],
        'min_epoch': 5,
        'max_epoch': 50
    })

    exp_logger = ExperimentLogger(args_dict['save_path'], args_dict)
    trainer = Trainer(**args_dict)

    train_loss, train_record = trainer.fit(
        dataloader_dict=dataloaders,
        checkpoint_controller=exp_logger,
        parameter_controller=exp_logger
    )

    print(f"第 {fold + 1} 折训练结束，模型权重保存至: {args_dict['save_path']}")

    test_loss, test_record_dict = trainer.test(
        checkpoint_controller=exp_logger,
        dataloader_dict=dataloaders,
        epoch=None
    )

    baseline_value, baseline_metrics = compute_mean_baseline_metrics(fold_data)
    val_preds, val_labels, _ = collect_predictions(
        trainer.model, dataloaders['validate'], device, label_scaler=label_scaler
    )
    calib_slope, calib_intercept, calib_enabled = fit_validation_calibrator(val_preds, val_labels)
    apply_calibration_outputs = False
    print(
        f"Validation calibrator: fitted={calib_enabled}, applied={apply_calibration_outputs}, "
        f"slope={calib_slope:.4f}, intercept={calib_intercept:.4f}"
    )

    train_preds, train_labels, train_trial_ids = collect_predictions(
        trainer.model, dataloaders['train'], device, label_scaler=label_scaler
    )
    save_prediction_table(
        args_dict['save_path'],
        train_trial_ids,
        train_labels,
        apply_calibrator(train_preds, calib_slope, calib_intercept) if apply_calibration_outputs else train_preds,
        baseline_value,
        'calibrated_train_predictions.csv'
    )
    train_prediction_csv = save_prediction_table(
        args_dict['save_path'],
        train_trial_ids,
        train_labels,
        train_preds,
        baseline_value,
        'train_predictions.csv'
    )

    preds, labels, trial_ids = collect_predictions(
        trainer.model, dataloaders['test'], device, label_scaler=label_scaler
    )
    print_prediction_distribution("Test", preds, labels)
    metrics = compute_regression_metrics(preds, labels)
    risk_acc = test_record_dict['overall'].get('risk_acc', 0.0)
    risk_f1 = test_record_dict['overall'].get('risk_f1', 0.0)
    low_pred_mean = test_record_dict['overall'].get('low_pred_mean', 0.0)
    high_pred_mean = test_record_dict['overall'].get('high_pred_mean', 0.0)
    print(
        f"Risk Head -> acc: {risk_acc:.4f}, f1: {risk_f1:.4f}, "
        f"low_pred_mean: {low_pred_mean:.4f}, high_pred_mean: {high_pred_mean:.4f}, "
        f"sep: {high_pred_mean - low_pred_mean:.4f}"
    )
    save_prediction_table(
        args_dict['save_path'],
        trial_ids,
        labels,
        apply_calibrator(preds, calib_slope, calib_intercept) if apply_calibration_outputs else preds,
        baseline_value,
        'calibrated_test_predictions.csv'
    )
    prediction_csv = save_prediction_table(
        args_dict['save_path'],
        trial_ids,
        labels,
        preds,
        baseline_value,
        'test_predictions.csv'
    )
    print(f"第 {fold + 1} 折训练集逐样本预测已保存: {train_prediction_csv}")

    print(
        f"第 {fold + 1} 折 Test -> "
        f"MAE: {metrics['mae']:.4f}, "
        f"RMSE: {metrics['rmse']:.4f}, "
        f"R2: {metrics['r2']:.4f}, "
        f"PCC: {metrics['pcc']:.4f}, "
        f"CCC: {metrics['ccc']:.4f}"
    )
    print(
        f"第 {fold + 1} 折 Mean Baseline(train mean={baseline_value:.4f}) -> "
        f"MAE: {baseline_metrics['mae']:.4f}, "
        f"RMSE: {baseline_metrics['rmse']:.4f}, "
        f"R2: {baseline_metrics['r2']:.4f}, "
        f"PCC: {baseline_metrics['pcc']:.4f}, "
        f"CCC: {baseline_metrics['ccc']:.4f}"
    )
    print(f"第 {fold + 1} 折测试集逐样本预测已保存: {prediction_csv}")

    metrics_with_baseline = metrics.copy()
    metrics_with_baseline['risk_acc'] = risk_acc
    metrics_with_baseline['risk_f1'] = risk_f1
    metrics_with_baseline['low_pred_mean'] = low_pred_mean
    metrics_with_baseline['high_pred_mean'] = high_pred_mean
    for key, value in baseline_metrics.items():
        metrics_with_baseline[f'baseline_{key}'] = value

    exp_logger.save_log_to_csv(
        epoch='test_final',
        test_record=metrics_with_baseline
    )

    try:
        from base.utils import plot_training_curves

        curve_save_dir = os.path.join(args_dict['save_path'], 'plots')
        os.makedirs(curve_save_dir, exist_ok=True)
        plot_training_curves(train_record, curve_save_dir, fold)
        print(f"第 {fold + 1} 折训练曲线已生成。")
    except Exception as e:
        print(f"训练曲线生成失败，已跳过: {e}")

    return metrics


def summarize_fold_metrics(exp_name, fold_metrics):
    print(f"\n【{exp_name}】5折交叉验证最终战报")

    if len(fold_metrics) == 0:
        print("没有可汇总的 fold 指标。")
        return

    metric_names = ['mae', 'rmse', 'r2', 'pcc', 'ccc']

    summary_rows = []

    for name in metric_names:
        values = [m[name] for m in fold_metrics if name in m]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))

        summary_rows.append({
            'metric': name,
            'mean': mean_value,
            'std': std_value
        })

        print(f"{name.upper()}: {mean_value:.4f} ± {std_value:.4f}")

    print("=" * 60 + "\n")

    return summary_rows
