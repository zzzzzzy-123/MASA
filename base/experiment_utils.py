import os
import sys
import json
import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from base.dataset import (
    EEG_Fusion_Dataset,
    compute_label_scaler,
    compute_train_scaler,
)
from base.trainer import Trainer
from model.model_utils import get_model


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader_generator(seed, fold, split_offset=0):
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(fold) * 100 + int(split_offset))
    return generator


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
                direct_score = output[0]
                ordinal_logits = output[1] if len(output) > 1 else None
                expected_score = direct_score
                if label_scaler is not None and ordinal_logits is not None:
                    ordinal_score_real = torch.sum(torch.sigmoid(ordinal_logits), dim=1, keepdim=True)
                    ordinal_score = (ordinal_score_real - label_scaler['mean']) / (label_scaler['std'] + 1e-8)
                    score_fusion_alpha = None
                    if hasattr(model, 'get_score_fusion_alpha'):
                        score_fusion_alpha = model.get_score_fusion_alpha()
                    if score_fusion_alpha is None:
                        expected_score = 0.7 * direct_score + 0.3 * ordinal_score
                    else:
                        expected_score = (
                            score_fusion_alpha * direct_score
                            + (1.0 - score_fusion_alpha) * ordinal_score
                        )
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


def collect_channel_attention_weights(model, dataloader, device):
    if not hasattr(model, 'latest_channel_weights'):
        return None

    model.eval()
    weights = []
    with torch.no_grad():
        for X, _, _ in dataloader:
            X = X.to(device)
            _ = model(X)
            latest = getattr(model, 'latest_channel_weights', None)
            if latest is not None:
                weights.append(latest.detach().cpu().numpy())

    if len(weights) == 0:
        return None

    return np.concatenate(weights, axis=0)


def collect_gru_token_mixer_diagnostics(model, dataloader, device):
    if not hasattr(model, 'latest_gru_shapes'):
        return None

    model.eval()
    with torch.no_grad():
        for X, _, _ in dataloader:
            X = X.to(device)
            _ = model(X)
            shapes = getattr(model, 'latest_gru_shapes', None)
            gate = {
                'gru_gate': getattr(model, 'latest_gru_gate', None),
                'cls_gate': getattr(model, 'latest_cls_gate', None),
            }
            if shapes is not None:
                return shapes, gate
    return None


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
    device,
):
    input_mode = getattr(args, 'input_mode', 'delta_pli')
    print(f"input_mode = {input_mode}")
    sample_folder = fold_data['train'][0][0]
    feature_path = os.path.join(sample_folder, 'eeg_PLI.npy')
    print("feature_source = Delta-PLI")
    print("Delta-PLI feature loaded = True")
    print(f"feature file path = {feature_path}")
    print("feature cache name = eeg_PLI.npy")
    if os.path.exists(feature_path):
        sample_feature = np.load(feature_path).astype(np.float32)
        print(
            "delta_pli mean/std/min/max = "
            f"{np.mean(sample_feature):.6f}/{np.std(sample_feature):.6f}/"
            f"{np.min(sample_feature):.6f}/{np.max(sample_feature):.6f}"
        )

    train_scaler = compute_train_scaler(fold_data['train'], features, input_mode=input_mode)
    label_scaler = compute_label_scaler(fold_data['train'])
    print(
        f"SI label scaler(train only): mean={label_scaler['mean']:.4f}, "
        f"std={label_scaler['std']:.4f}"
    )
    print("mode = regression_only")
    print("Regression-only training = True")
    print("Auxiliary classification heads = removed")
    print("Feature scaler fit split = train only")
    print("Label scaler fit split = train only")
    print("Threshold fit split = disabled")
    seed = int(getattr(args, 'seed', 2026))
    print(f"DataLoader worker seed base = {seed}")

    dataloaders = {
        'train': DataLoader(
            EEG_Fusion_Dataset(
                fold_data['train'],
                features,
                scaler=train_scaler,
                label_scaler=label_scaler,
                input_mode=input_mode,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=build_dataloader_generator(seed, fold, 1),
        ),
        'validate': DataLoader(
            EEG_Fusion_Dataset(
                fold_data['validate'],
                features,
                scaler=train_scaler,
                label_scaler=label_scaler,
                input_mode=input_mode,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=build_dataloader_generator(seed, fold, 2),
        ),
        'test': DataLoader(
            EEG_Fusion_Dataset(
                fold_data['test'],
                features,
                scaler=train_scaler,
                label_scaler=label_scaler,
                input_mode=input_mode,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=build_dataloader_generator(seed, fold, 3),
        ),
    }

    if features == ['eeg_PLI']:
        try:
            x_pli, _, _ = next(iter(dataloaders['train']))
            print(f"x_pli shape = {list(x_pli.shape)}")
            if x_pli.dim() != 4 or x_pli.shape[1] != 1 or x_pli.shape[2] != 140 or x_pli.shape[3] != 40:
                print(f"WARNING: expected x_pli shape [B, 1, 140, 40], got {list(x_pli.shape)}")
        except Exception as shape_err:
            print(f"WARNING: failed to inspect x_pli shape: {shape_err}")

    model = get_model(
        features=features,
        num_chan=num_chan,
        thickness=thickness,
        dropout=args.cnn1d_dropout,
        model_type=getattr(args, 'model_type', 'masa_tcn'),
        experiment_name=exp_name,
        channel_attention=getattr(args, 'channel_attention', True),
        gru_gate=getattr(args, 'gru_gate', 0.05),
        gru_num_layers=getattr(args, 'gru_num_layers', 1),
        gru_bidirectional=getattr(args, 'gru_bidirectional', False),
        use_gru=getattr(args, 'use_gru', True),
        use_band_edge_attention=getattr(args, 'use_band_edge_attention', False),
        use_temporal_band_edge_attention=getattr(args, 'use_temporal_band_edge_attention', False),
        attn_scale=getattr(args, 'attn_scale', 0.1),
        gamma_band=getattr(args, 'gamma_band', 0.2),
        gamma_edge=getattr(args, 'gamma_edge', 0.2),
        band_mlp_hidden_dim=getattr(args, 'band_mlp_hidden_dim', 16),
        edge_mlp_hidden_dim=getattr(args, 'edge_mlp_hidden_dim', 64),
        attention_dropout=getattr(args, 'attention_dropout', 0.0),
        learnable_score_fusion=getattr(args, 'learnable_score_fusion', False),
        init_score_fusion_alpha=getattr(args, 'init_score_fusion_alpha', 0.7),
        loss_weighting=getattr(args, 'loss_weighting', 'fixed'),
        input_channels=1,
    ).to(device)

    if getattr(args, 'learnable_score_fusion', False) and hasattr(model, 'score_fusion_logit'):
        included_in_parameters = any(param is model.score_fusion_logit for param in model.parameters())
        print(f"fusion_logit requires_grad = {model.score_fusion_logit.requires_grad}")
        print(f"fusion_logit included in optimizer = {included_in_parameters}")

    print("MASA-TCN in_channels = 1")
    print("PLI-oriented encoder enabled = True")
    if getattr(args, 'use_temporal_band_edge_attention', False):
        print("Backbone = PLIEncoderTemporalBandEdgeAttention")
        print(f"gamma_band = {getattr(args, 'gamma_band', 0.2)}")
        print(f"gamma_edge = {getattr(args, 'gamma_edge', 0.2)}")
        print(f"BandMLP hidden_dim = {getattr(args, 'band_mlp_hidden_dim', 16)}")
        print(f"EdgeMLP hidden_dim = {getattr(args, 'edge_mlp_hidden_dim', 64)}")
        print(f"Attention dropout = {getattr(args, 'attention_dropout', 0.0)}")
        if getattr(args, 'lambda_attn', 0.0) > 0:
            print("Attention regularization enabled = True")
            print(f"lambda_attn = {getattr(args, 'lambda_attn', 0.0)}")
    elif getattr(args, 'use_band_edge_attention', False):
        print("Backbone = PLIEncoderBandEdgeAttention")
        print(f"attn_scale = {getattr(args, 'attn_scale', 0.1)}")
    else:
        print("Backbone = PLIEncoder")
    print("MASA-TCN enabled = False")
    print(f"GRU Token Mixer enabled = {getattr(args, 'use_gru', True)}")

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
        'huber_weight': getattr(args, 'huber_weight', 0.70),
        'rank_weight': getattr(args, 'rank_weight', 0.30),
        'rank_min_label_gap': getattr(args, 'rank_min_label_gap', 0.4),
        'rank_min_label_gap_real': getattr(args, 'rank_min_label_gap_real', None),
        'loss_weighting': getattr(args, 'loss_weighting', 'fixed'),
        'use_gru': getattr(args, 'use_gru', True),
        'use_band_edge_attention': getattr(args, 'use_band_edge_attention', False),
        'use_temporal_band_edge_attention': getattr(args, 'use_temporal_band_edge_attention', False),
        'attn_scale': getattr(args, 'attn_scale', 0.1),
        'gamma_band': getattr(args, 'gamma_band', 0.2),
        'gamma_edge': getattr(args, 'gamma_edge', 0.2),
        'band_mlp_hidden_dim': getattr(args, 'band_mlp_hidden_dim', 16),
        'edge_mlp_hidden_dim': getattr(args, 'edge_mlp_hidden_dim', 64),
        'attention_dropout': getattr(args, 'attention_dropout', 0.0),
        'lambda_attn': getattr(args, 'lambda_attn', 0.0),
        'learnable_score_fusion': getattr(args, 'learnable_score_fusion', False),
        'init_score_fusion_alpha': getattr(args, 'init_score_fusion_alpha', 0.7),
        'best_score_mode': getattr(args, 'best_score_mode', 'default'),
        'min_epoch': 5,
        'max_epoch': 50,
    })

    exp_logger = ExperimentLogger(args_dict['save_path'], args_dict)
    trainer = Trainer(**args_dict)

    train_loss, train_record = trainer.fit(
        dataloader_dict=dataloaders,
        checkpoint_controller=exp_logger,
        parameter_controller=exp_logger,
    )

    print(f"? {fold + 1} ?????????????: {args_dict['save_path']}")

    test_loss, test_record_dict = trainer.test(
        checkpoint_controller=exp_logger,
        dataloader_dict=dataloaders,
        epoch=None,
    )

    baseline_value, baseline_metrics = compute_mean_baseline_metrics(fold_data)

    train_preds, train_labels, train_trial_ids = collect_predictions(
        trainer.model, dataloaders['train'], device, label_scaler=label_scaler
    )
    train_prediction_csv = save_prediction_table(
        args_dict['save_path'],
        train_trial_ids,
        train_labels,
        train_preds,
        baseline_value,
        'train_predictions.csv',
    )

    preds, labels, trial_ids = collect_predictions(
        trainer.model, dataloaders['test'], device, label_scaler=label_scaler
    )
    print_prediction_distribution("Test", preds, labels)

    gru_diag = collect_gru_token_mixer_diagnostics(trainer.model, dataloaders['test'], device)
    if gru_diag is not None:
        gru_shapes, gates = gru_diag
        print("GRU Token Mixer RegOnly enabled")
        print(f"x_pli shape: {gru_shapes.get('x_pli')}")
        if gru_shapes.get('pli_oriented_encoder'):
            print("PLI-oriented encoder enabled = True")
            if gru_shapes.get('temporal_band_edge_attention'):
                print("Backbone = PLIEncoderTemporalBandEdgeAttention")
                print(f"band_gate shape: {gru_shapes.get('band_gate')}")
                print(f"edge_gate shape: {gru_shapes.get('edge_gate')}")
            elif gru_shapes.get('band_edge_attention'):
                print("Backbone = PLIEncoderBandEdgeAttention")
            else:
                print("Backbone = PLIEncoder")
        print(f"h_seq shape: {gru_shapes.get('h_seq')}")
        print(f"h_base shape: {gru_shapes.get('h_base')}")
        if gru_shapes.get('gru_enabled') is False:
            print("h_fused = h_base")
        else:
            print(f"h_gru shape: {gru_shapes.get('h_gru')}")
            print(f"h_gru_proj shape: {gru_shapes.get('h_gru_proj')}")
        print(f"h_fused shape: {gru_shapes.get('h_fused')}")
        print(f"direct_score shape: {gru_shapes.get('direct_score')}")
        print(f"ordinal_logits shape: {gru_shapes.get('ordinal_logits')}")
        print(f"expected_score shape: {gru_shapes.get('expected_score')}")
        if gru_shapes.get('learnable_score_fusion'):
            print("Score fusion: expected_score = alpha * direct_score + (1-alpha) * ordinal_score")
            print(f"fusion_logit = {gru_shapes.get('fusion_logit'):.4f}")
            print(f"fusion_alpha_direct = {gru_shapes.get('score_fusion_alpha'):.4f}")
            print(f"fusion_alpha_ordinal = {gru_shapes.get('fusion_alpha_ordinal'):.4f}")
        gru_gate = gates.get('gru_gate') if isinstance(gates, dict) else gates
        if gru_gate is not None and gru_shapes.get('gru_enabled') is not False:
            print(f"gru_gate = {gru_gate:.4f}")

    metrics = compute_regression_metrics(preds, labels)
    save_prediction_table(
        args_dict['save_path'],
        trial_ids,
        labels,
        preds,
        baseline_value,
        'test_predictions.csv',
    )
    prediction_csv = os.path.join(args_dict['save_path'], 'test_predictions.csv')
    print(f"? {fold + 1} ????????????: {train_prediction_csv}")
    print(
        f"? {fold + 1} ? Test -> "
        f"MAE: {metrics['mae']:.4f}, "
        f"RMSE: {metrics['rmse']:.4f}, "
        f"R2: {metrics['r2']:.4f}, "
        f"PCC: {metrics['pcc']:.4f}, "
        f"CCC: {metrics['ccc']:.4f}"
    )
    print(
        f"? {fold + 1} ? Mean Baseline(train mean={baseline_value:.4f}) -> "
        f"MAE: {baseline_metrics['mae']:.4f}, "
        f"RMSE: {baseline_metrics['rmse']:.4f}, "
        f"R2: {baseline_metrics['r2']:.4f}, "
        f"PCC: {baseline_metrics['pcc']:.4f}, "
        f"CCC: {baseline_metrics['ccc']:.4f}"
    )
    print(f"? {fold + 1} ????????????: {prediction_csv}")

    metrics_with_baseline = metrics.copy()
    for key in ('log_var_huber', 'log_var_rank', 'weight_huber', 'weight_rank'):
        if key in test_record_dict['overall']:
            metrics_with_baseline[key] = test_record_dict['overall'][key]
    metrics_with_baseline['input_mode'] = input_mode
    metrics_with_baseline['feature_type'] = 'Delta-PLI'
    metrics_with_baseline['input_channels'] = 1
    metrics_with_baseline['baseline_correction'] = 'task_minus_baseline'
    for key, value in baseline_metrics.items():
        metrics_with_baseline[f'baseline_{key}'] = value

    exp_logger.save_log_to_csv(epoch='test_final', test_record=metrics_with_baseline)

    try:
        import matplotlib.pyplot as plt
        plot_dir = os.path.join(args_dict['save_path'], 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(trainer.train_losses, label='Train Loss')
        plt.plot(trainer.validate_losses, label='Validation Loss')
        plt.title(f'{exp_name} Fold {fold + 1} Training Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f'training_curves_fold_{fold}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"?????????: {plot_path}")
    except Exception as plot_err:
        print(f"?????????: {plot_err}")

    return metrics_with_baseline

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
