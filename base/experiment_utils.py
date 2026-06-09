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


def evaluate_split(model, dataloader, split_name, device, label_scaler=None):
    """Evaluate a full split with one shared, eval-mode prediction path."""
    preds, labels, trial_ids = collect_predictions(
        model,
        dataloader,
        device,
        label_scaler=label_scaler,
    )
    metrics = compute_regression_metrics(preds, labels)
    metrics.update({
        "split": split_name,
        "pred_mean": float(np.mean(preds)),
        "pred_std": float(np.std(preds)),
        "pred_min": float(np.min(preds)),
        "pred_max": float(np.max(preds)),
        "label_mean": float(np.mean(labels)),
        "label_std": float(np.std(labels)),
        "label_min": float(np.min(labels)),
        "label_max": float(np.max(labels)),
    })
    return metrics, preds, labels, trial_ids


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
                'tct_gate': getattr(model, 'latest_tct_gate', None),
                'tcn_gate': getattr(model, 'latest_tcn_gate', None),
                'parallel_tcn_gate': getattr(model, 'latest_parallel_tcn_gate', None),
                'graph_gate': getattr(model, 'latest_graph_gate', None),
            }
            if hasattr(model, "get_multi_pli_gate_values"):
                multi_gate_values = model.get_multi_pli_gate_values()
                if multi_gate_values is not None:
                    gate.update(multi_gate_values)
            if shapes is not None:
                return shapes, gate
    return None


def save_temporal_alpha_table(model, dataloader, device, label_scaler, save_path, fold):
    if not hasattr(model, "latest_temporal_alpha"):
        return None

    model.eval()
    rows = []
    with torch.no_grad():
        for X, Y, trial_ids in dataloader:
            X = X.to(device)
            output = model(X)
            alpha = getattr(model, "latest_temporal_alpha", None)
            if alpha is None:
                continue

            if isinstance(output, tuple):
                direct_score = output[0]
                ordinal_logits = output[1] if len(output) > 1 else None
                expected_score = direct_score
                if ordinal_logits is not None:
                    ordinal_score_real = torch.sum(torch.sigmoid(ordinal_logits), dim=1, keepdim=True)
                    ordinal_score = (ordinal_score_real - label_scaler['mean']) / (label_scaler['std'] + 1e-8)
                    score_fusion_alpha = None
                    if hasattr(model, 'get_score_fusion_alpha'):
                        score_fusion_alpha = model.get_score_fusion_alpha()
                    if score_fusion_alpha is None:
                        expected_score = 0.7 * direct_score + 0.3 * ordinal_score
                    else:
                        expected_score = score_fusion_alpha * direct_score + (1.0 - score_fusion_alpha) * ordinal_score
            else:
                expected_score = output

            preds = expected_score.detach().cpu().numpy().reshape(-1)
            labels = Y.detach().cpu().numpy().reshape(-1)
            preds = preds * label_scaler['std'] + label_scaler['mean']
            labels = labels * label_scaler['std'] + label_scaler['mean']
            alpha_np = alpha.detach().cpu().numpy()

            for idx, trial_id in enumerate(trial_ids):
                row = {
                    "sample_id": trial_id,
                    "label": float(labels[idx]),
                    "prediction": float(preds[idx]),
                }
                for t in range(alpha_np.shape[1]):
                    row[f"time_{t + 1}_alpha"] = float(alpha_np[idx, t])
                rows.append(row)

    if len(rows) == 0:
        return None

    shapes = getattr(model, "latest_gru_shapes", {}) or {}
    filename = f"graph_alpha_fold_{fold + 1}.csv" if shapes.get("temporal_block_graph_enabled") else f"temporal_alpha_fold_{fold + 1}.csv"
    csv_path = os.path.join(save_path, filename)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def save_parallel_tcn_tables(model, dataloader, device, label_scaler, save_path, fold, metrics=None, best_epoch=None):
    if not hasattr(model, "latest_parallel_branch_alphas"):
        return None, None

    model.eval()
    rows = []
    scale_weight = None
    dilations = None
    with torch.no_grad():
        for X, Y, trial_ids in dataloader:
            X = X.to(device)
            output = model(X)
            branch_alphas = getattr(model, "latest_parallel_branch_alphas", None)
            scale_weight = getattr(model, "latest_parallel_scale_weight", None)
            shapes = getattr(model, "latest_gru_shapes", {}) or {}
            dilations = shapes.get("parallel_tcn_dilations")
            if branch_alphas is None or scale_weight is None or dilations is None:
                continue

            if isinstance(output, tuple):
                direct_score = output[0]
                ordinal_logits = output[1] if len(output) > 1 else None
                expected_score = direct_score
                if ordinal_logits is not None:
                    ordinal_score_real = torch.sum(torch.sigmoid(ordinal_logits), dim=1, keepdim=True)
                    ordinal_score = (ordinal_score_real - label_scaler['mean']) / (label_scaler['std'] + 1e-8)
                    score_fusion_alpha = None
                    if hasattr(model, 'get_score_fusion_alpha'):
                        score_fusion_alpha = model.get_score_fusion_alpha()
                    if score_fusion_alpha is None:
                        expected_score = 0.7 * direct_score + 0.3 * ordinal_score
                    else:
                        expected_score = score_fusion_alpha * direct_score + (1.0 - score_fusion_alpha) * ordinal_score
            else:
                expected_score = output

            preds = expected_score.detach().cpu().numpy().reshape(-1)
            labels = Y.detach().cpu().numpy().reshape(-1)
            preds = preds * label_scaler['std'] + label_scaler['mean']
            labels = labels * label_scaler['std'] + label_scaler['mean']
            alpha_arrays = [alpha.detach().cpu().numpy() for alpha in branch_alphas]

            for idx, trial_id in enumerate(trial_ids):
                row = {
                    "sample_id": trial_id,
                    "label": float(labels[idx]),
                    "prediction": float(preds[idx]),
                }
                for dilation, alpha_np in zip(dilations, alpha_arrays):
                    for t in range(alpha_np.shape[1]):
                        row[f"d{dilation}_time_{t + 1}_alpha"] = float(alpha_np[idx, t])
                rows.append(row)

    alpha_csv = None
    if rows:
        alpha_csv = os.path.join(save_path, f"parallel_tcn_branch_alpha_fold_{fold + 1}.csv")
        pd.DataFrame(rows).to_csv(alpha_csv, index=False, encoding="utf-8-sig")

    scale_csv = None
    if scale_weight is not None and dilations is not None:
        scale_np = scale_weight.detach().cpu().numpy().reshape(-1)
        scale_row = {
            "fold": fold + 1,
            "best_epoch": best_epoch,
        }
        for dilation, value in zip(dilations, scale_np):
            scale_row[f"scale_weight_d{dilation}"] = float(value)
        if metrics is not None:
            scale_row.update({
                "test_mae": metrics.get("mae"),
                "test_rmse": metrics.get("rmse"),
                "test_r2": metrics.get("r2"),
                "test_pcc": metrics.get("pcc"),
                "test_ccc": metrics.get("ccc"),
            })
        scale_csv = os.path.join(save_path, f"parallel_tcn_scale_weight_fold_{fold + 1}.csv")
        pd.DataFrame([scale_row]).to_csv(scale_csv, index=False, encoding="utf-8-sig")

    return alpha_csv, scale_csv


def save_std_ratio_curve(trainer, save_path, fold):
    record = getattr(trainer, "train_record_dict", None)
    if not record:
        return None

    num_epochs = len(record.get("train_loss", []))
    if num_epochs == 0:
        return None

    rows = []
    for idx in range(num_epochs):
        train_pred_std = record.get("train_pred_std", [0.0] * num_epochs)[idx] if idx < len(record.get("train_pred_std", [])) else 0.0
        train_label_std = record.get("train_label_std", [0.0] * num_epochs)[idx] if idx < len(record.get("train_label_std", [])) else 0.0
        val_pred_std = record.get("val_pred_std", [0.0] * num_epochs)[idx] if idx < len(record.get("val_pred_std", [])) else 0.0
        val_label_std = record.get("val_label_std", [0.0] * num_epochs)[idx] if idx < len(record.get("val_label_std", [])) else 0.0
        rows.append({
            "epoch": idx + 1,
            "train_mae": record.get("train_mae", [np.nan] * num_epochs)[idx] if idx < len(record.get("train_mae", [])) else np.nan,
            "val_mae": record.get("val_mae", [np.nan] * num_epochs)[idx] if idx < len(record.get("val_mae", [])) else np.nan,
            "train_pred_std_ratio": float(train_pred_std / (train_label_std + 1e-8)),
            "val_pred_std_ratio": float(val_pred_std / (val_label_std + 1e-8)),
            "std_loss": record.get("train_std_loss", [0.0] * num_epochs)[idx] if idx < len(record.get("train_std_loss", [])) else 0.0,
            "lambda_std": record.get("train_lambda_std", [0.0] * num_epochs)[idx] if idx < len(record.get("train_lambda_std", [])) else 0.0,
            "lambda_std_times_std_loss": record.get("train_weighted_std_loss", [0.0] * num_epochs)[idx] if idx < len(record.get("train_weighted_std_loss", [])) else 0.0,
            "huber_loss": record.get("train_huber_loss", [0.0] * num_epochs)[idx] if idx < len(record.get("train_huber_loss", [])) else 0.0,
            "rank_loss": record.get("train_rank_loss", [0.0] * num_epochs)[idx] if idx < len(record.get("train_rank_loss", [])) else 0.0,
            "base_loss": record.get("train_base_loss", [0.0] * num_epochs)[idx] if idx < len(record.get("train_base_loss", [])) else 0.0,
            "total_loss": record.get("train_total_loss", [0.0] * num_epochs)[idx] if idx < len(record.get("train_total_loss", [])) else record.get("train_loss", [0.0] * num_epochs)[idx],
        })

    curve_path = os.path.join(save_path, f"std_ratio_curve_fold_{fold + 1}.csv")
    pd.DataFrame(rows).to_csv(curve_path, index=False, encoding="utf-8-sig")
    return curve_path


def save_multi_pli_gate_table(model, save_path, fold, metrics=None):
    if not hasattr(model, "get_multi_pli_gate_values"):
        return None
    gate_values = model.get_multi_pli_gate_values()
    if gate_values is None:
        return None

    row = {
        "fold": fold + 1,
        "multi_pli_branches": getattr(model, "multi_pli_branches", ""),
        "aux_residual_scale": getattr(model, "aux_residual_scale", ""),
        "shared_pli_encoder": getattr(model, "shared_pli_encoder", ""),
        **gate_values,
    }
    if metrics is not None:
        row.update({
            "test_mae": metrics.get("mae"),
            "test_rmse": metrics.get("rmse"),
            "test_r2": metrics.get("r2"),
            "test_pcc": metrics.get("pcc"),
            "test_ccc": metrics.get("ccc"),
        })

    csv_path = os.path.join(save_path, f"multi_pli_gate_curve_fold_{fold + 1}.csv")
    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


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


def compute_baseline_metrics_for_labels(train_labels, labels):
    baseline_value = float(np.mean(train_labels))
    baseline_preds = np.full_like(labels, baseline_value, dtype=np.float32)
    metrics = compute_regression_metrics(baseline_preds, labels)
    return baseline_value, metrics


def label_distribution_counts(labels, q30, q70):
    labels = np.asarray(labels, dtype=np.float32)
    low = labels <= q30
    high = labels >= q70
    mid = (~low) & (~high)
    total = max(int(labels.size), 1)
    return {
        "low_count": int(np.sum(low)),
        "mid_count": int(np.sum(mid)),
        "high_count": int(np.sum(high)),
        "low_ratio": float(np.sum(low) / total),
        "mid_ratio": float(np.sum(mid) / total),
        "high_ratio": float(np.sum(high) / total),
    }


def list_stats(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def append_csv_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def run_split_leakage_check(fold, fold_data):
    split_ids = {
        split: [item[2] for item in items]
        for split, items in (
            ("train", fold_data["train"]),
            ("val", fold_data["validate"]),
            ("test", fold_data["test"]),
        )
    }
    split_paths = {
        split: [item[0] for item in items]
        for split, items in (
            ("train", fold_data["train"]),
            ("val", fold_data["validate"]),
            ("test", fold_data["test"]),
        )
    }
    # Current trial_id is subject-level folder id, e.g. P103-T1, so subject_id == sample id.
    split_subjects = split_ids

    def overlap_size(mapping, a, b):
        return len(set(mapping[a]).intersection(set(mapping[b])))

    id_tv = overlap_size(split_ids, "train", "val")
    id_tt = overlap_size(split_ids, "train", "test")
    id_vt = overlap_size(split_ids, "val", "test")
    subj_tv = overlap_size(split_subjects, "train", "val")
    subj_tt = overlap_size(split_subjects, "train", "test")
    subj_vt = overlap_size(split_subjects, "val", "test")
    path_tv = overlap_size(split_paths, "train", "val")
    path_tt = overlap_size(split_paths, "train", "test")
    path_vt = overlap_size(split_paths, "val", "test")

    sample_ok = id_tv == 0 and id_tt == 0 and id_vt == 0
    subject_ok = subj_tv == 0 and subj_tt == 0 and subj_vt == 0
    path_ok = path_tv == 0 and path_tt == 0 and path_vt == 0

    print("Split leakage sanity check:")
    print(f"train_ids ∩ val_ids size = {id_tv}")
    print(f"train_ids ∩ test_ids size = {id_tt}")
    print(f"val_ids ∩ test_ids size = {id_vt}")
    print(f"train_subjects ∩ val_subjects size = {subj_tv}")
    print(f"train_subjects ∩ test_subjects size = {subj_tt}")
    print(f"val_subjects ∩ test_subjects size = {subj_vt}")
    print(f"train_paths ∩ val_paths size = {path_tv}")
    print(f"train_paths ∩ test_paths size = {path_tt}")
    print(f"val_paths ∩ test_paths size = {path_vt}")
    print(f"No sample ID overlap = {sample_ok}")
    print(f"No subject overlap = {subject_ok}")
    print(f"No feature path overlap = {path_ok}")

    if not (sample_ok and subject_ok and path_ok):
        raise ValueError(f"Fold {fold + 1} split leakage detected.")

    return {
        "sample_overlap_ok": sample_ok,
        "subject_overlap_ok": subject_ok,
        "path_overlap_ok": path_ok,
        "train_val_id_overlap": id_tv,
        "train_test_id_overlap": id_tt,
        "val_test_id_overlap": id_vt,
        "train_val_subject_overlap": subj_tv,
        "train_test_subject_overlap": subj_tt,
        "val_test_subject_overlap": subj_vt,
        "train_val_path_overlap": path_tv,
        "train_test_path_overlap": path_tt,
        "val_test_path_overlap": path_vt,
    }


def save_fold_distribution_check(fold, exp_name, fold_data, results_dir):
    train_labels = np.asarray([item[1] for item in fold_data["train"]], dtype=np.float32)
    q30, q70 = np.quantile(train_labels, [0.30, 0.70])
    rows = []
    for split_name, data_key in (("train", "train"), ("val", "validate"), ("test", "test")):
        labels = np.asarray([item[1] for item in fold_data[data_key]], dtype=np.float32)
        stats = list_stats(labels)
        counts = label_distribution_counts(labels, q30, q70)
        print(
            f"{split_name} label mean/std/min/max = "
            f"{stats['mean']:.4f}/{stats['std']:.4f}/{stats['min']:.4f}/{stats['max']:.4f}"
        )
        print(
            f"{split_name} low/mid/high count = "
            f"{counts['low_count']}/{counts['mid_count']}/{counts['high_count']} | "
            f"ratio = {counts['low_ratio']:.3f}/{counts['mid_ratio']:.3f}/{counts['high_ratio']:.3f}"
        )
        rows.append({
            "experiment_name": exp_name,
            "fold": fold + 1,
            "split": split_name,
            "q30_train": float(q30),
            "q70_train": float(q70),
            **{f"label_{key}": value for key, value in stats.items()},
            **counts,
        })

    out_path = os.path.join(results_dir, "split2034_fold_distribution_check.csv")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        df.to_csv(out_path, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return float(q30), float(q70)


def run_metric_sanity_check(fold, exp_name, trainer, dataloaders, fold_data, device, label_scaler, leakage_info, results_dir):
    print("\nBest checkpoint evaluation:")
    train_eval, _, train_labels, _ = evaluate_split(
        trainer.model, dataloaders["train"], "train", device, label_scaler=label_scaler
    )
    val_eval, _, val_labels, _ = evaluate_split(
        trainer.model, dataloaders["validate"], "val", device, label_scaler=label_scaler
    )
    test_eval, _, test_labels, _ = evaluate_split(
        trainer.model, dataloaders["test"], "test", device, label_scaler=label_scaler
    )

    train_raw_labels = np.asarray([item[1] for item in fold_data["train"]], dtype=np.float32)
    _, train_base = compute_baseline_metrics_for_labels(train_raw_labels, train_labels.astype(np.float32))
    _, val_base = compute_baseline_metrics_for_labels(train_raw_labels, val_labels.astype(np.float32))
    _, test_base = compute_baseline_metrics_for_labels(train_raw_labels, test_labels.astype(np.float32))

    evals = {
        "TrainEval": (train_eval, train_base),
        "ValEval": (val_eval, val_base),
        "TestEval": (test_eval, test_base),
    }
    for name, (metrics, baseline) in evals.items():
        print(
            f"{name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
            f"R2={metrics['r2']:.4f}, PCC={metrics['pcc']:.4f}, CCC={metrics['ccc']:.4f}, "
            f"pred_std={metrics['pred_std']:.4f}, label_std={metrics['label_std']:.4f}"
        )
        print(
            f"{name} train-mean baseline: model_MAE={metrics['mae']:.4f}, "
            f"mean_baseline_MAE={baseline['mae']:.4f}, "
            f"model_minus_baseline_MAE={metrics['mae'] - baseline['mae']:.4f}"
        )
        if metrics["pred_std"] < 0.2 * metrics["label_std"]:
            print(f"Warning: prediction range collapse detected on {name}.")

    row = {
        "experiment_name": exp_name,
        "fold": fold + 1,
        "train_size": len(fold_data["train"]),
        "val_size": len(fold_data["validate"]),
        "test_size": len(fold_data["test"]),
        **leakage_info,
        "train_label_mean": train_eval["label_mean"],
        "train_label_std": train_eval["label_std"],
        "train_label_min": train_eval["label_min"],
        "train_label_max": train_eval["label_max"],
        "val_label_mean": val_eval["label_mean"],
        "val_label_std": val_eval["label_std"],
        "val_label_min": val_eval["label_min"],
        "val_label_max": val_eval["label_max"],
        "test_label_mean": test_eval["label_mean"],
        "test_label_std": test_eval["label_std"],
        "test_label_min": test_eval["label_min"],
        "test_label_max": test_eval["label_max"],
        "train_model_MAE": train_eval["mae"],
        "val_model_MAE": val_eval["mae"],
        "test_model_MAE": test_eval["mae"],
        "train_mean_baseline_MAE": train_base["mae"],
        "val_mean_baseline_MAE": val_base["mae"],
        "test_mean_baseline_MAE": test_base["mae"],
        "train_pred_std": train_eval["pred_std"],
        "val_pred_std": val_eval["pred_std"],
        "test_pred_std": test_eval["pred_std"],
        "train_label_std": train_eval["label_std"],
        "val_label_std": val_eval["label_std"],
        "test_label_std": test_eval["label_std"],
        "train_pcc": train_eval["pcc"],
        "val_pcc": val_eval["pcc"],
        "test_pcc": test_eval["pcc"],
        "train_ccc": train_eval["ccc"],
        "val_ccc": val_eval["ccc"],
        "test_ccc": test_eval["ccc"],
    }
    append_csv_row(os.path.join(results_dir, "split2034_sanity_check.csv"), row)
    return row


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
    results_dir = os.path.join(save_root, "results")
    leakage_info = {
        "sample_overlap_ok": True,
        "subject_overlap_ok": True,
        "path_overlap_ok": True,
    }
    if getattr(args, "run_sanity_check", False):
        leakage_info = run_split_leakage_check(fold, fold_data)
        save_fold_distribution_check(fold, exp_name, fold_data, results_dir)

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
    print(f"DataLoader generator seed = {seed}")
    print("worker_init_fn enabled = True")
    print(f"cudnn deterministic = {torch.backends.cudnn.deterministic}")
    print(f"cudnn benchmark = {torch.backends.cudnn.benchmark}")

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
            expected_branches = {
                "delta_pli": 1,
                "delta_task": 2,
                "delta_rest": 2,
                "delta_task_rest": 3,
            }.get(input_mode, 1)
            print(f"expected PLI branches = {expected_branches}")
            if x_pli.dim() != 4 or x_pli.shape[1] != expected_branches or x_pli.shape[2] != 140 or x_pli.shape[3] != 40:
                print(
                    f"WARNING: expected x_pli shape [B, {expected_branches}, 140, 40], "
                    f"got {list(x_pli.shape)}"
                )
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
        attn_scale=getattr(args, 'attn_scale', 0.1),
        gamma_band=getattr(args, 'gamma_band', 0.2),
        gamma_edge=getattr(args, 'gamma_edge', 0.2),
        band_mlp_hidden_dim=getattr(args, 'band_mlp_hidden_dim', 16),
        edge_mlp_hidden_dim=getattr(args, 'edge_mlp_hidden_dim', 64),
        attention_dropout=getattr(args, 'attention_dropout', 0.0),
        use_tct_lite=getattr(args, 'use_tct_lite', False),
        tct_gate=getattr(args, 'tct_gate', 0.05),
        num_tct_layers=getattr(args, 'num_tct_layers', 1),
        tct_num_heads=getattr(args, 'tct_num_heads', 4),
        tct_ffn_dim=getattr(args, 'tct_ffn_dim', 128),
        tct_dropout=getattr(args, 'tct_dropout', 0.25),
        temporal_pos_embed=getattr(args, 'temporal_pos_embed', True),
        temporal_attention_pooling=getattr(args, 'temporal_attention_pooling', True),
        use_tcn_lite=getattr(args, 'use_tcn_lite', False),
        tcn_gate=getattr(args, 'tcn_gate', 0.05),
        tcn_channels=getattr(args, 'tcn_channels', 64),
        tcn_kernel_size=getattr(args, 'tcn_kernel_size', 3),
        tcn_dilations=getattr(args, 'tcn_dilations', [1, 2, 4]),
        tcn_dropout=getattr(args, 'tcn_dropout', 0.20),
        tcn_pooling=getattr(args, 'tcn_pooling', 'attention'),
        tcn_residual=getattr(args, 'tcn_residual', True),
        use_parallel_multiscale_tcn=getattr(args, 'use_parallel_multiscale_tcn', False),
        parallel_tcn_gate=getattr(args, 'parallel_tcn_gate', getattr(args, 'tcn_gate', 0.10)),
        parallel_tcn_dilations=getattr(args, 'parallel_tcn_dilations', [1, 2, 4]),
        parallel_tcn_kernel_size=getattr(args, 'parallel_tcn_kernel_size', 3),
        parallel_tcn_dropout=getattr(args, 'parallel_tcn_dropout', getattr(args, 'tcn_dropout', 0.20)),
        parallel_tcn_pooling=getattr(args, 'parallel_tcn_pooling', 'attention'),
        parallel_tcn_residual=getattr(args, 'parallel_tcn_residual', True),
        use_temporal_block_graph=getattr(args, 'use_temporal_block_graph', False),
        graph_gate=getattr(args, 'graph_gate', 0.10),
        edge_distances=getattr(args, 'edge_distances', [1]),
        graph_dropout=getattr(args, 'graph_dropout', 0.20),
        graph_pooling=getattr(args, 'graph_pooling', 'attention'),
        graph_residual=getattr(args, 'graph_residual', True),
        use_self_loop=getattr(args, 'use_self_loop', True),
        learnable_score_fusion=getattr(args, 'learnable_score_fusion', False),
        init_score_fusion_alpha=getattr(args, 'init_score_fusion_alpha', 0.7),
        use_multi_pli=getattr(args, 'use_multi_pli', False),
        multi_pli_branches=getattr(args, 'multi_pli_branches', 'delta_task'),
        multi_pli_fusion_type=getattr(args, 'multi_pli_fusion_type', 'gated_residual'),
        fusion_init_delta_weight=getattr(args, 'fusion_init_delta_weight', 0.80),
        aux_residual_scale=getattr(args, 'aux_residual_scale', 0.10),
        shared_pli_encoder=getattr(args, 'shared_pli_encoder', True),
        log_branch_gates=getattr(args, 'log_branch_gates', True),
        loss_weighting=getattr(args, 'loss_weighting', 'fixed'),
        input_channels=1,
    ).to(device)

    if getattr(args, 'learnable_score_fusion', False) and hasattr(model, 'score_fusion_logit'):
        included_in_parameters = any(param is model.score_fusion_logit for param in model.parameters())
        print(f"fusion_logit requires_grad = {model.score_fusion_logit.requires_grad}")
        print(f"fusion_logit included in optimizer = {included_in_parameters}")

    print("MASA-TCN in_channels = 1")
    print("PLI-oriented encoder enabled = True")
    if getattr(args, 'use_multi_pli', False):
        print("Multi-PLI shared-encoder fusion enabled = True")
        print(f"multi_pli_branches = {getattr(args, 'multi_pli_branches', 'delta_task')}")
        print(f"multi_pli_fusion_type = {getattr(args, 'multi_pli_fusion_type', 'gated_residual')}")
        print(f"aux_residual_scale = {getattr(args, 'aux_residual_scale', 0.10)}")
        print(f"fusion_init_delta_weight = {getattr(args, 'fusion_init_delta_weight', 0.80)}")
        print(f"lambda_fusion_prior = {getattr(args, 'lambda_fusion_prior', 0.0)}")
        print(f"shared_pli_encoder = {getattr(args, 'shared_pli_encoder', True)}")
        if getattr(args, 'multi_pli_fusion_type', 'gated_residual') == 'learnable_softmax':
            print("Learnable softmax fusion enabled = True")
            print(f"initial w_delta = {getattr(args, 'fusion_init_delta_weight', 0.80):.4f}")
            print(f"initial w_task = {1.0 - float(getattr(args, 'fusion_init_delta_weight', 0.80)):.4f}")
        else:
            print("Gated residual fusion enabled = True")
            print("initial gate_task = 0.5000")
            print("initial gate_rest = 0.5000")
            print(f"initial effective aux scale = {0.5 * float(getattr(args, 'aux_residual_scale', 0.10)):.4f}")
    if getattr(args, 'use_band_edge_attention', False):
        print("Backbone = PLIEncoderBandEdgeAttention")
        print(f"attn_scale = {getattr(args, 'attn_scale', 0.1)}")
        print(f"Attention dropout = {getattr(args, 'attention_dropout', 0.0)}")
        if getattr(args, 'lambda_attn', 0.0) > 0:
            print("Attention regularization enabled = True")
            print(f"lambda_attn = {getattr(args, 'lambda_attn', 0.0)}")
    else:
        print("Backbone = PLIEncoder")
    print("MASA-TCN enabled = False")
    print(f"GRU Token Mixer enabled = {getattr(args, 'use_gru', True)}")
    if getattr(args, 'use_tct_lite', False):
        print("Temporal Context Transformer Lite enabled = True")
        print(f"num_tct_layers = {getattr(args, 'num_tct_layers', 1)}")
        print(f"tct_num_heads = {getattr(args, 'tct_num_heads', 4)}")
        print(f"tct_ffn_dim = {getattr(args, 'tct_ffn_dim', 128)}")
        print(f"tct_dropout = {getattr(args, 'tct_dropout', 0.25)}")
        print(f"tct_gate = {getattr(args, 'tct_gate', 0.05)}")
        print(f"temporal_pos_embed = {getattr(args, 'temporal_pos_embed', True)}")
        print(f"temporal_attention_pooling = {getattr(args, 'temporal_attention_pooling', True)}")
    if getattr(args, 'use_tcn_lite', False):
        print("TCN-Lite enabled = True")
        print(f"tcn_channels = {getattr(args, 'tcn_channels', 64)}")
        print(f"tcn_kernel_size = {getattr(args, 'tcn_kernel_size', 3)}")
        print(f"tcn_dilations = {getattr(args, 'tcn_dilations', [1, 2, 4])}")
        print(f"tcn_dropout = {getattr(args, 'tcn_dropout', 0.20)}")
        print(f"tcn_gate = {getattr(args, 'tcn_gate', 0.05)}")
        print(f"tcn_pooling = {getattr(args, 'tcn_pooling', 'attention')}")
        print(f"tcn_residual = {getattr(args, 'tcn_residual', True)}")
    if getattr(args, 'use_parallel_multiscale_tcn', False):
        print("Parallel Multi-scale TCN-Lite enabled = True")
        print(f"parallel_tcn_dilations = {getattr(args, 'parallel_tcn_dilations', [1, 2, 4])}")
        print(f"parallel_tcn_kernel_size = {getattr(args, 'parallel_tcn_kernel_size', 3)}")
        print("parallel_tcn_channels = 64")
        print(f"parallel_tcn_dropout = {getattr(args, 'parallel_tcn_dropout', 0.20)}")
        print(f"tcn_gate = {getattr(args, 'parallel_tcn_gate', getattr(args, 'tcn_gate', 0.10))}")
        print("scale_fusion = softmax learnable weights")
        print(f"branch_pooling = {getattr(args, 'parallel_tcn_pooling', 'attention')}")
    if getattr(args, 'use_temporal_block_graph', False):
        print("Temporal Block Graph enabled = True")
        print(f"edge_distances = {getattr(args, 'edge_distances', [1])}")
        print(f"graph_dropout = {getattr(args, 'graph_dropout', 0.20)}")
        print(f"graph_gate = {getattr(args, 'graph_gate', 0.10)}")
        print("graph_hidden_dim = 64")
        print("graph_num_layers = 1")
        print(f"graph_pooling = {getattr(args, 'graph_pooling', 'attention')}")
        print(f"graph_residual = {getattr(args, 'graph_residual', True)}")
        print(f"use_self_loop = {getattr(args, 'use_self_loop', True)}")
        print("use_fixed_temporal_graph = True")

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
        'attn_scale': getattr(args, 'attn_scale', 0.1),
        'gamma_band': getattr(args, 'gamma_band', 0.2),
        'gamma_edge': getattr(args, 'gamma_edge', 0.2),
        'band_mlp_hidden_dim': getattr(args, 'band_mlp_hidden_dim', 16),
        'edge_mlp_hidden_dim': getattr(args, 'edge_mlp_hidden_dim', 64),
        'attention_dropout': getattr(args, 'attention_dropout', 0.0),
        'use_tct_lite': getattr(args, 'use_tct_lite', False),
        'tct_gate': getattr(args, 'tct_gate', 0.05),
        'num_tct_layers': getattr(args, 'num_tct_layers', 1),
        'tct_num_heads': getattr(args, 'tct_num_heads', 4),
        'tct_ffn_dim': getattr(args, 'tct_ffn_dim', 128),
        'tct_dropout': getattr(args, 'tct_dropout', 0.25),
        'temporal_pos_embed': getattr(args, 'temporal_pos_embed', True),
        'temporal_attention_pooling': getattr(args, 'temporal_attention_pooling', True),
        'use_tcn_lite': getattr(args, 'use_tcn_lite', False),
        'tcn_gate': getattr(args, 'tcn_gate', 0.05),
        'tcn_channels': getattr(args, 'tcn_channels', 64),
        'tcn_kernel_size': getattr(args, 'tcn_kernel_size', 3),
        'tcn_dilations': getattr(args, 'tcn_dilations', [1, 2, 4]),
        'tcn_dropout': getattr(args, 'tcn_dropout', 0.20),
        'tcn_pooling': getattr(args, 'tcn_pooling', 'attention'),
        'tcn_residual': getattr(args, 'tcn_residual', True),
        'use_parallel_multiscale_tcn': getattr(args, 'use_parallel_multiscale_tcn', False),
        'parallel_tcn_gate': getattr(args, 'parallel_tcn_gate', getattr(args, 'tcn_gate', 0.10)),
        'parallel_tcn_dilations': getattr(args, 'parallel_tcn_dilations', [1, 2, 4]),
        'parallel_tcn_kernel_size': getattr(args, 'parallel_tcn_kernel_size', 3),
        'parallel_tcn_dropout': getattr(args, 'parallel_tcn_dropout', getattr(args, 'tcn_dropout', 0.20)),
        'parallel_tcn_pooling': getattr(args, 'parallel_tcn_pooling', 'attention'),
        'parallel_tcn_residual': getattr(args, 'parallel_tcn_residual', True),
        'use_temporal_block_graph': getattr(args, 'use_temporal_block_graph', False),
        'graph_gate': getattr(args, 'graph_gate', 0.10),
        'edge_distances': getattr(args, 'edge_distances', [1]),
        'graph_dropout': getattr(args, 'graph_dropout', 0.20),
        'graph_pooling': getattr(args, 'graph_pooling', 'attention'),
        'graph_residual': getattr(args, 'graph_residual', True),
        'use_self_loop': getattr(args, 'use_self_loop', True),
        'lambda_attn': getattr(args, 'lambda_attn', 0.0),
        'use_std_ratio_loss': getattr(args, 'use_std_ratio_loss', False),
        'target_std_ratio': getattr(args, 'target_std_ratio', 0.25),
        'lambda_std': getattr(args, 'lambda_std', 0.0),
        'learnable_score_fusion': getattr(args, 'learnable_score_fusion', False),
        'init_score_fusion_alpha': getattr(args, 'init_score_fusion_alpha', 0.7),
        'use_multi_pli': getattr(args, 'use_multi_pli', False),
        'multi_pli_branches': getattr(args, 'multi_pli_branches', 'delta_task'),
        'multi_pli_fusion_type': getattr(args, 'multi_pli_fusion_type', 'gated_residual'),
        'fusion_init_delta_weight': getattr(args, 'fusion_init_delta_weight', 0.80),
        'lambda_fusion_prior': getattr(args, 'lambda_fusion_prior', 0.0),
        'aux_residual_scale': getattr(args, 'aux_residual_scale', 0.10),
        'shared_pli_encoder': getattr(args, 'shared_pli_encoder', True),
        'log_branch_gates': getattr(args, 'log_branch_gates', True),
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
    if getattr(args, 'use_std_ratio_loss', False):
        std_curve_path = save_std_ratio_curve(trainer, args_dict['save_path'], fold)
        if std_curve_path:
            print(f"std_ratio curve saved: {std_curve_path}")

    print(f"? {fold + 1} ?????????????: {args_dict['save_path']}")

    test_loss, test_record_dict = trainer.test(
        checkpoint_controller=exp_logger,
        dataloader_dict=dataloaders,
        epoch=None,
    )

    if getattr(args, "run_sanity_check", False):
        run_metric_sanity_check(
            fold,
            exp_name,
            trainer,
            dataloaders,
            fold_data,
            device,
            label_scaler,
            leakage_info,
            results_dir,
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

    val_preds, val_labels, val_trial_ids = collect_predictions(
        trainer.model, dataloaders['validate'], device, label_scaler=label_scaler
    )
    val_prediction_csv = save_prediction_table(
        args_dict['save_path'],
        val_trial_ids,
        val_labels,
        val_preds,
        baseline_value,
        'val_predictions.csv',
    )

    preds, labels, trial_ids = collect_predictions(
        trainer.model, dataloaders['test'], device, label_scaler=label_scaler
    )
    print_prediction_distribution("Test", preds, labels)
    test_pred_std = float(np.std(preds))
    test_label_std = float(np.std(labels))
    test_pred_std_ratio = float(test_pred_std / (test_label_std + 1e-8))
    print(f"Test pred_std / label_std: {test_pred_std_ratio:.4f}")
    if test_pred_std_ratio < 0.10:
        print("Warning: severe prediction collapse.")
    elif test_pred_std_ratio < 0.25:
        print("Warning: mild prediction collapse.")
    elif test_pred_std_ratio > 0.50:
        print("Warning: prediction variance may be too large.")

    gru_diag = collect_gru_token_mixer_diagnostics(trainer.model, dataloaders['test'], device)
    temporal_alpha_std_for_summary = None
    if gru_diag is not None:
        gru_shapes, gates = gru_diag
        print("GRU Token Mixer RegOnly enabled")
        print(f"x_pli shape: {gru_shapes.get('x_pli')}")
        if gru_shapes.get('pli_oriented_encoder'):
            print("PLI-oriented encoder enabled = True")
            if gru_shapes.get('band_edge_attention'):
                print("Backbone = PLIEncoderBandEdgeAttention")
            else:
                print("Backbone = PLIEncoder")
        print(f"h_seq shape: {gru_shapes.get('h_seq')}")
        print(f"h_base shape: {gru_shapes.get('h_base')}")
        if (
            gru_shapes.get('gru_enabled') is False
            and not gru_shapes.get('tct_lite_enabled')
            and not gru_shapes.get('tcn_lite_enabled')
            and not gru_shapes.get('parallel_multiscale_tcn_enabled')
            and not gru_shapes.get('temporal_block_graph_enabled')
        ):
            print("h_fused = h_base")
        elif gru_shapes.get('gru_enabled') is not False:
            print(f"h_gru shape: {gru_shapes.get('h_gru')}")
            print(f"h_gru_proj shape: {gru_shapes.get('h_gru_proj')}")
        if gru_shapes.get('tct_lite_enabled'):
            print("Temporal Context Transformer Lite enabled = True")
            print(f"h_tct shape: {gru_shapes.get('h_tct')}")
            print(f"temporal_alpha shape: {gru_shapes.get('temporal_alpha')}")
            for layer_info in gru_shapes.get('tct_layers', []):
                print(
                    f"tct_layer_outputs shape | layer {layer_info.get('layer')}: "
                    f"{layer_info.get('output_shape')}"
                )
            print(
                "temporal_alpha mean/std/min/max: "
                f"{gru_shapes.get('temporal_alpha_mean'):.6f} / "
                f"{gru_shapes.get('temporal_alpha_std'):.6f} / "
                f"{gru_shapes.get('temporal_alpha_min'):.6f} / "
                f"{gru_shapes.get('temporal_alpha_max'):.6f}"
            )
            temporal_alpha_std_for_summary = gru_shapes.get('temporal_alpha_std')
            if temporal_alpha_std_for_summary is not None and temporal_alpha_std_for_summary < 0.003:
                print("Warning: temporal attention is nearly uniform.")
            print(f"tct_gate = {gru_shapes.get('tct_gate'):.4f}")
        if gru_shapes.get('tcn_lite_enabled'):
            print("TCN-Lite enabled = True")
            print(f"h_tcn_seq shape: {gru_shapes.get('h_tcn_seq')}")
            print(f"h_tcn shape: {gru_shapes.get('h_tcn')}")
            print(f"temporal_alpha shape: {gru_shapes.get('temporal_alpha')}")
            for layer_info in gru_shapes.get('tcn_layers', []):
                print(
                    f"tcn_layer_outputs shape | layer {layer_info.get('layer')} "
                    f"(dilation={layer_info.get('dilation')}): {layer_info.get('output_shape')}"
                )
            print(
                "temporal_alpha mean/std/min/max: "
                f"{gru_shapes.get('temporal_alpha_mean'):.6f} / "
                f"{gru_shapes.get('temporal_alpha_std'):.6f} / "
                f"{gru_shapes.get('temporal_alpha_min'):.6f} / "
                f"{gru_shapes.get('temporal_alpha_max'):.6f}"
            )
            temporal_alpha_std_for_summary = gru_shapes.get('temporal_alpha_std')
            if temporal_alpha_std_for_summary is not None and temporal_alpha_std_for_summary < 0.003:
                print("Warning: temporal attention is nearly uniform.")
            print(f"tcn_gate = {gru_shapes.get('tcn_gate'):.4f}")
            print(f"tcn_channels = {gru_shapes.get('tcn_channels')}")
            print(f"tcn_kernel_size = {gru_shapes.get('tcn_kernel_size')}")
            print(f"tcn_dilations = {gru_shapes.get('tcn_dilations')}")
            print(f"tcn_dropout = {gru_shapes.get('tcn_dropout')}")
            print(f"tcn_pooling = {gru_shapes.get('tcn_pooling')}")
            print(f"tcn_residual = {gru_shapes.get('tcn_residual')}")
        if gru_shapes.get('parallel_multiscale_tcn_enabled'):
            print("Parallel Multi-scale TCN-Lite enabled = True")
            print(f"parallel_tcn_dilations = {gru_shapes.get('parallel_tcn_dilations')}")
            print(f"parallel_tcn_kernel_size = {gru_shapes.get('parallel_tcn_kernel_size')}")
            print(f"parallel_tcn_channels = {gru_shapes.get('parallel_tcn_channels')}")
            print(f"parallel_tcn_dropout = {gru_shapes.get('parallel_tcn_dropout')}")
            print(f"scale_fusion = {gru_shapes.get('scale_fusion')} learnable weights")
            print(f"branch_pooling = {gru_shapes.get('branch_pooling')}")
            for dilation in gru_shapes.get('parallel_tcn_dilations', []):
                print(f"branch_d{dilation}_seq shape: {gru_shapes.get(f'branch_d{dilation}_seq')}")
                print(f"h_d{dilation} shape: {gru_shapes.get(f'h_d{dilation}')}")
                print(f"branch_alpha_d{dilation} shape: {gru_shapes.get(f'branch_alpha_d{dilation}')}")
                print(
                    f"branch_alpha_d{dilation} mean/std/min/max: "
                    f"{gru_shapes.get(f'branch_alpha_d{dilation}_mean'):.6f} / "
                    f"{gru_shapes.get(f'branch_alpha_d{dilation}_std'):.6f} / "
                    f"{gru_shapes.get(f'branch_alpha_d{dilation}_min'):.6f} / "
                    f"{gru_shapes.get(f'branch_alpha_d{dilation}_max'):.6f}"
                )
                print(f"scale_weight_d{dilation} = {gru_shapes.get(f'scale_weight_d{dilation}'):.6f}")
            print(f"scale_weight shape: {gru_shapes.get('scale_weight')}")
            print(f"h_tcn shape: {gru_shapes.get('h_tcn')}")
            print(f"tcn_gate = {gru_shapes.get('parallel_tcn_gate'):.4f}")
        if gru_shapes.get('temporal_block_graph_enabled'):
            print("Temporal Block Graph enabled = True")
            print(f"num_temporal_nodes = {gru_shapes.get('num_temporal_nodes')}")
            print(f"edge_distances = {gru_shapes.get('edge_distances')}")
            print(f"num_edges_without_self_loop = {gru_shapes.get('num_edges_without_self_loop')}")
            print(f"num_edges_with_self_loop = {gru_shapes.get('num_edges_with_self_loop')}")
            print(f"A_norm shape = {gru_shapes.get('A_norm')}")
            print(f"A_norm density = {gru_shapes.get('A_norm_density')}")
            print(f"graph_seq shape: {gru_shapes.get('graph_seq')}")
            print(f"h_graph shape: {gru_shapes.get('h_graph')}")
            print(f"graph_alpha shape: {gru_shapes.get('graph_alpha')}")
            print(
                "graph_alpha mean/std/min/max: "
                f"{gru_shapes.get('temporal_alpha_mean'):.6f} / "
                f"{gru_shapes.get('temporal_alpha_std'):.6f} / "
                f"{gru_shapes.get('temporal_alpha_min'):.6f} / "
                f"{gru_shapes.get('temporal_alpha_max'):.6f}"
            )
            temporal_alpha_std_for_summary = gru_shapes.get('temporal_alpha_std')
            if temporal_alpha_std_for_summary is not None and temporal_alpha_std_for_summary < 0.003:
                print("Warning: graph temporal attention is nearly uniform.")
            print(f"graph_gate = {gru_shapes.get('graph_gate'):.4f}")
            print(f"graph_dropout = {gru_shapes.get('graph_dropout')}")
            print(f"graph_pooling = {gru_shapes.get('graph_pooling')}")
            print(f"graph_residual = {gru_shapes.get('graph_residual')}")
            print(f"use_self_loop = {gru_shapes.get('use_self_loop')}")
            print(f"use_fixed_temporal_graph = {gru_shapes.get('use_fixed_temporal_graph')}")
        print(f"h_fused shape: {gru_shapes.get('h_fused')}")
        print(f"direct_score shape: {gru_shapes.get('direct_score')}")
        print(f"ordinal_logits shape: {gru_shapes.get('ordinal_logits')}")
        print(f"expected_score shape: {gru_shapes.get('expected_score')}")
        if gru_shapes.get('learnable_score_fusion'):
            print("Score fusion: expected_score = alpha * direct_score + (1-alpha) * ordinal_score")
            print(f"fusion_logit = {gru_shapes.get('fusion_logit'):.4f}")
            print(f"fusion_alpha_direct = {gru_shapes.get('score_fusion_alpha'):.4f}")
            print(f"fusion_alpha_ordinal = {gru_shapes.get('fusion_alpha_ordinal'):.4f}")
        if gru_shapes.get('use_multi_pli'):
            print("Multi-PLI shared-encoder fusion enabled = True")
            print(f"multi_pli_input shape: {gru_shapes.get('multi_pli_input')}")
            print(f"multi_pli_branches = {gru_shapes.get('multi_pli_branches')}")
            fusion_type = gru_shapes.get('multi_pli_fusion_type')
            print(f"multi_pli_fusion_type = {fusion_type}")
            print(f"aux_residual_scale = {gru_shapes.get('aux_residual_scale')}")
            print(f"shared_pli_encoder = {gru_shapes.get('shared_pli_encoder')}")
            if fusion_type == 'gated_residual':
                print(f"gate_task = {gru_shapes.get('gate_task', float('nan')):.6f}")
                print(f"gate_rest = {gru_shapes.get('gate_rest', float('nan')):.6f}")
                print(f"effective_task_scale = {gru_shapes.get('effective_task_scale', float('nan')):.6f}")
                print(f"effective_rest_scale = {gru_shapes.get('effective_rest_scale', float('nan')):.6f}")
            elif fusion_type == 'learnable_softmax':
                print(f"w_delta = {gru_shapes.get('fusion_w_delta', float('nan')):.6f}")
                print(f"w_task = {gru_shapes.get('fusion_w_task', float('nan')):.6f}")
                print(f"fusion_prior_loss = {gru_shapes.get('fusion_prior_loss', float('nan')):.6f}")
                print(f"lambda_fusion_prior = {getattr(args, 'lambda_fusion_prior', 0.0):.6f}")
        gru_gate = gates.get('gru_gate') if isinstance(gates, dict) else gates
        if gru_gate is not None and gru_shapes.get('gru_enabled') is not False:
            print(f"gru_gate = {gru_gate:.4f}")

    temporal_alpha_csv = save_temporal_alpha_table(
        trainer.model,
        dataloaders['test'],
        device,
        label_scaler,
        args_dict['save_path'],
        fold,
    )
    if temporal_alpha_csv is not None:
        print(f"Temporal alpha weights saved: {temporal_alpha_csv}")

    metrics = compute_regression_metrics(preds, labels)
    multi_pli_gate_csv = save_multi_pli_gate_table(
        trainer.model,
        args_dict['save_path'],
        fold,
        metrics=metrics,
    )
    if multi_pli_gate_csv is not None:
        print(f"Multi-PLI gate curve saved: {multi_pli_gate_csv}")
    parallel_alpha_csv, parallel_scale_csv = save_parallel_tcn_tables(
        trainer.model,
        dataloaders['test'],
        device,
        label_scaler,
        args_dict['save_path'],
        fold,
        metrics=metrics,
        best_epoch=None if getattr(trainer, 'best_epoch_info', None) is None else int(trainer.best_epoch_info.get('epoch', -1)) + 1,
    )
    if parallel_alpha_csv is not None:
        print(f"Parallel TCN branch alpha weights saved: {parallel_alpha_csv}")
    if parallel_scale_csv is not None:
        print(f"Parallel TCN scale weights saved: {parallel_scale_csv}")
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
    print(f"? {fold + 1} ????????????: {val_prediction_csv}")
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
    print(f"Model MAE - Mean baseline MAE: {metrics['mae'] - baseline_metrics['mae']:.4f}")
    if metrics['mae'] - baseline_metrics['mae'] < 0:
        print("This fold outperforms mean baseline.")
    else:
        print("Warning: this fold does not outperform mean baseline.")
    print(f"? {fold + 1} ????????????: {prediction_csv}")

    metrics_with_baseline = metrics.copy()
    for key in (
        'log_var_huber', 'log_var_rank', 'weight_huber', 'weight_rank',
        'std_loss', 'target_std_ratio', 'lambda_std', 'weighted_std_loss',
    ):
        if key in test_record_dict['overall']:
            metrics_with_baseline[key] = test_record_dict['overall'][key]
    fusion_type_for_metrics = getattr(args, 'multi_pli_fusion_type', 'gated_residual')
    if getattr(args, 'use_multi_pli', False) and fusion_type_for_metrics == 'learnable_softmax':
        for key in ('fusion_w_delta', 'fusion_w_task', 'fusion_prior_loss', 'lambda_fusion_prior', 'weighted_fusion_prior_loss'):
            if key in test_record_dict['overall']:
                metrics_with_baseline[key] = test_record_dict['overall'][key]
    metrics_with_baseline['input_mode'] = input_mode
    metrics_with_baseline['feature_type'] = 'Delta-PLI'
    metrics_with_baseline['input_channels'] = 1
    metrics_with_baseline['baseline_correction'] = 'task_minus_baseline'
    metrics_with_baseline['pred_std'] = float(np.std(preds))
    metrics_with_baseline['label_std'] = float(np.std(labels))
    metrics_with_baseline['pred_std_ratio'] = float(np.std(preds) / (np.std(labels) + 1e-8))
    metrics_with_baseline['model_minus_baseline_mae'] = float(metrics['mae'] - baseline_metrics['mae'])
    if temporal_alpha_std_for_summary is not None:
        metrics_with_baseline['temporal_alpha_std'] = float(temporal_alpha_std_for_summary)
        if getattr(args, 'use_temporal_block_graph', False):
            metrics_with_baseline['graph_alpha_std'] = float(temporal_alpha_std_for_summary)
    if gru_diag is not None and gru_shapes.get('use_multi_pli'):
        fusion_type = gru_shapes.get('multi_pli_fusion_type')
        keys = ['aux_residual_scale']
        if fusion_type == 'gated_residual':
            keys.extend(['gate_task', 'gate_rest', 'effective_task_scale', 'effective_rest_scale'])
        elif fusion_type == 'learnable_softmax':
            keys.extend(['fusion_w_delta', 'fusion_w_task', 'fusion_prior_loss'])
        for key in keys:
            metrics_with_baseline[key] = gru_shapes.get(key)
        metrics_with_baseline['multi_pli_branches'] = gru_shapes.get('multi_pli_branches')
        metrics_with_baseline['multi_pli_fusion_type'] = gru_shapes.get('multi_pli_fusion_type')
        metrics_with_baseline['shared_pli_encoder'] = gru_shapes.get('shared_pli_encoder')
    if gru_diag is not None and gru_shapes.get('parallel_multiscale_tcn_enabled'):
        for dilation in gru_shapes.get('parallel_tcn_dilations', []):
            for suffix in ('mean', 'std', 'min', 'max'):
                key = f'branch_alpha_d{dilation}_{suffix}'
                metrics_with_baseline[key] = gru_shapes.get(key)
            metrics_with_baseline[f'scale_weight_d{dilation}'] = gru_shapes.get(f'scale_weight_d{dilation}')
        metrics_with_baseline['parallel_tcn_dilations'] = gru_shapes.get('parallel_tcn_dilations')
        metrics_with_baseline['parallel_tcn_gate'] = gru_shapes.get('parallel_tcn_gate')
        metrics_with_baseline['parallel_tcn_dropout'] = gru_shapes.get('parallel_tcn_dropout')
    if gru_diag is not None and gru_shapes.get('temporal_block_graph_enabled'):
        for key in ('edge_distances', 'num_edges_without_self_loop', 'num_edges_with_self_loop', 'A_norm_density'):
            metrics_with_baseline[key] = gru_shapes.get(key)
    metrics_with_baseline['based_on'] = getattr(args, 'based_on', '')
    metrics_with_baseline['fold_split_seed'] = getattr(args, 'fold_split_seed', '')
    metrics_with_baseline['train_seed'] = getattr(args, 'seed', '')
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
