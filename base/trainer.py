import time
import copy
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
import torch.nn.functional as F
from torch import optim

from base.scheduler import GradualWarmupScheduler
from base.utils import ensure_dir


def pairwise_rank_loss(pred, label, margin=1.0, min_label_gap=0.4, return_count=False):
    pred = pred.view(-1)
    label = label.view(-1)

    label_diff = label.unsqueeze(1) - label.unsqueeze(0)
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)

    sign = torch.sign(label_diff)
    valid = torch.abs(label_diff) >= min_label_gap

    valid_count = valid.sum()
    if valid_count == 0:
        zero_loss = torch.tensor(0.0, device=pred.device)
        return (zero_loss, 0) if return_count else zero_loss

    loss = torch.relu(margin - sign * pred_diff)
    loss_value = loss[valid].mean()
    return (loss_value, int(valid_count.detach().cpu().item())) if return_count else loss_value


class Trainer(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.model_name = kwargs['model_name']
        self.model = kwargs['model'].to(self.device)
        self.save_path = kwargs['save_path']
        self.fold = kwargs['fold']

        self.min_epoch = kwargs['min_epoch']
        self.max_epoch = kwargs['max_epoch']
        self.start_epoch = 0
        self.early_stopping = int(kwargs.get('early_stopping', 20))
        self.early_stopping_counter = self.early_stopping
        self.scheduler = kwargs['scheduler']
        self.learning_rate = kwargs['learning_rate']
        self.min_learning_rate = kwargs['min_learning_rate']
        self.patience = int(kwargs.get('patience', 5))
        self.weight_decay = float(kwargs.get('weight_decay', 1e-5))

        self.criterion = kwargs['criterion']
        self.factor = kwargs['factor']
        self.verbose = kwargs['verbose']
        self.milestone = kwargs['milestone']
        self.load_best_at_each_epoch = kwargs['load_best_at_each_epoch']
        self.best_score_mode = kwargs.get('best_score_mode', 'default')

        self.best_epoch_info = None
        self.optimizer, self.scheduler = None, None
        if len(self.get_parameters()) != 0:
            self.init_optimizer_and_scheduler()

        self.batch_size = kwargs['batch_size']
        self.emotion = kwargs['emotion']
        self.metrics = kwargs['metrics']
        self.save_plot = kwargs['save_plot']
        self.label_mean = float(kwargs.get('label_mean', 0.0))
        self.label_std = float(kwargs.get('label_std', 1.0))
        self.huber_weight = float(kwargs.get('huber_weight', 0.70))
        self.rank_weight = float(kwargs.get('rank_weight', 0.30))
        self.rank_min_label_gap = float(kwargs.get('rank_min_label_gap', 0.4))
        rank_min_label_gap_real = kwargs.get('rank_min_label_gap_real', None)
        self.rank_min_label_gap_real = (
            None if rank_min_label_gap_real is None else float(rank_min_label_gap_real)
        )
        self.loss_weighting = kwargs.get('loss_weighting', 'fixed')
        self.lambda_attn = float(kwargs.get('lambda_attn', 0.0))

        # For checkpoint
        self.fit_finished = False
        self.fold_finished = False
        self.resume = False
        self.time_fit_start = None

        self.train_losses = []
        self.validate_losses = []
        self.csv_filename = None

    def denormalize_labels(self, values):
        return values * self.label_std + self.label_mean

    def train(self, **kwargs):
        kwargs['train_mode'] = True
        self.model.train()
        loss, result_dict = self.loop(**kwargs)
        return loss, result_dict

    def validate(self, **kwargs):
        kwargs['train_mode'] = False
        with torch.no_grad():
            self.model.eval()
            loss, result_dict = self.loop(**kwargs)
        return loss, result_dict

    def test(self, checkpoint_controller, **kwargs):
        """🌟 护法优化：删除了旧版繁琐的 predict_loop，直接调用干净的 loop 测成绩"""
        kwargs['train_mode'] = False
        with torch.no_grad():
            self.model.eval()
            loss, result_dict = self.loop(**kwargs)

            # 保存测试集成绩到 CSV
            checkpoint_controller.save_log_to_csv(
                kwargs.get('epoch', 0),
                mean_train_record=None,
                mean_validate_record=None,
                test_record=result_dict['overall']
            )
            return loss, result_dict

    def fit(self, dataloader_dict, checkpoint_controller, parameter_controller):
        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'ccc': -1e10,
                'score': -1e10
            }

        # ==== 🌟 初始化绘图专用的全局记录本 ====
        if not hasattr(self, 'train_record_dict'):
            self.train_record_dict = {
                'train_loss': [], 'val_loss': [],
                'train_ccc': [], 'val_ccc': [],
                'train_mae': [], 'val_mae': [],
                'train_rmse': [], 'val_rmse': [],
                'train_pcc': [], 'val_pcc': [],
                'lr': []
            }

        train_loss = 0.0

        for epoch in np.arange(start_epoch, self.max_epoch):

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            improvement = False

            # 🌟 只在热身结束（过了 min_epoch）之后，才允许因为学习率过低而停止！
            if epoch > self.min_epoch and self.optimizer.param_groups[0]['lr'] < self.min_learning_rate:
                print("\n【📉 学习率触底】已达到最低学习率限制，触发提前停止！")
                break

            time_epoch_start = time.time()

            if self.verbose:
                print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            # 获取训练和验证结果
            train_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            train_loss, train_record_dict = self.train(**train_kwargs)

            validate_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            validate_loss, validate_record_dict = self.validate(**validate_kwargs)

            # ==== 记录这一轮的数据 ====
            self.train_record_dict['train_loss'].append(train_loss)
            self.train_record_dict['val_loss'].append(validate_loss)
            self.train_record_dict['lr'].append(self.optimizer.param_groups[0]['lr'])

            for k in ['mae', 'ccc', 'rmse', 'pcc']:
                if k in train_record_dict['overall']:
                    self.train_record_dict[f'train_{k}'].append(train_record_dict['overall'][k])
                if k in validate_record_dict['overall']:
                    self.train_record_dict[f'val_{k}'].append(validate_record_dict['overall'][k])

            if validate_loss < 0:
                raise ValueError('validate loss negative')

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            validate_overall = validate_record_dict['overall']
            validate_ccc = validate_overall['ccc']
            validate_pcc = validate_overall['pcc'][0] if isinstance(validate_overall['pcc'], list) else validate_overall['pcc']
            validate_mae = validate_overall.get('mae', validate_loss)
            validate_rmse = validate_overall.get('rmse', validate_loss)
            validate_r2 = validate_overall.get('r2', 0.0)
            validate_pred_std = validate_overall.get('pred_std', 0.0)

            safe_pcc = validate_pcc if np.isfinite(validate_pcc) else 0.0
            safe_ccc = validate_ccc if np.isfinite(validate_ccc) else 0.0
            safe_rmse = validate_rmse if np.isfinite(validate_rmse) else validate_loss
            safe_mae = validate_mae if np.isfinite(validate_mae) else validate_loss

            if self.best_score_mode == 'corr_v2':
                validate_score = safe_pcc + 0.5 * safe_ccc - 0.02 * safe_rmse
            elif self.best_score_mode == 'val_mae':
                validate_score = -safe_mae
            else:
                validate_score = (
                    safe_pcc
                    + 0.2 * safe_ccc
                    - 0.05 * validate_mae
                )
            if self.best_score_mode != 'val_mae' and validate_pred_std < 1.0:
                validate_score -= 1.0

            if validate_score > self.best_epoch_info['score']:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_state_dict.pth"))
                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'ccc': validate_ccc,
                    'score': validate_score,
                    'epoch': epoch,
                }

            if self.verbose:
                print(
                    "\n Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} |  best={} | "
                    "improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        int(self.best_epoch_info['epoch']) + 1,
                        improvement,
                        self.early_stopping_counter))
                print(
                    "BestScore mode={} | val_mae={:.4f}, val_rmse={:.4f}, val_r2={:.4f}, "
                    "val_pcc={:.4f}, val_ccc={:.4f}, val_pred_std={:.4f}, validate_score={:.4f}".format(
                        self.best_score_mode,
                        validate_mae if np.isfinite(validate_mae) else 0.0,
                        safe_rmse,
                        validate_r2 if np.isfinite(validate_r2) else 0.0,
                        safe_pcc,
                        safe_ccc,
                        validate_pred_std if np.isfinite(validate_pred_std) else 0.0,
                        validate_score,
                    )
                )
                if self.best_score_mode == 'val_mae':
                    print("Checkpoint selection metric = validation MAE")
                train_parts = train_record_dict['overall']
                val_parts = validate_record_dict['overall']
                for prefix, parts in (("train", train_parts), ("val  ", val_parts)):
                    print(
                        "LossParts {} | huber_loss={:.4f}, rank_loss={:.4f}, "
                        "attn_reg={:.6f}, lambda_attn={:.6f}, base_loss={:.4f}, "
                        "log_var_huber={:.4f}, log_var_rank={:.4f}, "
                        "weight_huber={:.4f}, weight_rank={:.4f}, "
                        "fusion_logit={:.4f}, fusion_alpha_direct={:.4f}, "
                        "fusion_alpha_ordinal={:.4f}, total_loss={:.4f}, "
                        "pred_std={:.4f}, label_std={:.4f}, rank_pairs={:.0f}".format(
                            prefix,
                            parts.get('huber_loss', 0.0),
                            parts.get('rank_loss', 0.0),
                            parts.get('attn_reg', 0.0),
                            parts.get('lambda_attn', 0.0),
                            parts.get('base_loss', 0.0),
                            parts.get('log_var_huber', 0.0),
                            parts.get('log_var_rank', 0.0),
                            parts.get('weight_huber', 1.0),
                            parts.get('weight_rank', 1.0),
                            parts.get('fusion_logit', 0.0),
                            parts.get('score_fusion_alpha', 0.7),
                            parts.get('fusion_alpha_ordinal', 0.3),
                            parts.get('total_loss', 0.0),
                            parts.get('batch_pred_std', 0.0),
                            parts.get('batch_label_std', 0.0),
                            parts.get('valid_rank_pair_count', 0.0),
                        )
                    )
                alpha_value = train_parts.get('score_fusion_alpha', 0.7)
                if alpha_value < 0.2 or alpha_value > 0.9:
                    print("Warning: fusion alpha is close to single-head dominance.")

                print(train_record_dict['overall'])
                print(validate_record_dict['overall'])
                print("------")

            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict['overall'], validate_record_dict['overall'])

            # 提前停止逻辑
            if self.early_stopping and epoch > self.min_epoch:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            self.scheduler.step(metrics=validate_loss)
            self.start_epoch = epoch + 1

            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)
        self.model.load_state_dict(self.best_epoch_info['model_weights'])

        return train_loss, self.train_record_dict

    def loop(self, **kwargs):
        """Run one train/validate/test pass for regression-only SI prediction."""
        dataloader_dict = kwargs['dataloader_dict']
        epoch = kwargs['epoch']
        train_mode = kwargs['train_mode']
        dataloader = dataloader_dict['train'] if train_mode else (
            dataloader_dict['test'] if epoch is None else dataloader_dict['validate']
        )

        running_loss = 0.0
        total_samples = 0
        all_preds, all_labels = [], []
        loss_sums = {
            'huber_loss': 0.0,
            'rank_loss': 0.0,
            'attn_reg': 0.0,
            'lambda_attn': 0.0,
            'base_loss': 0.0,
            'total_loss': 0.0,
            'batch_pred_std': 0.0,
            'batch_label_std': 0.0,
            'rank_margin': 0.0,
            'valid_rank_pair_count': 0.0,
        }

        for batch_idx, (X, Y, trial_ids) in tqdm(enumerate(dataloader), total=len(dataloader)):
            b_size = X.size(0)
            if b_size == 0:
                continue

            inputs = X.to(self.device)
            if train_mode:
                inputs = inputs + torch.randn_like(inputs) * 0.05

            raw_labels = Y.to(self.device).float().view(b_size, 1)

            if train_mode:
                self.optimizer.zero_grad()
            else:
                for m in self.model.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        if m.running_var is not None:
                            torch.nan_to_num_(m.running_var, nan=1e-5, posinf=1e-5, neginf=1e-5)
                        if m.running_mean is not None:
                            torch.nan_to_num_(m.running_mean, nan=0.0, posinf=0.0, neginf=0.0)

            model_output = self.model(inputs)
            if isinstance(model_output, (tuple, list)):
                direct_score = model_output[0]
                ordinal_logits = model_output[1] if len(model_output) > 1 else None
            else:
                direct_score = model_output
                ordinal_logits = None

            expected_score = direct_score
            if ordinal_logits is not None:
                ordinal_score_real = torch.sum(torch.sigmoid(ordinal_logits), dim=1, keepdim=True)
                ordinal_score = (ordinal_score_real - self.label_mean) / (self.label_std + 1e-8)
                score_fusion_alpha = None
                if hasattr(self.model, 'get_score_fusion_alpha'):
                    score_fusion_alpha = self.model.get_score_fusion_alpha()
                if score_fusion_alpha is None:
                    expected_score = 0.7 * direct_score + 0.3 * ordinal_score
                else:
                    expected_score = score_fusion_alpha * direct_score + (1.0 - score_fusion_alpha) * ordinal_score

            huber_loss = F.smooth_l1_loss(expected_score, raw_labels)
            rank_gap = self.rank_min_label_gap
            rank_margin_for_log = self.rank_min_label_gap
            if self.rank_min_label_gap_real is not None:
                rank_gap = self.rank_min_label_gap_real / (self.label_std + 1e-8)
                rank_margin_for_log = self.rank_min_label_gap_real
            rank_loss, valid_rank_pair_count = pairwise_rank_loss(
                expected_score,
                raw_labels,
                margin=1.0,
                min_label_gap=rank_gap,
                return_count=True,
            )

            if (
                self.loss_weighting == 'homoscedastic_uncertainty'
                and hasattr(self.model, 'loss_weighting')
                and self.model.loss_weighting is not None
            ):
                loss = self.model.loss_weighting(huber_loss, rank_loss)
            else:
                loss = self.huber_weight * huber_loss + self.rank_weight * rank_loss

            base_loss = loss
            attn_reg = torch.tensor(0.0, device=self.device)
            if train_mode and self.lambda_attn > 0.0 and hasattr(self.model, 'attention_regularization'):
                attn_reg = self.model.attention_regularization()
                loss = loss + self.lambda_attn * attn_reg

            if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(expected_score).any():
                if train_mode:
                    self.optimizer.zero_grad()
                continue

            loss_val = loss.mean().item()
            running_loss += loss_val * b_size
            total_samples += b_size

            pred_std = torch.std(expected_score, unbiased=False)
            label_std = torch.std(raw_labels, unbiased=False)
            loss_sums['huber_loss'] += huber_loss.detach().item() * b_size
            loss_sums['rank_loss'] += rank_loss.detach().item() * b_size
            loss_sums['attn_reg'] += attn_reg.detach().item() * b_size
            loss_sums['lambda_attn'] += self.lambda_attn * b_size
            loss_sums['base_loss'] += base_loss.detach().mean().item() * b_size
            loss_sums['total_loss'] += loss.detach().mean().item() * b_size
            loss_sums['batch_pred_std'] += pred_std.detach().item() * b_size
            loss_sums['batch_label_std'] += label_std.detach().item() * b_size
            loss_sums['rank_margin'] += rank_margin_for_log * b_size
            loss_sums['valid_rank_pair_count'] += valid_rank_pair_count

            if self.loss_weighting == 'homoscedastic_uncertainty' and hasattr(self.model, 'loss_weighting') and self.model.loss_weighting is not None:
                weighting_values = self.model.loss_weighting.current_values()
                for key in ('log_var_huber', 'log_var_rank', 'weight_huber', 'weight_rank'):
                    loss_sums.setdefault(key, 0.0)
                    loss_sums[key] += weighting_values[key] * b_size

            if hasattr(self.model, 'get_score_fusion_alpha'):
                alpha_value = self.model.get_score_fusion_alpha()
                if alpha_value is not None:
                    fusion_values = self.model.get_score_fusion_values() if hasattr(self.model, 'get_score_fusion_values') else None
                    if fusion_values is None:
                        fusion_values = {
                            'fusion_logit': 0.0,
                            'fusion_alpha_direct': float(alpha_value.detach().cpu().item()),
                            'fusion_alpha_ordinal': float((1.0 - alpha_value).detach().cpu().item()),
                        }
                    loss_sums.setdefault('fusion_logit', 0.0)
                    loss_sums.setdefault('score_fusion_alpha', 0.0)
                    loss_sums.setdefault('fusion_alpha_ordinal', 0.0)
                    loss_sums['fusion_logit'] += fusion_values['fusion_logit'] * b_size
                    loss_sums['score_fusion_alpha'] += fusion_values['fusion_alpha_direct'] * b_size
                    loss_sums['fusion_alpha_ordinal'] += fusion_values['fusion_alpha_ordinal'] * b_size

            if train_mode:
                loss.backward()
                for p in self.model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            for i in range(b_size):
                pred_real = expected_score[i].detach().cpu().mean().item()
                label_real = raw_labels[i].detach().cpu().mean().item()
                if not np.isnan(pred_real) and not np.isnan(label_real):
                    all_preds.append(pred_real)
                    all_labels.append(label_real)

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        loss_means = {}
        for key, value in loss_sums.items():
            if key == 'valid_rank_pair_count':
                loss_means[key] = value
            else:
                loss_means[key] = value / total_samples if total_samples > 0 else 0.0

        all_preds = self.denormalize_labels(np.array(all_preds))
        all_labels = self.denormalize_labels(np.array(all_labels))

        mae_val, rmse_val, ccc_val = 0.0, 0.0, 0.0
        pcc_val = [0.0, 0.0]
        pred_std_val = float(np.std(all_preds)) if len(all_preds) > 0 else 0.0
        label_std_val = float(np.std(all_labels)) if len(all_labels) > 0 else 0.0
        pred_min_val = float(np.min(all_preds)) if len(all_preds) > 0 else 0.0
        pred_max_val = float(np.max(all_preds)) if len(all_preds) > 0 else 0.0
        pred_mean_val = float(np.mean(all_preds)) if len(all_preds) > 0 else 0.0
        label_min_val = float(np.min(all_labels)) if len(all_labels) > 0 else 0.0
        label_max_val = float(np.max(all_labels)) if len(all_labels) > 0 else 0.0
        label_mean_val = float(np.mean(all_labels)) if len(all_labels) > 0 else 0.0

        if len(all_preds) > 0:
            mae_val = float(np.mean(np.abs(all_preds - all_labels)))
            rmse_val = float(np.sqrt(np.mean((all_preds - all_labels) ** 2)))
            if np.std(all_preds) > 1e-6 and np.std(all_labels) > 1e-6:
                try:
                    p, p_v = pearsonr(all_preds, all_labels)
                    pcc_val = [float(p), float(p_v)]
                    mean_p, mean_l = np.mean(all_preds), np.mean(all_labels)
                    var_p, var_l = np.var(all_preds), np.var(all_labels)
                    cov = np.mean((all_preds - mean_p) * (all_labels - mean_l))
                    ccc_val = float((2 * cov) / (var_p + var_l + (mean_p - mean_l) ** 2))
                except Exception:
                    pass

        overall = {
            'mae': mae_val,
            'rmse': rmse_val,
            'pcc': pcc_val,
            'ccc': ccc_val,
            'pred_std': pred_std_val,
            'label_std': label_std_val,
            'pred_min': pred_min_val,
            'pred_max': pred_max_val,
            'pred_mean': pred_mean_val,
            'label_min': label_min_val,
            'label_max': label_max_val,
            'label_mean': label_mean_val,
            **loss_means,
        }

        epoch_result_dict = {
            'mae': mae_val,
            'rmse': rmse_val,
            'pcc': pcc_val,
            'ccc': ccc_val,
            'pred_std': pred_std_val,
            'label_std': label_std_val,
            'overall': overall,
        }

        return epoch_loss, epoch_result_dict

    def get_parameters(self):
        r"""
        Get the parameters to update.
        """
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        return params_to_update

    def init_optimizer_and_scheduler(self, epoch=0):
        self.optimizer = optim.AdamW(
            self.get_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        reduce_on_plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=self.patience, factor=self.factor)

        if epoch < self.min_epoch:
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, total_epoch=self.min_epoch, after_scheduler=reduce_on_plateau_scheduler)
        else:
            self.scheduler = reduce_on_plateau_scheduler
            if epoch > 0:
                self.scheduler.best = self.best_epoch_info['loss']
