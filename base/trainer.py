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


def pairwise_rank_loss(pred, label, margin=1.0, min_label_gap=0.4):
    pred = pred.view(-1)
    label = label.view(-1)

    label_diff = label.unsqueeze(1) - label.unsqueeze(0)
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)

    sign = torch.sign(label_diff)
    valid = torch.abs(label_diff) >= min_label_gap

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    loss = torch.relu(margin - sign * pred_diff)
    return loss[valid].mean()


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
        self.early_stopping = 20
        self.early_stopping_counter = self.early_stopping
        self.scheduler = kwargs['scheduler']
        self.learning_rate = kwargs['learning_rate']
        self.min_learning_rate = kwargs['min_learning_rate']
        self.patience = 5

        self.criterion = kwargs['criterion']
        self.factor = kwargs['factor']
        self.verbose = kwargs['verbose']
        self.milestone = kwargs['milestone']
        self.load_best_at_each_epoch = kwargs['load_best_at_each_epoch']

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
        self.risk_low = float(kwargs.get('risk_low', self.label_mean - 0.43 * self.label_std))
        self.risk_high = float(kwargs.get('risk_high', self.label_mean + 0.43 * self.label_std))

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

            validate_ccc = validate_record_dict['overall']['ccc']

            validate_overall = validate_record_dict['overall']
            validate_pcc = validate_overall['pcc'][0] if isinstance(validate_overall['pcc'], list) else validate_overall['pcc']
            validate_pred_std = validate_overall.get('pred_std', 0.0)
            validate_score = (
                validate_pcc
                + 0.2 * validate_overall['ccc']
                - 0.05 * validate_overall.get('mae', validate_loss)
            )
            if validate_pred_std < 1.0:
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
        """🌟 核心数据流动与 Loss 计算枢纽"""
        dataloader_dict, epoch, train_mode = kwargs['dataloader_dict'], kwargs['epoch'], kwargs['train_mode']
        dataloader = dataloader_dict['train'] if train_mode else (
            dataloader_dict['test'] if epoch is None else dataloader_dict['validate'])

        running_loss = 0.0
        total_samples = 0
        all_preds, all_labels = [], []
        all_risk_preds, all_risk_labels = [], []

        for batch_idx, (X, Y, trial_ids) in tqdm(enumerate(dataloader), total=len(dataloader)):

            b_size = X.size(0)
            if b_size == 0: continue

            inputs = X.to(self.device)

            # 数据增强：训练模式下给输入特征加入高斯噪声
            if train_mode:
                noise = torch.randn_like(inputs) * 0.05
                inputs = inputs + noise

            # 真实分数，形状 (Batch, 1)
            raw_labels = Y.to(self.device).float().view(b_size, 1)

            if train_mode:
                self.optimizer.zero_grad()
            else:
                # 终极杀毒：清洗 BatchNorm 的僵尸缓存
                for m in self.model.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        if m.running_var is not None:
                            torch.nan_to_num_(m.running_var, nan=1e-5, posinf=1e-5, neginf=1e-5)
                        if m.running_mean is not None:
                            torch.nan_to_num_(m.running_mean, nan=0.0, posinf=0.0, neginf=0.0)

            model_output = self.model(inputs)
            if len(model_output) == 3:
                direct_score, ordinal_logits, risk_logits = model_output
            else:
                ordinal_logits, direct_score = model_output
                risk_logits = None
            ordinal_score_real = torch.sum(torch.sigmoid(ordinal_logits), dim=1, keepdim=True)
            ordinal_score = (ordinal_score_real - self.label_mean) / (self.label_std + 1e-8)
            expected_score = 0.7 * direct_score + 0.3 * ordinal_score

            huber_loss = F.smooth_l1_loss(expected_score, raw_labels)
            rank_loss = pairwise_rank_loss(expected_score, raw_labels, margin=1.0, min_label_gap=0.4)
            pred_std = torch.std(expected_score, unbiased=False)
            label_std = torch.std(raw_labels, unbiased=False)
            std_loss = torch.relu(0.6 * label_std - pred_std)
            real_labels = raw_labels * self.label_std + self.label_mean
            thresholds = torch.arange(
                ordinal_logits.size(1),
                device=self.device,
                dtype=real_labels.dtype
            ).view(1, -1)
            ordinal_targets = (real_labels > thresholds).float()
            cls_loss = torch.tensor(0.0, device=self.device)
            sep_loss = torch.tensor(0.0, device=self.device)
            if risk_logits is not None:
                flat_real_labels = real_labels.view(-1)
                valid_cls_mask = (flat_real_labels <= self.risk_low) | (flat_real_labels >= self.risk_high)
                risk_labels = (flat_real_labels >= self.risk_high).long()
                if valid_cls_mask.any():
                    cls_loss = F.cross_entropy(risk_logits[valid_cls_mask], risk_labels[valid_cls_mask])
                low_mask = flat_real_labels <= self.risk_low
                high_mask = flat_real_labels >= self.risk_high
                if low_mask.any() and high_mask.any():
                    low_mean = torch.mean(expected_score.view(-1)[low_mask])
                    high_mean = torch.mean(expected_score.view(-1)[high_mask])
                    target_sep = 1.5 / (self.label_std + 1e-8)
                    sep_loss = torch.relu(target_sep - (high_mean - low_mean))

            loss = (
                0.42 * huber_loss
                + 0.25 * rank_loss
                + 0.23 * cls_loss
                + 0.05 * sep_loss
                + 0.05 * std_loss
            )

            # 保护大脑：如果损失爆炸，跳过该 batch
            if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(expected_score).any():
                if train_mode: self.optimizer.zero_grad()
                continue

            loss_val = loss.mean().item()
            running_loss += loss_val * b_size
            total_samples += b_size

            if train_mode:
                loss.backward()
                # 刮骨疗毒，保留健康梯度
                for p in self.model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # 记录真实尺度的预测值用于算 Metric
            for i in range(b_size):
                pred_real = expected_score[i].detach().cpu().mean().item()
                label_real = raw_labels[i].detach().cpu().mean().item()

                if not np.isnan(pred_real) and not np.isnan(label_real):
                    all_preds.append(pred_real)
                    all_labels.append(label_real)

            if risk_logits is not None:
                valid_np = valid_cls_mask.detach().cpu().numpy()
                risk_pred_np = torch.argmax(risk_logits, dim=1).detach().cpu().numpy()
                risk_label_np = risk_labels.detach().cpu().numpy()
                all_risk_preds.extend(risk_pred_np[valid_np].tolist())
                all_risk_labels.extend(risk_label_np[valid_np].tolist())

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0

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
        risk_acc_val = 0.0
        risk_f1_val = 0.0
        if len(all_risk_labels) > 0:
            risk_preds_np = np.asarray(all_risk_preds)
            risk_labels_np = np.asarray(all_risk_labels)
            risk_acc_val = float(np.mean(risk_preds_np == risk_labels_np))
            f1_values = []
            for cls_idx in [0, 1]:
                tp = np.sum((risk_preds_np == cls_idx) & (risk_labels_np == cls_idx))
                fp = np.sum((risk_preds_np == cls_idx) & (risk_labels_np != cls_idx))
                fn = np.sum((risk_preds_np != cls_idx) & (risk_labels_np == cls_idx))
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1_values.append(2 * precision * recall / (precision + recall + 1e-8))
            risk_f1_val = float(np.mean(f1_values))

        low_mask = all_labels <= self.risk_low
        high_mask = all_labels >= self.risk_high
        low_pred_mean = float(np.mean(all_preds[low_mask])) if np.any(low_mask) else 0.0
        high_pred_mean = float(np.mean(all_preds[high_mask])) if np.any(high_mask) else 0.0

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
                except:
                    pass

        epoch_result_dict = {
            'mae': mae_val, 'rmse': rmse_val, 'pcc': pcc_val, 'ccc': ccc_val,
            'pred_std': pred_std_val, 'label_std': label_std_val,
            'overall': {
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
                'risk_acc': risk_acc_val,
                'risk_f1': risk_f1_val,
                'low_pred_mean': low_pred_mean,
                'high_pred_mean': high_pred_mean
            }
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
        self.optimizer = optim.AdamW(self.get_parameters(), lr=self.learning_rate, weight_decay=1e-5)

        reduce_on_plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=self.patience, factor=self.factor)

        if epoch < self.min_epoch:
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, total_epoch=self.min_epoch, after_scheduler=reduce_on_plateau_scheduler)
        else:
            self.scheduler = reduce_on_plateau_scheduler
            if epoch > 0:
                self.scheduler.best = self.best_epoch_info['loss']
