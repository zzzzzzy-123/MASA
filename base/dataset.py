import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random


class EEG_Fusion_Dataset(Dataset):


    def __init__(self, data_list, feature_combo=['eeg_DE'], scaler=None, label_scaler=None):
        self.data_list = data_list
        self.feature_combo = feature_combo
        # 接收训练集传来的 scaler: {'mean': array, 'std': array}
        self.scaler = scaler
        self.label_scaler = label_scaler

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        folder_path, si_score, trial_id = self.data_list[index]
        fused_features = []

        for feat_name in self.feature_combo:
            feat_path = os.path.join(folder_path, f"{feat_name}.npy")
            data = np.load(feat_path)  # (40刺激, 特征维)

            if feat_name == 'eeg_PLI' and data.shape[1] == 140:
                # PLI is stored as 5 frequency bands x 28 edges. Convert it to
                # 28 edges x 5 bands so grouped Conv1d sees each edge's bands together.
                data_3d = data.reshape(40, 5, 28).transpose(0, 2, 1)
            else:
                if data.shape[1] % 8 != 0:
                    raise ValueError(f"{feat_name} feature dimension {data.shape[1]} is not divisible by 8")
                feat_dim = data.shape[1] // 8
                data_3d = data.reshape(40, 8, feat_dim)

            fused_features.append(data_3d)

        fused_3d = np.concatenate(fused_features, axis=-1)
        fused_3d = fused_3d.transpose(1, 2, 0)
        final_matrix = fused_3d.reshape(-1, 40)  # (Channels*Dims, 40)

        # 在此处进行特征级别的 Z-score！
        if self.scaler is not None:
            # 使用训练集计算出的均值和方差对当前样本进行标准化
            mean = self.scaler['mean']
            std = self.scaler['std']
            final_matrix = (final_matrix - mean) / (std + 1e-8)

        if self.label_scaler is not None:
            si_score = (si_score - self.label_scaler['mean']) / (self.label_scaler['std'] + 1e-8)

        final_tensor = torch.tensor(final_matrix, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor([si_score], dtype=torch.float32)

        return final_tensor, label_tensor, trial_id


# 增加一个辅助函数：用于在划分完 Fold 后，计算训练集的 mean 和 std
def compute_train_scaler(train_list, feature_combo):
    print("正在严格计算训练集特征分布 (防泄露 Z-score)...")
    all_matrices = []

    for item in train_list:
        folder_path = item[0]
        fused_features = []
        for feat_name in feature_combo:
            data = np.load(os.path.join(folder_path, f"{feat_name}.npy"))
            if feat_name == 'eeg_PLI' and data.shape[1] == 140:
                data_3d = data.reshape(40, 5, 28).transpose(0, 2, 1)
            else:
                if data.shape[1] % 8 != 0:
                    raise ValueError(f"{feat_name} feature dimension {data.shape[1]} is not divisible by 8")
                data_3d = data.reshape(40, 8, data.shape[1] // 8)
            fused_features.append(data_3d)

        fused_3d = np.concatenate(fused_features, axis=-1).transpose(1, 2, 0)
        final_matrix = fused_3d.reshape(-1, 40)
        all_matrices.append(final_matrix)

    # 把训练集所有病人的数据拼成一个巨大的张量
    huge_tensor = np.stack(all_matrices, axis=0)  # 形状: (N个病人, 特征总维数, 40次刺激)

    # 沿着病人和刺激次数计算均值和标准差，保留特征维度的独立性
    train_mean = np.mean(huge_tensor, axis=(0, 2), keepdims=True).squeeze(0)
    train_std = np.std(huge_tensor, axis=(0, 2), keepdims=True).squeeze(0)

    return {'mean': train_mean, 'std': train_std}


def compute_label_scaler(train_list):
    train_labels = np.asarray([item[1] for item in train_list], dtype=np.float32)
    label_mean = float(np.mean(train_labels))
    label_std = float(np.std(train_labels))
    risk_low, risk_high = np.quantile(train_labels, [1 / 3, 2 / 3])
    if label_std < 1e-6:
        label_std = 1.0

    return {
        'mean': label_mean,
        'std': label_std,
        'risk_low': float(risk_low),
        'risk_high': float(risk_high),
    }


class DataArranger(object):
    """
    负责扫描文件夹，读取 Excel 标签，并划分 5 折交叉验证。
    """

    def __init__(self, dataset_path, seed=42):
        self.seed = seed
        self.dataset_path = dataset_path
        # 强制指定五折交叉验证
        self.num_folds = 5

        # 生成带标签的病人列表
        self.processed_trials = self._generate_trial_list()

    def _generate_trial_list(self):
        print("\n【🚀 启动破釜沉舟程序 3.0】扫描纯净特征与真实 SI 标签！")

        compacted_dir = os.path.join(self.dataset_path, "Data_Processed", "compacted_EEG")
        excel_path = os.path.join(self.dataset_path, "minor_scale_2_gai.xlsx")

        valid_folders = [f for f in os.listdir(compacted_dir) if os.path.isdir(os.path.join(compacted_dir, f))]

        label_dict = {}
        try:
            df = pd.read_excel(excel_path)
            df.columns = df.columns.str.strip()
            for idx, row in df.iterrows():
                xuhao = str(int(row['序号']))
                folder_name = f"P{xuhao}-T1"
                label_dict[folder_name] = float(row['SI'])
            print(f"【📊 标签雷达】成功从 Excel 中扫描并锁定了 {len(label_dict)} 个病人的 SI 真实分数！")
        except Exception as e:
            print(f"🚨 读取 Excel 失败，请检查路径或表格是否被占用: {e}")

        processed_trials = []
        for folder_name in valid_folders:
            full_folder_path = os.path.join(compacted_dir, folder_name)
            # 只要文件夹里存在任何 .npy 文件，就认为有效
            if any(f.endswith('.npy') for f in os.listdir(full_folder_path)):
                si_score = label_dict.get(folder_name, 0.0)
                processed_trials.append([full_folder_path, si_score, folder_name])

        total_trials = len(processed_trials)
        print(f"【✅ 弹药填装完毕】强行将 {total_trials} 个带有真实 SI 标签的样本推入训练场！")

        return processed_trials

    def _stratified_fold_indices(self, labels, num_folds, seed):
        labels = np.asarray(labels, dtype=np.float32)
        bins = np.quantile(labels, [1 / 3, 2 / 3])
        strata = np.digitize(labels, bins)

        folds = [[] for _ in range(num_folds)]
        rng = random.Random(seed)
        for stratum in sorted(set(strata.tolist())):
            indices = [idx for idx, value in enumerate(strata) if value == stratum]
            rng.shuffle(indices)
            for offset, idx in enumerate(indices):
                folds[offset % num_folds].append(idx)

        return folds

    def _stratified_val_indices(self, labels, val_ratio, seed):
        labels = np.asarray(labels, dtype=np.float32)
        bins = np.quantile(labels, [1 / 3, 2 / 3])
        strata = np.digitize(labels, bins)

        rng = random.Random(seed)
        val_indices = []
        for stratum in sorted(set(strata.tolist())):
            indices = [idx for idx, value in enumerate(strata) if value == stratum]
            rng.shuffle(indices)
            n_val = max(1, int(round(len(indices) * val_ratio)))
            val_indices.extend(indices[:n_val])

        return set(val_indices)

    def get_fold(self, fold_idx):
        """
        获取指定 fold_idx (0 到 4) 的训练集、验证集、测试集列表。
        划分比例：测试集 20%，剩下的 80% 中，抽取 20% 作为验证集，80% 作为训练集。
        """
        assert 0 <= fold_idx < self.num_folds, "fold_idx 必须在 0 到 4 之间"

        trials = self.processed_trials.copy()
        labels = [float(item[1]) for item in trials]

        folds = self._stratified_fold_indices(labels, self.num_folds, self.seed)
        test_indices = set(folds[fold_idx])

        test_list = [trials[idx] for idx in sorted(test_indices)]
        train_val_list = [item for idx, item in enumerate(trials) if idx not in test_indices]
        train_val_labels = [float(item[1]) for item in train_val_list]

        val_indices = self._stratified_val_indices(
            train_val_labels,
            val_ratio=0.2,
            seed=self.seed + fold_idx
        )

        train_list = [
            item for idx, item in enumerate(train_val_list)
            if idx not in val_indices
        ]
        val_list = [
            item for idx, item in enumerate(train_val_list)
            if idx in val_indices
        ]

        return {
            'train': train_list,
            'validate': val_list,
            'test': test_list
        }
