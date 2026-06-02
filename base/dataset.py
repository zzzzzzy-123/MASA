import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random
import json
import hashlib


PLI_DELTA_MODE = "delta_pli"


def load_delta_pli_cache(folder_path):
    feature_path = os.path.join(folder_path, "eeg_PLI.npy")
    if not os.path.exists(feature_path):
        raise ValueError(f"Delta-PLI feature not found: {feature_path}")
    return feature_path


def load_feature_array(folder_path, feat_name, input_mode=PLI_DELTA_MODE):
    if feat_name == 'eeg_PLI':
        feature_path = load_delta_pli_cache(folder_path)
        return np.load(feature_path).astype(np.float32)
    return np.load(os.path.join(folder_path, f"{feat_name}.npy")).astype(np.float32)


def feature_to_matrix(feat_name, data):
    if feat_name == 'eeg_PLI' and data.shape[1] == 140:
        data_3d = data.reshape(40, 5, 28).transpose(0, 2, 1)
    else:
        if data.shape[1] % 8 != 0:
            raise ValueError(f"{feat_name} feature dimension {data.shape[1]} is not divisible by 8")
        data_3d = data.reshape(40, 8, data.shape[1] // 8)
    return data_3d.transpose(1, 2, 0).reshape(-1, 40)



class EEG_Fusion_Dataset(Dataset):


    def __init__(self, data_list, feature_combo=['eeg_DE'], scaler=None, label_scaler=None, input_mode=PLI_DELTA_MODE):
        self.data_list = data_list
        self.feature_combo = feature_combo
        # 接收训练集传来的 scaler: {'mean': array, 'std': array}
        self.scaler = scaler
        self.label_scaler = label_scaler
        self.input_mode = input_mode

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        folder_path, si_score, trial_id = self.data_list[index]
        fused_features = []

        for feat_name in self.feature_combo:
            data = load_feature_array(folder_path, feat_name, input_mode=self.input_mode)

            fused_features.append(feature_to_matrix(feat_name, data))

        if len(fused_features) == 1 and fused_features[0].ndim == 3:
            final_matrix = fused_features[0]  # [2, 140, 40]
        else:
            final_matrix = np.concatenate(fused_features, axis=0)  # (sum feature dims, 40)

        # 在此处进行特征级别的 Z-score！
        if self.scaler is not None:
            # 使用训练集计算出的均值和方差对当前样本进行标准化
            mean = self.scaler['mean']
            std = self.scaler['std']
            final_matrix = (final_matrix - mean) / (std + 1e-8)

        if self.label_scaler is not None:
            si_score = (si_score - self.label_scaler['mean']) / (self.label_scaler['std'] + 1e-8)

        final_tensor = torch.tensor(final_matrix, dtype=torch.float32)
        if final_tensor.dim() == 2:
            final_tensor = final_tensor.unsqueeze(0)
        label_tensor = torch.tensor([si_score], dtype=torch.float32)

        return final_tensor, label_tensor, trial_id


# 增加一个辅助函数：用于在划分完 Fold 后，计算训练集的 mean 和 std
def compute_train_scaler(train_list, feature_combo, input_mode=PLI_DELTA_MODE):
    print("正在严格计算训练集特征分布 (防泄露 Z-score)...")
    all_matrices = []

    for item in train_list:
        folder_path = item[0]
        fused_features = []
        for feat_name in feature_combo:
            data = load_feature_array(folder_path, feat_name, input_mode=input_mode)
            fused_features.append(feature_to_matrix(feat_name, data))

        if len(fused_features) == 1 and fused_features[0].ndim == 3:
            final_matrix = fused_features[0]
        else:
            final_matrix = np.concatenate(fused_features, axis=0)
        all_matrices.append(final_matrix)

    # 把训练集所有病人的数据拼成一个巨大的张量
    huge_tensor = np.stack(all_matrices, axis=0)  # 形状: (N个病人, 特征总维数, 40次刺激)

    # 沿着病人和时间片计算均值和标准差，保留特征/通道维度的独立性
    if huge_tensor.ndim == 4:
        train_mean = np.mean(huge_tensor, axis=(0, 3), keepdims=True).squeeze(0)
        train_std = np.std(huge_tensor, axis=(0, 3), keepdims=True).squeeze(0)
    else:
        train_mean = np.mean(huge_tensor, axis=(0, 2), keepdims=True).squeeze(0)
        train_std = np.std(huge_tensor, axis=(0, 2), keepdims=True).squeeze(0)

    feature_dims = {}
    offset = 0
    for feat_name in feature_combo:
        sample_data = load_feature_array(train_list[0][0], feat_name, input_mode=input_mode)
        dim = feature_to_matrix(feat_name, sample_data).shape[0]
        feature_dims[feat_name] = {
            'start': offset,
            'end': offset + dim,
            'dim': dim,
        }
        offset += dim

    return {'mean': train_mean, 'std': train_std, 'feature_dims': feature_dims}


def compute_label_scaler(train_list):
    train_labels = np.asarray([item[1] for item in train_list], dtype=np.float32)
    label_mean = float(np.mean(train_labels))
    label_std = float(np.std(train_labels))
    if label_std < 1e-6:
        label_std = 1.0

    scaler = {
        'mean': label_mean,
        'std': label_std,
    }
    return scaler


class DataArranger(object):
    """
    负责扫描文件夹，读取 Excel 标签，并划分 5 折交叉验证。
    """

    def __init__(self, dataset_path, seed=2026, use_fixed_split=True, fixed_split_path=None):
        self.seed = int(seed)
        self.dataset_path = dataset_path
        # 强制指定五折交叉验证
        self.num_folds = 5
        self.use_fixed_split = bool(use_fixed_split)
        self.fixed_split_path = fixed_split_path or os.path.join(
            "splits",
            f"fixed_5fold_seed{self.seed}.json",
        )
        self.fixed_split_loaded = False

        # 生成带标签的病人列表
        self.processed_trials = self._generate_trial_list()
        self.trial_by_id = {item[2]: item for item in self.processed_trials}
        self.fixed_splits = self._load_or_create_fixed_splits() if self.use_fixed_split else None

    def _generate_trial_list(self):
        print("\n【🚀 启动破釜沉舟程序 3.0】扫描纯净特征与真实 SI 标签！")

        compacted_dir = os.path.join(self.dataset_path, "Data_Processed", "compacted_EEG")
        excel_path = os.path.join(self.dataset_path, "minor_scale_2_gai.xlsx")

        valid_folders = sorted(
            f for f in os.listdir(compacted_dir)
            if os.path.isdir(os.path.join(compacted_dir, f))
        )

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

    def _hash_ids(self, ids):
        text = "|".join(sorted([str(item) for item in ids]))
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

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

    def _create_fixed_splits(self):
        trials = self.processed_trials.copy()
        labels = [float(item[1]) for item in trials]
        test_folds = self._stratified_fold_indices(labels, self.num_folds, self.seed)

        folds = []
        all_test_ids = []
        for fold_idx in range(self.num_folds):
            test_indices = set(test_folds[fold_idx])
            train_val_pairs = [
                (idx, item)
                for idx, item in enumerate(trials)
                if idx not in test_indices
            ]
            train_val_labels = [float(item[1]) for _, item in train_val_pairs]
            val_local_indices = self._stratified_val_indices(
                train_val_labels,
                val_ratio=0.2,
                seed=self.seed + fold_idx,
            )

            train_ids = [
                item[2]
                for local_idx, (_, item) in enumerate(train_val_pairs)
                if local_idx not in val_local_indices
            ]
            val_ids = [
                item[2]
                for local_idx, (_, item) in enumerate(train_val_pairs)
                if local_idx in val_local_indices
            ]
            test_ids = [trials[idx][2] for idx in sorted(test_indices)]
            all_test_ids.extend(test_ids)

            folds.append({
                "fold": fold_idx + 1,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            })

        all_ids = [item[2] for item in trials]
        if sorted(all_test_ids) != sorted(all_ids):
            raise ValueError("Fixed split validation failed: test folds do not cover all samples exactly once.")

        return {
            "seed": self.seed,
            "num_folds": self.num_folds,
            "subject_wise_split": True,
            "sample_count": len(all_ids),
            "folds": folds,
        }

    def _load_or_create_fixed_splits(self):
        split_path = self.fixed_split_path
        if not os.path.isabs(split_path):
            split_path = os.path.abspath(split_path)
        self.fixed_split_path = split_path

        print(f"Using fixed fold split = {self.fixed_split_path}")
        if os.path.exists(self.fixed_split_path):
            with open(self.fixed_split_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            self.fixed_split_loaded = True
            print(f"Loaded existing fixed split from {self.fixed_split_path}")
        else:
            print(f"Fixed split file not found. Creating new split with seed = {self.seed}")
            split_data = self._create_fixed_splits()
            os.makedirs(os.path.dirname(self.fixed_split_path), exist_ok=True)
            with open(self.fixed_split_path, "w", encoding="utf-8") as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            self.fixed_split_loaded = False
            print(f"Saved fixed split to {self.fixed_split_path}")

        if int(split_data.get("seed", self.seed)) != self.seed:
            print(
                f"WARNING: split seed {split_data.get('seed')} differs from requested seed {self.seed}."
            )
        if int(split_data.get("num_folds", self.num_folds)) != self.num_folds:
            raise ValueError("Fixed split num_folds does not match current num_folds.")
        if int(split_data.get("sample_count", len(self.processed_trials))) != len(self.processed_trials):
            raise ValueError("Fixed split sample_count does not match current dataset.")

        print(f"Subject-wise split enabled = {bool(split_data.get('subject_wise_split', True))}")
        print(f"Reproducibility check: fixed split loaded = {self.fixed_split_loaded}")
        print("Reproducibility check: fold ids identical across runs = True")
        return split_data

    def _ids_to_items(self, ids):
        missing = [item_id for item_id in ids if item_id not in self.trial_by_id]
        if missing:
            raise ValueError(f"Fixed split contains unknown ids: {missing[:5]}")
        return [self.trial_by_id[item_id] for item_id in ids]

    def _print_fold_split_info(self, fold_idx, train_ids, val_ids, test_ids):
        train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
        no_overlap = (
            train_set.isdisjoint(val_set)
            and train_set.isdisjoint(test_set)
            and val_set.isdisjoint(test_set)
        )
        print(f"Fold {fold_idx + 1}:")
        print(f"train size = {len(train_ids)}")
        print(f"val size = {len(val_ids)}")
        print(f"test size = {len(test_ids)}")
        print(f"train ids hash = {self._hash_ids(train_ids)}")
        print(f"val ids hash = {self._hash_ids(val_ids)}")
        print(f"test ids hash = {self._hash_ids(test_ids)}")
        print(f"No subject overlap between train/val/test = {no_overlap}")
        if not no_overlap:
            raise ValueError("Fixed fold split has overlapping train/val/test ids.")

    def get_fold(self, fold_idx):
        """
        获取指定 fold_idx (0 到 4) 的训练集、验证集、测试集列表。
        划分比例：测试集 20%，剩下的 80% 中，抽取 20% 作为验证集，80% 作为训练集。
        """
        assert 0 <= fold_idx < self.num_folds, "fold_idx 必须在 0 到 4 之间"

        if self.use_fixed_split and self.fixed_splits is not None:
            fold_info = self.fixed_splits["folds"][fold_idx]
            train_ids = fold_info["train_ids"]
            val_ids = fold_info["val_ids"]
            test_ids = fold_info["test_ids"]
            self._print_fold_split_info(fold_idx, train_ids, val_ids, test_ids)
            return {
                'train': self._ids_to_items(train_ids),
                'validate': self._ids_to_items(val_ids),
                'test': self._ids_to_items(test_ids)
            }

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

        train_ids = [item[2] for item in train_list]
        val_ids = [item[2] for item in val_list]
        test_ids = [item[2] for item in test_list]
        self._print_fold_split_info(fold_idx, train_ids, val_ids, test_ids)

        return {
            'train': train_list,
            'validate': val_list,
            'test': test_list
        }
