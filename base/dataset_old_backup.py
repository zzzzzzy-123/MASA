import torch
import os
from operator import itemgetter

import numpy as np
import random
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import Dataset

from base.utils import roll_list


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, Imgs):
        L, C, H, W = Imgs.shape

        tensor = []
        for k in range(L):
            img = self.normalize(Imgs[k, :, :, :])
            tensor.append(img)

        return torch.stack(tensor, dim=0)


class MyDataset(Dataset):
    def __init__(self, data_list, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode,
                 mean_std=None,
                 time_delay=0, feature_extraction=0):
        self.data_list = data_list
        self.continuous_label_dim = continuous_label_dim
        self.mean_std = mean_std
        self.mean_std_info = 0
        self.time_delay = time_delay
        self.modality = modality
        self.multiplier = multiplier
        self.feature_dimension = feature_dimension
        self.feature_extraction = feature_extraction
        self.window_length = window_length
        self.mode = mode
        self.transform_dict = {}
        self.get_3D_transforms()

    def get_feature_transform(self, feature):
        if "video" in feature:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean_std[feature]['mean']],
                                     std=[self.mean_std[feature]['std']])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        return transform

    def get_3D_transforms(self):
        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        for feature in self.modality:
            if "continuous_label" not in feature and "video" not in feature:
                self.transform_dict[feature] = self.get_feature_transform(feature)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path, trial, length, index = self.data_list[index]

        examples = {}

        for feature in self.modality:
            examples[feature] = self.get_example(path, length, index, feature)

        if len(index) < self.window_length:
            index = np.arange(self.window_length)

        return examples, trial, length, index

    def get_example(self, path, length, index, feature):

        x = random.randint(0, self.multiplier[feature] - 1)
        random_index = index * self.multiplier[feature] + x

        if length < self.window_length:
            shape = (self.window_length,) + self.feature_dimension[feature]
            dtype = np.float32
            if feature == "video":
                dtype = np.int8
            example = np.zeros(shape=shape, dtype=dtype)
            example[index] = self.load_data(path, random_index, feature)
        else:
            example = self.load_data(path, random_index, feature)

        if "continuous_label" in feature and self.time_delay != 0:
            example = np.concatenate(
                (example[self.time_delay:, :],
                 np.repeat(example[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if "continuous_label" not in feature:
            example = self.transform_dict[feature](np.asarray(example, dtype=np.float32))

        return example

    def load_data(self, path, indices, feature):
        filename = os.path.join(path, feature + ".npy")
        data = np.zeros(((len(indices),) + self.feature_dimension[feature]), dtype=np.float32)

        if os.path.isfile(filename):
            if self.feature_extraction:
                data = np.load(filename, mmap_mode='c')
            else:
                data = np.load(filename, mmap_mode='c')[indices]

            if "continuous_label" in feature:
                data = self.processing_label(data)
        else:
            # ==== 🚨 增加防雷警报：找不到文件必须大喊，绝不吃哑巴亏！ ====
            if "label" not in feature:
                print(f"\n🚨 [致命错误] 找不到特征文件: {filename} ！程序正在返回全0假数据！\n")

        return data

    def processing_label(self, label):
        label = label[:, self.continuous_label_dim]
        if label.ndim == 1:
            label = label[:, None]
        return label


class MyDatasetPreLoad(Dataset):
    def __init__(self, data_list, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode,
                 mean_std=None,
                 time_delay=0, feature_extraction=0):
        self.data_list = data_list
        self.continuous_label_dim = continuous_label_dim
        self.mean_std = mean_std
        self.mean_std_info = 0
        self.time_delay = time_delay
        self.modality = modality
        self.multiplier = multiplier
        self.feature_dimension = feature_dimension
        self.feature_extraction = feature_extraction
        self.window_length = window_length
        self.mode = mode
        self.transform_dict = {}
        self.get_3D_transforms()

        self.examples_list, self.trial_list, self.length_list, self.index_list = self.pre_load()

    def get_feature_transform(self, feature):
        if "video" in feature:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean_std[feature]['mean']],
                                     std=[self.mean_std[feature]['std']])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        return transform

    def get_3D_transforms(self):
        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        for feature in self.modality:
            if "continuous_label" not in feature and "video" not in feature:
                self.transform_dict[feature] = self.get_feature_transform(feature)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.examples_list[index], self.trial_list[index], self.length_list[index], self.index_list[index]

    def pre_load(self):
        examples_list, trial_list, length_list, index_list = [], [], [], []
        for index in range(len(self.data_list)):
            path, trial, length, index = self.data_list[index]
            examples = {}
            for feature in self.modality:
                examples[feature] = self.get_example(path, length, index, feature)

            if len(index) < self.window_length:
                index = np.arange(self.window_length)

            examples_list.append(examples)
            trial_list.append(trial)
            length_list.append(length)
            index_list.append(index)

        return examples_list, trial_list, length_list, index_list

    def get_example(self, path, length, index, feature):
        x = random.randint(0, self.multiplier[feature] - 1)
        random_index = index * self.multiplier[feature] + x

        if length < self.window_length:
            shape = (self.window_length,) + self.feature_dimension[feature]
            dtype = np.float32
            if feature == "video":
                dtype = np.int8
            example = np.zeros(shape=shape, dtype=dtype)
            example[index] = self.load_data(path, random_index, feature)
        else:
            example = self.load_data(path, random_index, feature)

        if "continuous_label" in feature and self.time_delay != 0:
            example = np.concatenate(
                (example[self.time_delay:, :],
                 np.repeat(example[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if "continuous_label" not in feature:
            example = self.transform_dict[feature](np.asarray(example, dtype=np.float32))

        return example

    def load_data(self, path, indices, feature):
        filename = os.path.join(path, feature + ".npy")
        data = np.zeros(((len(indices),) + self.feature_dimension[feature]), dtype=np.float32)

        if os.path.isfile(filename):
            if self.feature_extraction:
                data = np.load(filename, mmap_mode='c')
            else:
                data = np.load(filename, mmap_mode='c')[indices]

            if "continuous_label" in feature:
                data = self.processing_label(data)
        else:
            # ==== 🚨 增加防雷警报 ====
            if "label" not in feature:
                print(f"\n🚨 [致命错误] 找不到特征文件: {filename} ！程序正在返回全0假数据！\n")

        return data

    def processing_label(self, label):
        label = label[:, self.continuous_label_dim]
        if label.ndim == 1:
            label = label[:, None]
        return label


class DataArranger(object):
    def __init__(self, dataset_info, dataset_path, debug, task, case, seed):
        self.task = task
        self.case = case
        self.seed = seed
        self.dataset_info = dataset_info
        self.debug = debug
        self.trial_list = self.generate_raw_trial_list(dataset_path)
        self.partition_range = self.partition_range_fn()
        self.fold_to_partition = self.assign_fold_to_partition()

    def generate_partitioned_trial_list(self, window_length, hop_length, fold, windowing=True):
        import os, random, numpy as np
        import pandas as pd

        print("\n【🚀 启动破釜沉舟程序 2.0】抛弃烂尾档案，直接从 Excel 提取绝密标签！")

        compacted_dir = r"C:\Users\云瑾\Desktop\data\Data_Processed\compacted_EEG"
        excel_path = r"C:\Users\云瑾\Desktop\data\minor_scale_2_gai.xlsx"

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
            npy_path = os.path.join(full_folder_path, "eeg_DE.npy")

            # ==== 只处理存在 EEG 文件的文件夹 ====
            if os.path.exists(npy_path):
                # ==== 检查是否存在标签，如果没有就跳过 ====
                if folder_name not in label_dict:
                    print(f"⚠️ 警告：找不到 {folder_name} 对应 SI 标签，已跳过")
                    continue  # 跳过无标签样本

                si_score = label_dict[folder_name]

                my_info = {}
                my_info['eeg_DE_npy_path'] = npy_path
                my_info['trial_id'] = folder_name  # 实际是被试 ID
                my_info['SI'] = np.array([si_score], dtype=np.float32)
                my_info['class_label'] = np.array([si_score], dtype=np.float32)
                my_info['continuous_label'] = np.array([si_score], dtype=np.float32)

                try:
                    data_mat = np.load(npy_path)
                    dims = list(data_mat.shape)
                    if 48 in dims:
                        dims.remove(48)
                    length = dims[0] if dims else 100
                except Exception as e:
                    print(f"⚠️ 读取 EEG 文件失败 {npy_path}，使用默认长度 100: {e}")
                    length = 100

                my_info['eeg_length'] = length

                # ==== 添加到 processed_subjects（原来的 processed_trials） ====
                processed_subjects.append([full_folder_path, my_info, length])

        total_trials = len(processed_trials)
        print(f"【✅ 弹药填装完毕】强行将 {total_trials} 个带有真实 SI 标签的样本推入训练场！")

        random.Random(self.seed).shuffle(processed_trials)
        num_folds = 5
        fold_size = max(1, total_trials // num_folds)

        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < num_folds - 1 else total_trials
        test_trials = processed_trials[test_start:test_end]

        remaining_trials = processed_trials[:test_start] + processed_trials[test_end:]
        random.Random(self.seed + fold).shuffle(remaining_trials)
        num_train = int(len(remaining_trials) * 0.8)

        train_trials = remaining_trials[:num_train]
        validate_trials = remaining_trials[num_train:]

        partitioned_trial = {
            'train': train_trials,
            'validate': validate_trials,
            'test': test_trials
        }

        windowed_partitioned_trial = {key: [] for key in partitioned_trial}
        for partition, trials in partitioned_trial.items():
            for path, trial, length in trials:
                safe_window = min(window_length, max(1, length))
                if windowing:
                    windowed_indices = self.windowing(np.arange(length), window_length=safe_window,
                                                      hop_length=hop_length)
                else:
                    windowed_indices = self.windowing(np.arange(length), window_length=max(1, length),
                                                      hop_length=hop_length)

                if not windowed_indices:
                    windowed_indices = [[0, max(1, length - 1)]]

                for index in windowed_indices:
                    windowed_partitioned_trial[partition].append([path, trial, length, index])

        return windowed_partitioned_trial

    def generate_raw_trial_list(self, dataset_path):
        trial_path = os.path.join(dataset_path, self.dataset_info['data_folder'])
        train_list = []

        for trial in self.generate_iterator():
            idx = self.dataset_info['trial'].index(trial)
            trial = self.dataset_info['trial'][idx]
            path = os.path.join(trial_path, trial)
            length = self.dataset_info['length'][idx]
            train_list.append([path, trial, length])

        return train_list

    def generate_iterator(self):
        iterator = []

        for idx, trial in enumerate(self.dataset_info['trial']):
            if self.task == "reg":
                if self.dataset_info['has_continuous_label'][idx]:
                    iterator.append(trial)
            elif self.task == "cls":
                if self.dataset_info['has_eeg'][idx]:
                    iterator.append(trial)
        return iterator

    def partition_range_fn(self):

        if self.task == "reg":
            if self.case == "trs":
                partition_range = [np.arange(a, a + 24) for a in np.arange(0, 239, 24)]

                partition_range[-1] = np.insert(partition_range[-1], obj=[0], values=partition_range[-2][-1])
                partition_range[-2] = np.delete(partition_range[-2], [-1])
                partition_range[-1] = np.delete(partition_range[-1], [-1])

            elif self.case == "loso":
                partition_range = [np.arange(0, 19), np.arange(19, 24), np.arange(24, 37),
                                   np.arange(37, 46), np.arange(46, 59), np.arange(59, 68),
                                   np.arange(68, 81), np.arange(81, 94), np.arange(94, 99),
                                   np.arange(99, 105), np.arange(105, 118), np.arange(118, 120),
                                   np.arange(120, 121), np.arange(121, 132), np.arange(132, 149),
                                   np.arange(149, 159), np.arange(159, 173), np.arange(173, 188),
                                   np.arange(188, 200), np.arange(200, 216), np.arange(216, 222),
                                   np.arange(222, 226), np.arange(226, 236), np.arange(236, 239)]

        elif self.task == "cls":
            if self.case == "trs":
                partition_range = [np.arange(a, a + 53) for a in np.arange(0, 527, 53)]
                for i in range(3):
                    partition_range[-1] = np.insert(partition_range[-1], obj=[0], values=partition_range[-2][-1])
                    partition_range[-2] = np.delete(partition_range[-2], [-1])
            elif self.case == "loso":
                partition_range = [np.arange(0, 20), np.arange(20, 40), np.arange(40, 57),
                                   np.arange(57, 77), np.arange(77, 97), np.arange(97, 117),
                                   np.arange(117, 137), np.arange(137, 157), np.arange(157, 171),
                                   np.arange(171, 191), np.arange(191, 211), np.arange(211, 231),
                                   np.arange(231, 251), np.arange(251, 267), np.arange(267, 287),
                                   np.arange(287, 307), np.arange(307, 327), np.arange(327, 347),
                                   np.arange(347, 367), np.arange(367, 387), np.arange(387, 407),
                                   np.arange(407, 427), np.arange(427, 447), np.arange(447, 467),
                                   np.arange(467, 487), np.arange(487, 507), np.arange(507, 527)]
        else:
            raise ValueError("Unknown task!")

        if self.debug == 1:
            if self.case == "loso":
                num_folds = 24
            elif self.case == "trs":
                num_folds = 10
            else:
                raise ValueError("Unknown case!")
            partition_range = [np.arange(a, a + 1) for a in range(num_folds)]

        return partition_range

    def assign_fold_to_partition(self):
        if self.task == "reg":
            if self.case == "trs":
                fold_to_partition = {'train': 9, 'validate': 0, 'test': 1}
            elif self.case == "loso":
                fold_to_partition = {'train': 23, 'validate': 0, 'test': 1}
            else:
                raise ValueError("Unknown case!!")
        elif self.task == "cls":
            if self.case == "trs":
                fold_to_partition = {'train': 9, 'validate': 0, 'test': 1}
            elif self.case == "loso":
                fold_to_partition = {'train': 26, 'validate': 0, 'test': 1}
            else:
                raise ValueError("Unknown case!!")
        else:
            raise ValueError("Unknown task!")

        return fold_to_partition

    @staticmethod
    def get_feature_list():
        # 🌟 护法修改点 3：这里也换成 eeg_DE
        feature_list = ['eeg_DE', 'eeg_raw']
        return feature_list

    @staticmethod
    def windowing(x, window_length, hop_length):
        length = len(x)

        if length >= window_length:
            steps = (length - window_length) // hop_length + 1

            sampled_x = []
            for i in range(steps):
                start = i * hop_length
                end = start + window_length
                sampled_x.append(x[start:end])

            if sampled_x[-1][-1] < length - 1:
                sampled_x.append(x[-window_length:])
        else:
            sampled_x = [x]

        return sampled_x