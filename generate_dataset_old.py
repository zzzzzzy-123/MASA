from base.preprocessing import GenericDataPreprocessing
from base.utils import expand_index_by_multiplier, load_pickle, save_to_pickle, get_filename_from_a_folder_given_extension, ensure_dir
from base.label_config import *

import os
import scipy.io as sio

import pandas as pd
import numpy as np

import xml.etree.ElementTree as et


class Preprocessing(GenericDataPreprocessing):
    def __init__(self, config):
        super().__init__(config)

        # ==== 🚀 终极补给：在启动时强行把 Excel 表格加载进来！ ====
        import pandas as pd
        import os
        label_path = os.path.join(self.config['root_directory'], self.config['label_file'])
        if os.path.exists(label_path):
            self.label_df = pd.read_excel(label_path)
            print(f"✅ 成功加载标签表：{self.config['label_file']}，共 {len(self.label_df)} 条病人数据！")
        else:
            print(f"❌ 警告：找不到标签表 {label_path}")

    def generate_iterator(self):
        import re
        path = os.path.join(self.config['root_directory'], self.config['raw_data_folder'])

        # 智能提取文件名中的数字用来排序 (比如 "SIEEG (1)_main.csv" 提取出 1)
        def extract_num(filename):
            nums = re.findall(r'\d+', filename)
            return int(nums[0]) if nums else 0

        # 过滤掉隐藏文件，并按提取到的数字智能排序
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        iterator = [os.path.join(path, f) for f in sorted(files, key=extract_num)]
        return iterator

    def generate_per_trial_info_dict(self):
        import re
        per_trial_info_path = os.path.join(self.config['output_root_directory'], "processing_records.pkl")

        if os.path.isfile(per_trial_info_path):
            per_trial_info = load_pickle(per_trial_info_path)
        else:
            per_trial_info = {}
            iterator = self.generate_iterator()

            # ==== 🚀 全新构建：专为 Risk_EEG 打造的极简处理逻辑 ====
            for idx, file in enumerate(iterator):
                this_trial = {}
                print(f"正在构建档案: {file}")

                # 从文件名提取病人编号 (比如 "SIEEG (1)_main.csv" 提取出 1)
                filename = os.path.basename(file)
                nums = re.findall(r'\d+', filename)
                session = int(nums[0]) if nums else idx + 1

                # 填充通用信息 (直接绕过原本找 TSV/XML 的废话)
                this_trial['discard'] = 0
                this_trial['has_continuous_label'] = 0
                this_trial['continuous_label'] = None
                this_trial['annotated_index'] = None
                this_trial['video_trim_range'] = [(0, 999999)]  # 假装有个很长的时间段，防止截断报错

                this_trial['has_eeg'] = 1
                this_trial['eeg_path'] = [file]  # 直接把当前文件认作脑电文件

                this_trial['audio_path'] = ""
                this_trial['subject_no'] = session
                this_trial['trial_no'] = 1
                this_trial['trial'] = f"P{session}-T1"

                this_trial['target_fps'] = 64
                this_trial['video_annotated_index'] = []
                this_trial['class_label'] = ""  # 我们不用 XML，标签直接从 Excel 读，这里留空

                per_trial_info[idx] = this_trial

        ensure_dir(per_trial_info_path)
        save_to_pickle(per_trial_info_path, per_trial_info)
        self.per_trial_info = per_trial_info

    def generate_dataset_info(self):
        class_label = {}
        for idx, record in self.per_trial_info.items():
            # 🚨 检查是否成功处理了脑电，如果没处理成功就跳过，不录入 dataset_info！
            if not record.get('has_eeg', False):
                continue

            self.dataset_info['trial'].append(record.get('trial', f'P{record.get("subject_no")}-T1'))
            self.dataset_info['trial_no'].append(record.get('trial_no', 1))
            self.dataset_info['subject_no'].append(record.get('subject_no', 0))
            self.dataset_info['has_continuous_label'].append(record.get('has_continuous_label', 0))
            self.dataset_info['has_eeg'].append(record.get('has_eeg', 1))

            if record.get('has_continuous_label', 0):
                self.dataset_info['length'].append(len(record.get('continuous_label', [])))
            else:
                self.dataset_info['length'].append(len(record.get('video_annotated_index', [])) // 16)

        self.dataset_info['multiplier'] = self.config['multiplier']
        self.dataset_info['data_folder'] = self.config['npy_folder']

        path = os.path.join(self.config['output_root_directory'], 'dataset_info.pkl')
        save_to_pickle(path, self.dataset_info)

    def extract_class_label_fn(self, record):
        # ==== ✂️ 最后一刀：彻底切除 Mahnob 的 XML 标签解析！ ====
        # 咱们的 SI 分数会在训练时直接从 Excel 物理级穿透读取，不需要这些废话！
        return {}

    def extract_continuous_label_fn(self, idx, npy_folder):

        if self.per_trial_info[idx]["has_continuous_label"]:
            raw_continuous_label = self.per_trial_info[idx]['continuous_label']

            if self.config['save_npy']:
                filename = os.path.join(npy_folder, "continuous_label.npy")
                if not os.path.isfile(filename):
                    ensure_dir(filename)
                    np.save(filename, raw_continuous_label)

    def load_continuous_label(self, path, **kwargs):

        cols = [emotion.lower() for emotion in self.config['emotion_list']]

        if os.path.isfile(path):
            continuous_label = pd.read_csv(path, sep=";",
                                           skipinitialspace=True, usecols=cols,
                                           index_col=False).values.squeeze()
        else:
            continuous_label = 0

        return continuous_label

    def get_annotated_index(self, annotated_index, **kwargs):

        feature = kwargs['feature']
        multiplier = self.config['multiplier'][feature]

        if kwargs['has_continuous_label']:
            annotated_index = expand_index_by_multiplier(annotated_index, multiplier)

        # If the trial is not continuously labeled, then the whole facial video is used.
        else:
            pass

        return annotated_index

    def get_sub_trial_info_for_continuously_labeled(self):
        return[]
        # label_file = os.path.join(self.config['root_directory'], "lable_continous_Mahnob.mat")
        # mat_content = sio.loadmat(label_file)
        # sub_trial_having_continuous_label = mat_content['trials_included']

        # return sub_trial_having_continuous_label

    @staticmethod
    def read_start_end_from_mahnob_tsv(tsv_file):
        if os.path.isfile(tsv_file):
            data = pd.read_csv(tsv_file, sep='\t', skiprows=23)
            end = data[data['Event'] == 'MovieEnd'].index[0]
            start_end = [(0, end)]
        else:
            start_end = None
        return start_end

    def read_all_continuous_label(self):
        r"""
        :return: the continuous labels for each trial (dict).
        """

        label_file = os.path.join(self.config['root_directory'], "lable_continous_Mahnob.mat")
        mat_content = sio.loadmat(label_file)
        annotation_cell = np.squeeze(mat_content['labels'])

        label_list = []
        for index in range(len(annotation_cell)):
            label_list.append(annotation_cell[index].T)
        return label_list

    @staticmethod
    def init_dataset_info():
        dataset_info = {
            "trial": [],
            "subject_no": [],
            "trial_no": [],
            "length": [],
            "has_continuous_label": [],
            "has_eeg": [],
        }
        return dataset_info


if __name__ == "__main__":
    from configs import config

    pre = Preprocessing(config)
    pre.generate_per_trial_info_dict()
    pre.prepare_data()
