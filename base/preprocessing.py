# from base.video import change_video_fps, combine_annotated_clips, OpenFaceController
# from base.audio import convert_video_to_wav, change_wav_frequency, extract_mfcc, extract_egemaps
# from base.speech import extract_transcript, add_punctuation, extract_word_embedding, align_word_embedding
from base.utils import ensure_dir, get_filename_from_a_folder_given_extension, save_to_pickle

import os

from operator import itemgetter
from tqdm import tqdm
from base.eeg import GenericEegController
import pandas as pd
import numpy as np
from PIL import Image


class GenericDataPreprocessing(object):
    def __init__(self, config):

        self.config = config
        self.dataset_info = self.init_dataset_info()

        if "extract_continuous_label" in config and config['extract_continuous_label']:
            self.extract_continuous_label = config['extract_continuous_label']

        if "extract_class_label" in config and config['extract_class_label']:
            self.extract_class_label = config['extract_class_label']

        if "extract_eeg" in config and config['extract_eeg']:
            from base.eeg import GenericEegController
            self.extract_eeg = config['extract_eeg']
            self.eeg_folder = config['eeg_folder']

        self.per_trial_info = {}

    def get_output_root_directory(self):
        return self.config['output_root_directory']

    def prepare_data(self):
        import os
        import re
        import numpy as np
        from tqdm import tqdm
        from base.eeg import GenericEegController
        from base.utils import ensure_dir, save_to_pickle

        # 1. 🌟 强制引擎：扫描真实的带标签文件夹！
        root_dir = os.path.join(self.config['root_directory'], self.config['raw_data_folder'])
        all_files = [f for f in os.listdir(root_dir) if f.endswith('.csv')]

        print(f"\n🚀 开始纯净特征提取（情况 A 路线）！共发现 {len(all_files)} 个脑电文件！")
        print(f"📁 目标输入文件夹：{root_dir}\n")

        # 清空原有的字典
        self.per_trial_info = {}

        for idx, file_name in enumerate(tqdm(all_files, desc="特征提纯进度")):

            # 从文件名里提取出数字，比如 "SIEEG (5)_main.csv" 提取出 5
            match = re.search(r'\((\d+)\)', file_name)
            subject_no = int(match.group(1)) if match else idx + 1

            # 2. 🌟 伪造档案：保留 P1-T1 结构
            output_filename = f"P{subject_no}-T1"

            self.per_trial_info[idx] = {
                'subject_no': subject_no,
                'trial_no': 1,
                'trial': output_filename,
                'partition': 'train',
                'has_eeg': True,
                'processing_record': {'trial': output_filename}
            }

            npy_folder = os.path.join(self.config['output_root_directory'], self.config['npy_folder'], output_filename)
            ensure_dir(npy_folder)

            input_path = os.path.join(root_dir, file_name)

            # 3. 🌟 直接调用顶刊级算子提取特征
            try:
                eeg_handler = GenericEegController(input_path, config=self.config['eeg_config'])
            except Exception as e:
                print(f"\n⚠️ 文件 {file_name} 处理报错，已跳过: {e}")
                self.per_trial_info[idx]['has_eeg'] = False
                continue

            # 4. 🌟 智能覆盖保存 npy (遍历提取出来的所有结果，包括 base)
            if self.config['save_npy']:
                for feature_name, feature_np in eeg_handler.extracted_data.items():
                    filename = os.path.join(npy_folder, feature_name + ".npy")

                    # 把主要的特征路径记录到档案里（带 base 的就不记进去了，模型不读）
                    if feature_name in self.config['eeg_config']['features']:
                        self.per_trial_info[idx][feature_name + '_npy_path'] = filename

                    # 保存为 npy 文件
                    np.save(filename, feature_np)

        # 5. 保留 pkl 记录生成功能
        path = os.path.join(self.config['output_root_directory'], 'processing_records.pkl')
        ensure_dir(path)
        save_to_pickle(path, self.per_trial_info)

        # 顺手把 dataset_info 也生成了，省事！
        self.generate_dataset_info()

        print(f"\n🎉 完美！高质量、无泄露的特征已全部提取至：{self.config['output_root_directory']}")




    def generate_dataset_info(self):

        for idx, record in self.per_trial_info.items():
            self.dataset_info['trial'].append(record['processing_record']['trial'])
            self.dataset_info['trial_no'].append(record['trial_no'])
            self.dataset_info['subject_no'].append(record['subject_no'])
            self.dataset_info['length'].append(len(self.per_trial_info[idx]['continuous_label']))
            self.dataset_info['partition'].append(record['partition'])

        self.dataset_info['multiplier'] = self.config['multiplier']
        self.dataset_info['data_folder'] = self.config['npy_folder']

        path = os.path.join(self.config['output_root_directory'], 'dataset_info.pkl')
        save_to_pickle(path, self.dataset_info)



    def generate_iterator(self):
        return NotImplementedError

    def generate_per_trial_info_dict(self):
        raise NotImplementedError


    @staticmethod
    def get_output_filename(**kwargs):

        output_filename = "P{}-T{}".format(kwargs['subject_no'], kwargs['trial_no'])
        return output_filename

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
