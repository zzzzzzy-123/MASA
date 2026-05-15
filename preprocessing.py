import os
import re
import numpy as np
from tqdm import tqdm
from base.eeg import GenericEegController
from base.utils import ensure_dir


class MASA_Express_Preprocessor:
    def __init__(self, config):
        self.config = config

    def run(self):
        root_dir = os.path.join(self.config['root_directory'], self.config['raw_data_folder'])

        # 递归扫描所有以 SIEEG 开头并以 .csv 结尾的文件
        all_files = []
        for r, d, f_list in os.walk(root_dir):
            for f in f_list:
                if f.startswith('SIEEG') and f.endswith('.csv'):
                    all_files.append(os.path.join(r, f))

        print(f"\n🚀 开始纯净特征提取！共发现 {len(all_files)} 个脑电文件！")

        for file_path in tqdm(all_files, desc="特征提纯进度"):
            file_name = os.path.basename(file_path)

            # 从文件名提取数字序号，例如 "SIEEG (5)_main.csv" 提取出 5
            match = re.search(r'\((\d+)\)', file_name)
            subject_no = int(match.group(1)) if match else 0

            if subject_no == 0:
                print(f"⚠️ 警告: 无法从 {file_name} 中提取序号，已跳过。")
                continue

            # 统一命名格式
            output_filename = f"P{subject_no}-T1"
            npy_folder = os.path.join(self.config['output_root_directory'], self.config['npy_folder'], output_filename)
            ensure_dir(npy_folder)

            # 核心：调用底层的顶刊级特征提取算法！
            try:
                eeg_handler = GenericEegController(file_path, config=self.config['eeg_config'])
            except Exception as e:
                print(f"\n⚠️ 文件 {file_name} 处理报错，已跳过: {e}")
                continue

            # 智能保存所有生成的特征为 .npy
            if self.config['save_npy']:
                for feature_name, feature_np in eeg_handler.extracted_data.items():
                    filename = os.path.join(npy_folder, feature_name + ".npy")
                    np.save(filename, feature_np)

        print(
            f"\n🎉 完美！高质量的特征已全部提取至：{os.path.join(self.config['output_root_directory'], self.config['npy_folder'])}")


if __name__ == "__main__":
    from configs import config

    # 只需要这两行，直接起飞！
    preprocessor = MASA_Express_Preprocessor(config)
    preprocessor.run()