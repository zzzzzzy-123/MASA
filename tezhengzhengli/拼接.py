import pandas as pd
import os
import glob
import re

root_path = r"C:\Users\云瑾\Desktop\data\Data_NO_SI\compacted_EEG1"
features = ['DE', 'FE', 'PLI', 'RP']

# 按特征分类存储数据
data_dict = {feat: [] for feat in features}

all_csv_files = glob.glob(os.path.join(root_path, '**/*.csv'), recursive=True)


# 提取文件夹名称中的 P 数值进行排序，确保最终表格的行顺序严格按照 P1, P2, P3... 排列
def get_p_num(file_path):
    match = re.search(r'P(\d+)-T1', file_path)
    return int(match.group(1)) if match else 99999


all_csv_files.sort(key=get_p_num)

for file_path in all_csv_files:
    file_name = os.path.basename(file_path)

    if 'base' in file_name.lower():
        continue

    for feat in features:
        # 匹配对应特征的文件 (如 eeg_DE.csv)
        if f"{feat}.csv" in file_name:
            df = pd.read_csv(file_path, header=None)
            matrix_data = df.iloc[1:, :]
            flattened = matrix_data.values.flatten()

            data_dict[feat].append(flattened)
            break

# 分别保存为四个独立的 csv 文件
for feat, data in data_dict.items():
    if data:
        df_out = pd.DataFrame(data)
        out_file = os.path.join(root_path, f"{feat}_combined.csv")
        # index=False, header=False 确保纯数据输出，尺寸严格保持为 270x1600 或 270x5600
        df_out.to_csv(out_file, index=False, header=False)