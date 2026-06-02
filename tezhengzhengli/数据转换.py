import numpy as np
import pandas as pd
import os


def batch_convert_and_replace(root_path):
    # 遍历根目录下所有的子文件夹和文件
    for subdir, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.npy'):
                # 构建完整的 .npy 文件路径
                npy_path = os.path.join(subdir, file)
                # 构建目标 .csv 文件路径
                csv_path = os.path.splitext(npy_path)[0] + '.csv'

                try:
                    # 1. 加载并转换数据
                    data = np.load(npy_path)

                    # 脑电数据处理：若超过2维则展平
                    if data.ndim > 2:
                        data = data.reshape(data.shape[0], -1)

                    # 2. 保存为 CSV
                    df = pd.DataFrame(data)
                    df.to_csv(csv_path, index=False)

                    # 3. 核心步骤：转换成功后删除原文件
                    os.remove(npy_path)

                    print(f"已覆盖替换: {file} -> {os.path.basename(csv_path)}")

                except Exception as e:
                    print(f"处理失败 {npy_path}: {e}")


# 设置你的文件夹路径
target_directory = r'C:\Users\云瑾\Desktop\data\Data_Processed\compacted_EEG'

# 执行批量处理
batch_convert_and_replace(target_directory)
print("\n--- 批量替换完成，.npy 文件已清理 ---")