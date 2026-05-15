import os
import pandas as pd
import glob


def process_and_overwrite(parent_folder):
    # 1. 递归搜索所有子文件夹下包含 "base" 的 .csv 文件
    search_pattern = os.path.join(parent_folder, "**", "*base*.csv")
    file_list = glob.glob(search_pattern, recursive=True)

    if not file_list:
        print("⚠️ 未找到任何符合条件的文件，请检查路径。")
        return

    print(f"🚀 找到 {len(file_list)} 个文件，准备执行【覆盖转置】操作...\n")

    for file_path in file_list:
        try:
            # 2. 读取数据 (假设单列)
            df = pd.read_csv(file_path, header=None)

            # 3. 检查数据是否为空或只有一行（防止重复运行导致数据被删光）
            if len(df) <= 1:
                print(f"跳过: {file_path} (数据行数不足，可能已被处理过)")
                continue

            # 4. 删除第一行标签并转置
            df_cleaned = df.drop(0)
            df_transposed = df_cleaned.T

            # 5. 直接保存回原路径 (file_path)，实现覆盖
            df_transposed.to_csv(file_path, index=False, header=False)

            print(f"✅ 已覆盖: {file_path}")

        except Exception as e:
            print(f"❌ 失败: 处理 {file_path} 时出错: {e}")

    print("\n✨ 所有匹配文件已处理完毕。")


# ==========================================
# --- 请将路径修改为你电脑上的实际路径 ---
# ==========================================
target_path = r"C:\Users\云瑾\Desktop\data\Data_NO_SI\compacted_EEG1"

process_and_overwrite(target_path)