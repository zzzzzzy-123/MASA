import os
import shutil
from pathlib import Path
import pickle

import numpy as np
import math
import matplotlib.pyplot as plt  # 🌟 新增：绘图专用引擎


def load_npy(path, feature):
    filename = os.path.join(path, feature + ".npy")
    data = np.load(filename, mmap_mode='c')
    return data


def load_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def save_to_pickle(path, data, replace=False):
    if replace:
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
    else:
        if not os.path.isfile(path):
            with open(path, 'wb') as handle:
                pickle.dump(data, handle)


def copy_file(input_filename, output_filename):
    if not os.path.isfile(output_filename):
        shutil.copy(input_filename, output_filename)


def expand_index_by_multiplier(index, multiplier):
    expanded_index = []
    for value in index:
        expanded_value = [i for i in np.arange(value * multiplier, (value + 1) * multiplier)]
        expanded_index.extend(expanded_value)

    return expanded_index


def get_filename_from_a_folder_given_extension(folder, extension, string=""):
    file_list = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(extension):
            if string in file:
                file_list.append(os.path.join(folder, file))

    return file_list


def ensure_dir(file_path):
    directory = file_path
    if file_path[-3] == "." or file_path[-4] == ".":
        directory = os.path.dirname(file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def roll_list(arrays, shift):
    """Roll a list of numpy arrays by a specified shift amount.

    Parameters:
    arrays (list of np.ndarray): The list of numpy arrays to be rolled.
    shift (int): The number of positions by which to shift the list.

    Returns:
    list of np.ndarray: The rolled list of numpy arrays.
    """
    # Calculate the effective shift
    shift %= len(arrays)
    # Roll the list
    return arrays[shift:] + arrays[:shift]


# ==============================================================================
# 🌟 以下为新增：终极炼丹监控面板 (自动绘制 Loss, LR, RMSE/CCC 曲线)
# ==============================================================================
def plot_training_curves(record_dict, save_dir, fold_idx):
    """
    终极炼丹监控面板：自动绘制 Loss, LR, RMSE/CCC 曲线并保存
    """
    ensure_dir(save_dir)  # 确保文件夹存在 (复用了你上面的 ensure_dir 函数)

    # 提取共有多少个 Epoch
    epochs = None
    for key, values in record_dict.items():
        if isinstance(values, list) and len(values) > 0:
            epochs = range(1, len(values) + 1)
            break

    if epochs is None:
        print("⚠️ 警告：训练记录为空，无法画图！")
        return

    # 创建一个 3 行 1 列的大图面板
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Training Metrics - Fold {fold_idx}', fontsize=16, fontweight='bold')

    # ---- 1. 绘制 Loss 曲线 ----
    if 'train_loss' in record_dict:
        ax1.plot(epochs, record_dict['train_loss'], label='Train Loss', color='tab:red', linewidth=2)
    if 'val_loss' in record_dict:
        ax1.plot(epochs, record_dict['val_loss'], label='Val Loss', color='tab:orange', linewidth=2, linestyle='--')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ---- 2. 绘制 RMSE / CCC 曲线 ----
    # 智能寻找字典里有没有 rmse 或者 ccc 关键字
    metric_keys = [k for k in record_dict.keys() if 'rmse' in k.lower() or 'ccc' in k.lower() or 'pcc' in k.lower()]
    for k in metric_keys:
        # 给不同的指标分配不同的样式
        linestyle = '--' if 'val' in k.lower() else '-'
        ax2.plot(epochs, record_dict[k], label=k.upper(), linewidth=2, linestyle=linestyle)
    ax2.set_ylabel('Metrics (RMSE / CCC / PCC)', fontweight='bold')
    if metric_keys:
        ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # ---- 3. 绘制 Learning Rate (LR) 曲线 ----
    lr_key = [k for k in record_dict.keys() if 'lr' in k.lower()]
    if lr_key:
        ax3.plot(epochs, record_dict[lr_key[0]], label='Learning Rate', color='tab:green', linewidth=2)
    ax3.set_xlabel('Epochs', fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontweight='bold')
    # 因为 LR 可能出现从 1e-6 飙升到 1e-3 的情况，用对数坐标系显示最直观
    ax3.set_yscale('log')
    if lr_key:
        ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = os.path.join(save_dir, f'training_curves_fold_{fold_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi=300 保证论文级高清
    plt.close()
    print(f"📸 咔嚓！第 {fold_idx} 折的训练曲线已保存至: {save_path}")