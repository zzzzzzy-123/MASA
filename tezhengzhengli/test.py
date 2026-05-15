import numpy as np

# 随便加载一个你提好特征的 npy 文件（比如 DE 的）
data_de = np.load(r"C:\Users\云瑾\Desktop\data\Data_Processed\compacted_EEG\P1-T1\eeg_DE.npy")
print("DE 特征的形状是:", data_de.shape)

# 如果你有提取好的 FE，也加载看看
# data_fe = np.load(".../eeg_FE.npy")
# print("FE 特征的形状是:", data_fe.shape)