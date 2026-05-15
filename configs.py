config = {
    "extract_class_label": 1,
    "extract_continuous_label": 0,  # 你是整段脑电对应一个打分，所以连续标签设为0

    "extract_eeg": 1,
    "eeg_folder": "eeg",
    "eeg_config": {
        "sampling_frequency": 250,  # 你数据真实的采样率
        "window_sec": 5,  # 5秒提取一次特征
        "hop_sec": 5,  # 每5秒滑窗一次
        "buffer_sec": 5,
        "num_electrodes": 8,  # 你的真实通道数：8
        "interest_bands": [(4, 8), (8, 10), (10, 13), (13, 20), (20, 30)],
        "channel_slice": {'eeg': slice(0, 8)},
        'features': ['eeg_DE', 'eeg_RP', 'eeg_FE', 'eeg_PLI',
             'eeg_DE_base', 'eeg_RP_base', 'eeg_FE_base', 'eeg_PLI_base'],
        "filter_type": 'butter',
        "filter_order": 4,
        "f_trans_interest_bands": [[4, 8], [8, 10], [10, 13], [13, 20], [20, 30]],
        "sfreq": 250,
    },

    "save_npy": 1,
    "npy_folder": "compacted_EEG",

    "dataset_name": "Risk_EEG",
    # 填入包含 "minor_scale_2_gai.xlsx - Sheet1.csv" 标签表的那个上级文件夹路径
    "root_directory": r"C:\Users\云瑾\Desktop\data",

    # 填入截图里这个装着所有 SIEEG 文件夹的名字
    "raw_data_folder": "SIEEG_no_SI",

    # 标签文件的准确名字（确保它放在 root_directory 下）
    "label_file": "minor_scale_2_gai.xlsx",

    # 预处理后保存的位置
    "output_root_directory": r"C:\Users\云瑾\Desktop\data\Data_NO_SI",

    "emotion_list": ["SI", "SAS", "SDS"],  # 替换掉了 Valence/Arousal

    "multiplier": {
        "eeg_raw": 1,
        "eeg_DE": 1,
        "continuous_label": 1,
    },

    "feature_dimension": {
    "eeg_raw": (2000,),
    "eeg_DE": (40,),  # 8通道 * 5频段
    "eeg_RP": (40,),  # 8通道 * 5频段
    "eeg_FE": (40,),   # 8通道 * 5频段 = 40维
    "eeg_PLI": (140,),  # 28条边 * 5频段 = 140维
    "eeg_DE_base": (40,),
    "eeg_RP_base": (40,),
    "eeg_FE_base": (40,),
    "eeg_PLI_base": (140,),
    "class_label": (1,),
},

    "max_epoch": 50,
    "min_epoch": 0,
    "early_stopping": 20,
    "load_best_at_each_epoch": 1,
    "time_delay": 0,
    "metrics": ["rmse", "pcc", "ccc"],
    "save_plot": 1,
}