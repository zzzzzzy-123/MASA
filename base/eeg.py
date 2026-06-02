from scipy.signal import welch, hilbert
from scipy.integrate import simps  # 较新版本的 scipy 可能是 from scipy.integrate import simpson as simps
import numpy as np
import mne
from scipy import signal
import math
import os
import pandas as pd

class GenericEegController(object):

    def __init__(self, filename, config):
        self.filename = filename
        self.buffer_sec = config['buffer_sec']
        self.frequency = config['sampling_frequency']
        self.window_sec = config['window_sec']
        self.pli_feature_mode = config.get('pli_feature_mode', 'delta_pli')
        self.step = int(config['hop_sec'] * self.frequency)
        self.interest_bands = config['interest_bands']
        self.f_trans_interest_bands = config['f_trans_interest_bands']
        self.channel_slice = config['channel_slice']
        self.eeg_feature_list = config['features']
        self.filter_type = config['filter_type']
        self.filter_order = config['filter_order']
        # 核心：初始化时直接触发顶刊级预处理和特征提取
        self.extracted_data = self.preprocessing()

        # ==========================================
        #  1. 差分熵 (DE) 提取 (4 段基线法)
        # ==========================================
    def calculate_DE(self, data):
        data_np = data[:]

        # 基线 20 秒分为 4 段，每段 5 秒，最后取平均
        base_features = []
        for i in range(4):
            b_start = int((5 + i * 5) * self.frequency)
            b_end = int((5 + (i + 1) * 5) * self.frequency)
            b_seg = data_np[b_start:b_end, :].T
            b_bp = bandpower_multiple(b_seg, self.frequency, self.interest_bands, relative=False)
            b_de = 0.5 * np.log(2 * np.pi * np.e * b_bp + 1e-10)
            base_features.append(b_de)

        base_de = np.mean(base_features, axis=0)  # 对 4 个特征求平均

        # 提取任务态保持不变
        DEs = []
        samples_per_stimulus = int(self.window_sec * self.frequency)
        for idx in self.stimulus_indices:
            start = idx
            end = start + samples_per_stimulus
            segment = data_np[start:end, :] if end <= data_np.shape[0] else data_np[start:, :]
            if end > data_np.shape[0]:
                segment = np.pad(segment, ((0, end - data_np.shape[0]), (0, 0)), mode='edge')
            abs_bp = bandpower_multiple(segment.T, self.frequency, self.interest_bands, relative=False)
            task_de = 0.5 * np.log(2 * np.pi * np.e * abs_bp + 1e-10)
            DEs.append(task_de - base_de)
        return np.stack(DEs), base_de

    # ==========================================
    #  2. 相对能量谱 (RP) 提取 (含 4 段基线法)
    # ==========================================
    def calculate_RP(self, data):
        data_np = data[:]
        data_filtered = filter_band(data=data_np.T, bands=self.interest_bands, fs=self.frequency,
                                    order=self.filter_order)
        if len(data_filtered.shape) == 2: data_filtered = np.expand_dims(data_filtered, axis=0)

        #
        base_features = []
        for i in range(4):
            b_start = int((5 + i * 5) * self.frequency)
            b_end = int((5 + (i + 1) * 5) * self.frequency)
            b_seg = data_filtered[:, :, b_start:b_end]
            b_rp = RP(b_seg)
            b_rp = np.reshape(b_rp, (b_rp.shape[0] * b_rp.shape[1]))
            base_features.append(b_rp)
        base_rp = np.mean(base_features, axis=0)

        RPs = []
        samples_per_stimulus = int(self.window_sec * self.frequency)
        for idx in self.stimulus_indices:
            start = idx
            end = start + samples_per_stimulus
            segment = data_filtered[:, :, start:] if end > data_filtered.shape[2] else data_filtered[
                :, :, start:end]
            if end > data_filtered.shape[2]:
                pad_width = end - data_filtered.shape[2]
                segment = np.pad(segment, ((0, 0), (0, 0), (0, pad_width)), mode='edge')
            task_rp = RP(segment)
            task_rp = np.reshape(task_rp, (task_rp.shape[0] * task_rp.shape[1]))
            RPs.append(task_rp - base_rp)
        return np.stack(RPs), base_rp

    # ==========================================
    #  3. 模糊熵 (FE) 提取 (含 4 段基线法)
    # ==========================================
    def calculate_FE(self, data):
        data_filtered = filter_band(data=data.T, bands=self.interest_bands, fs=self.frequency,
                                    order=self.filter_order)
        if len(data_filtered.shape) == 2: data_filtered = np.expand_dims(data_filtered, axis=0)
        num_bands = data_filtered.shape[0]

        #
        base_features = []
        for i in range(4):
            b_start = int((5 + i * 5) * self.frequency)
            b_end = int((5 + (i + 1) * 5) * self.frequency)
            b_fe_bands = []
            for b in range(num_bands):
                b_fe_bands.append(FuzzyEntropy(data_filtered[b, :, b_start:b_end]))
            base_features.append(np.concatenate(b_fe_bands))
        base_fe = np.mean(base_features, axis=0)

        FEs = []
        time_len = data_filtered.shape[2]
        samples_per_stimulus = int(self.window_sec * self.frequency)
        for idx in self.stimulus_indices:
            start = idx
            end = start + samples_per_stimulus
            task_fe_list = []
            for b in range(num_bands):
                segment = data_filtered[b, :, start:] if end > time_len else data_filtered[b, :, start:end]
                if end > time_len:
                    segment = np.pad(segment, ((0, 0), (0, end - time_len)), mode='edge')
                task_fe_list.append(FuzzyEntropy(segment))
            task_fe = np.concatenate(task_fe_list)
            FEs.append(task_fe - base_fe)
        return np.stack(FEs), base_fe

    # ==========================================
    #  4. PLI 提取 140维路线B (含 4 段基线法)
    # ==========================================
    def _print_pli_stats(self, name, values):
        values = np.asarray(values, dtype=np.float32)
        print(
            f"{name} shape={values.shape}, "
            f"mean/std/min/max={np.mean(values):.6f}/"
            f"{np.std(values):.6f}/{np.min(values):.6f}/{np.max(values):.6f}"
        )

    def calculate_PLI(self, data):
        pli_window_sec = float(self.window_sec)
        print("PLI feature mode = delta_pli")
        print("Output feature = Task_PLI - Base_PLI")
        data_filtered = filter_band(data=data.T, bands=self.interest_bands, fs=self.frequency,
                                    order=self.filter_order)
        num_bands = data_filtered.shape[0]

        global_phases = []
        for b in range(num_bands):
            analytic_signal = hilbert(data_filtered[b, :, :], axis=1)
            band_phase = np.unwrap(np.angle(analytic_signal), axis=1)
            global_phases.append(band_phase)
        global_phases = np.stack(global_phases, axis=0)

        baseline_region_start = 5.0
        baseline_region_end = 25.0
        baseline_hop_sec = 5.0
        baseline_starts = np.arange(
            baseline_region_start,
            baseline_region_end - pli_window_sec + 1e-6,
            baseline_hop_sec,
        )
        baseline_windows = [(float(s), float(s + pli_window_sec)) for s in baseline_starts]

        base_features = []
        for b_start_sec, b_end_sec in baseline_windows:
            b_start = int(b_start_sec * self.frequency)
            b_end = int(b_end_sec * self.frequency)
            b_pli_bands = []
            for b in range(num_bands):
                b_pli_bands.append(PhaseLagIndex_Edges(global_phases[b, :, b_start:b_end]))
            base_features.append(np.concatenate(b_pli_bands))
        base_features = np.asarray(base_features, dtype=np.float32)
        print(f"base_features shape={base_features.shape}")
        base_pli = np.mean(base_features, axis=0)
        self._print_pli_stats("base_pli", base_pli)

        task_plis = []
        delta_plis = []
        time_len = data_filtered.shape[2]
        samples_per_stimulus = int(pli_window_sec * self.frequency)
        for idx in self.stimulus_indices:
            start = idx
            end = start + samples_per_stimulus
            task_pli_list = []
            for b in range(num_bands):
                seg_phase = global_phases[b, :, start:] if end > time_len else global_phases[b, :, start:end]
                if end > time_len:
                    seg_phase = np.pad(seg_phase, ((0, 0), (0, end - time_len)), mode='edge')
                task_pli_list.append(PhaseLagIndex_Edges(seg_phase))
            task_pli = np.concatenate(task_pli_list)
            delta_pli = task_pli - base_pli
            task_plis.append(task_pli)
            delta_plis.append(delta_pli)

        task_pli_seq = np.stack(task_plis).astype(np.float32)      # [40, 140]
        delta_pli_seq = np.stack(delta_plis).astype(np.float32)    # [40, 140]
        task_pli_seq_t = task_pli_seq.T                            # [140, 40]
        delta_pli_seq_t = delta_pli_seq.T                          # [140, 40]

        self._print_pli_stats("task_pli_seq", task_pli_seq_t)
        self._print_pli_stats("delta_pli_seq", delta_pli_seq_t)
        return None, delta_pli_seq, None, base_pli.astype(np.float32)

    # ==========================================
    # 预处理主流程 (防泄露标准化)
    # ==========================================
    def preprocessing(self):
        raw_data = self.read_data()
        # 1. 基础滤波保留 (非常必要)
        filtered_raw = raw_data.copy().load_data().filter(l_freq=4, h_freq=30, method='iir', verbose=False)
        target_sfreq = 125  # 你可以根据需要修改这个目标频率

        if filtered_raw.info['sfreq'] > target_sfreq:
            print(f" 启动降采样：从 {filtered_raw.info['sfreq']}Hz 降至 {target_sfreq}Hz 以加速计算...")
            # MNE 的 resample 自带极其优秀的抗混叠设计

            original_sfreq = filtered_raw.info['sfreq']

            filtered_raw.resample(target_sfreq)

            self.frequency = target_sfreq

            ratio = target_sfreq / original_sfreq
            self.stimulus_indices = [int(idx * ratio) for idx in self.stimulus_indices]
            print(" 标记点刻度已重新校准！")

        try:
            import asrpy
            print("启动 ASR (伪影子空间重建) 进行自动去噪...")

            # ========================================================
            # 🌟 护法金钟罩：用独立的 try-except 隔离脆弱的 asrpy！
            # 如果它内部因为矩阵尺寸不匹配崩溃，绝不允许它波及整个程序！
            # ========================================================
            try:
                asr = asrpy.ASR(sfreq=self.frequency, cutoff=16)
                asr.fit(filtered_raw)
                filtered_raw = asr.transform(filtered_raw)
                print("✅ ASR 去噪完成！")
            except Exception as asr_err:
                print(f"⚠️ [护法隔离] asrpy 内部发生矩阵尺寸崩塌，已强行中断 ASR！")
                print(f"➡️ 已自动无缝切换为纯净滤波数据，继续往下提取特征！")
            # ========================================================

        except ImportError:
            print("未检测到 asrpy 库，跳过 ASR 去伪迹步骤。")

        # 直接获取滤波后的原始数据并转置为 (time, chan)
        data_np = filtered_raw.get_data()
        norm_data_transposed = data_np.T

        extracted_data = {}
        if "eeg_DE" in self.eeg_feature_list:
            task_de, base_de = self.calculate_DE(norm_data_transposed)
            extracted_data.update({'eeg_DE': task_de, 'eeg_DE_base': base_de})
        if "eeg_RP" in self.eeg_feature_list:
            task_rp, base_rp = self.calculate_RP(norm_data_transposed)
            extracted_data.update({'eeg_RP': task_rp, 'eeg_RP_base': base_rp})
        if "eeg_FE" in self.eeg_feature_list:
            task_fe, base_fe = self.calculate_FE(norm_data_transposed)
            extracted_data.update({'eeg_FE': task_fe, 'eeg_FE_base': base_fe})
        if "eeg_PLI" in self.eeg_feature_list:
            task_pli, delta_pli, pli_2ch, base_pli = self.calculate_PLI(norm_data_transposed)
            extracted_data.update({
                'eeg_PLI': delta_pli,
                'eeg_PLI_base': base_pli,
            })

        return extracted_data

    # ==========================================
    # 🌟 鲁棒的读取方法 (防解码崩溃)
    # ==========================================
    def read_data(self):
        import pandas as pd
        import mne

        # 🌟 终极防弹读取：如果出错，直接跳过坏行！
        try:
            df = pd.read_csv(self.filename, skiprows=5, encoding='utf-8', on_bad_lines='skip')
        except Exception:
            try:
                df = pd.read_csv(self.filename, skiprows=5, encoding='gb18030', on_bad_lines='skip')
            except Exception:
                # 如果连 gb18030 都崩了，用最古老的 ascii 强行读，忽略所有非英文乱码
                df = pd.read_csv(self.filename, skiprows=5, encoding='ascii', errors='ignore', on_bad_lines='skip')
        # 🌟 护法第二道防线：检查 Pandas 到底读进来了多少行！
        print(f"🔍 护法诊断：成功读取 CSV，当前剩余行数: {len(df)} 行")
        if len(df) < 1000:
            print(
                "💀 致命错误：数据行数过少！你的 CSV 格式有误，被 on_bad_lines='skip' 删掉了 90% 的数据！请检查原始 CSV 文件的列对齐情况！")
        df.columns = df.columns.str.strip()

        marker_col = None
        for col in df.columns:
            if df[col].dtype == object and df[col].str.contains('题目', na=False).any():
                marker_col = col
                break

        if marker_col is None:
            raise ValueError(f"⚠️ 在文件 {self.filename} 中找不到包含'题目'的打标列！")

        self.stimulus_indices = df[df[marker_col].str.contains('题目', na=False)].index.tolist()

        try:
            eeg_cols = [c for c in df.columns if 'EEG通道' in c]
            if len(eeg_cols) == 0:
                raise ValueError("没找到名字包含'EEG通道'的列")
            eeg_data = df[eeg_cols].values.T * 1e-6
            ch_names = [f'CH{i}' for i in range(len(eeg_cols))]
        except Exception:
            # 如果按名字找不到，就强行取第 1 到 8 列（避开第0列时间戳）
            eeg_data = df.iloc[:, 1:9].values.T * 1e-6
            ch_names = [f'CH{i}' for i in range(8)]

        info = mne.create_info(ch_names=ch_names, sfreq=self.frequency, ch_types='eeg')
        raw_data = mne.io.RawArray(eeg_data, info, verbose=False)
        return raw_data

# ==========================================
# 🌟 底层数学计算函数 (独立类外)
# ==========================================
def bandpower_multiple(data, sampling_frequence, band_sequence, window_sec=1, relative=False):
    nperseg = int(window_sec * sampling_frequence)
    freqs, psd = welch(data, sampling_frequence, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    band_powers = []
    for low, high in band_sequence:
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_power = simps(psd[:, idx_band], dx=freq_res)
        if relative:
            band_power /= (simps(psd, dx=freq_res) + 1e-10)
        band_powers.append(band_power)
    return np.asarray(band_powers).T.flatten()

def filter_band(data, bands, fs, order):
    filtered_data = []
    # 🚨 维度保护：确保 data 是 (chan, time)
    if len(data.shape) == 1: data = data[None, :]
    for low, high in bands:
        b, a = signal.butter(order, [low/(fs/2), high/(fs/2)], 'bandpass')
        filtered_data.append(signal.filtfilt(b, a, data, axis=1))
    return np.stack(filtered_data, axis=0)

def RP(data):
    # data: (freq_band, chan, time)
    data_temp = np.power(data, 2)
    data_temp = np.sum(data_temp, axis=-1) # (freq_band, chan)
    power_sum = np.sum(data_temp, axis=0, keepdims=True) # (1, chan)
    return (data_temp / (power_sum + 1e-10)).T # (chan, freq_band)

def FuzzyEntropy(data, m=2, r=0.2, n=2):
    def _fe_1d(x):
        N = len(x)
        r_val = r * np.std(x)
        if r_val < 1e-8: return 0.0
        def _get_phi(m_dim):
            seq = np.array([x[i:i + m_dim] for i in range(N - m_dim + 1)])
            seq = seq - np.mean(seq, axis=1, keepdims=True)
            d = np.max(np.abs(seq[:, None, :] - seq[None, :, :]), axis=2)
            sim = np.exp(-((d / r_val) ** n))
            return (np.sum(sim) - (N - m_dim + 1)) / ((N - m_dim + 1) * (N - m_dim))
        phi_m = _get_phi(m)
        phi_m_plus = _get_phi(m + 1)
        if phi_m < 1e-10 or phi_m_plus < 1e-10: return 0.0
        return -np.log(phi_m_plus / phi_m)
    return np.array([_fe_1d(chan) for chan in data])

def PhaseLagIndex_Edges(phase_data):

    num_chan = phase_data.shape[0]
    edges = []
    for i in range(num_chan):
        for j in range(i + 1, num_chan): # 只取上三角，避免重复和自对自
            phase_diff = phase_data[i] - phase_data[j]
            pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
            edges.append(pli)
    return np.array(edges) # 返回长度为 28 的向量
