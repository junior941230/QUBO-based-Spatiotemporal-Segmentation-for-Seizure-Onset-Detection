from scipy.signal import welch
import numpy as np


def extract_band_power(epoch, sfreq=256):
    """
    輸入一個 epoch (通道 x 時間點)
    輸出特徵向量 (通道 x 5個頻帶) -> 展平成一維
    """
    # 定義頻帶
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12),
             'Beta': (12, 30), 'Gamma': (30, 40)}

    feat_list = []

    # 對每個通道計算 PSD
    # nperseg 決定頻率解析度，設為 sfreq 代表解析度為 1Hz
    freqs, psd = welch(epoch, fs=sfreq, nperseg=sfreq)

    for band, (low, high) in bands.items():
        # 找出該頻帶對應的頻率索引
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # 計算平均能量
        band_power = psd[:, idx_band].mean(axis=1)
        feat_list.append(band_power)

    # 將結果變成一維向量: [Ch1_Delta, Ch2_Delta..., Ch1_Theta...]
    return np.concatenate(feat_list)
