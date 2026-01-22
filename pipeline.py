import mne
from sklearn.svm import SVC
import numpy as np
from parser import parse_seizure_file
from neal import SimulatedAnnealingSampler
DURATION = 1.0


def preprocess_one_file(edf_path, seizure_times):
    """
    EDF 檔案前處理流程
    """
    # 1. 讀取資料 (preload=True 才能做濾波)
    # verbose=False 可以讓它安靜一點
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # 3. 濾波
    raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)  # Bandpass
    raw.notch_filter(freqs=60.0, verbose=False)        # Notch

    # 4. 切割 Epochs (例如 1 秒一段)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=DURATION, overlap=0.0, verbose=False)

    # 取得數據矩陣 X: (Epoch數, 通道數, 時間點數)
    X_data = epochs.get_data(copy=True)

    # 5. 製作標籤 y
    # 先產生全 0 的向量
    num_epochs = len(X_data)
    y_data = np.zeros(num_epochs)

    # 根據 Parser 給的時間填入 1
    # raw.first_samp 是檔案開始的採樣點索引，raw.info['sfreq'] 是採樣率(256Hz)
    sfreq = raw.info['sfreq']

    for start_sec, end_sec in seizure_times:
        # 將秒數轉換成 Epoch 的 index
        start_idx = int(start_sec / DURATION)
        end_idx = int(end_sec / DURATION)

        # 邊界檢查，防止超出範圍
        start_idx = max(0, start_idx)
        end_idx = min(num_epochs, end_idx)

        # 標記為發作 (1)
        y_data[start_idx: end_idx] = 1

    return X_data, y_data


def solve_qubo_seizure(all_scores, lmbda=0.5,threshold=0.5):
    """
    輸入: 
        all_scores: SVM 產出的機率向量 (E,)
        lmbda: 平滑係數 (控制連續性的強弱)
    輸出:
        y_star: 優化後的二元序列 (0 或 1)
    """
    
    E = len(all_scores)
    Q = {} # 使用字典格式儲存稀疏矩陣，對 D-Wave 較友善

    # 1. 建構 Unary Terms (對角線)
    # 我們偏移一下機率，讓 < 0.5 變成正能量(偏向0)，> 0.5 變成負能量(偏向1)
    for e in range(E):
        Q[(e, e)] = -(all_scores[e] - threshold)

    # 2. 建構 Pairwise Terms (時間平滑)
    for e in range(E - 1):
        # 罰金項: lambda * (y_e - y_{e+1})^2
        Q[(e, e)] += lmbda
        Q[(e+1, e+1)] += lmbda
        Q[(e, e+1)] = -2 * lmbda

    # 3. 使用模擬退火求解 (模擬量子運算行為)
    sampler = SimulatedAnnealingSampler()
    # num_reads 是抽樣次數，可以先設 10-50 次
    sampleset = sampler.sample_qubo(Q, num_reads=20)
    
    # 取得能量最低的最佳解
    best_sample = sampleset.first.sample
    y_star = np.array([best_sample[i] for i in range(E)])
    
    return y_star

if __name__ == "__main__":
    # 讀取發作時間資料
    FILE_PATH = "DESTINATION/chb01/chb01-summary.txt"
    seizures = parse_seizure_file(FILE_PATH)
    seizure_intervals = seizures['chb01_03.edf']
    X, y = preprocess_one_file(
        'DESTINATION/chb01/chb01_03.edf', seizure_intervals)
    print(f"資料形狀: {X.shape}, 標籤形狀: {y.shape}")
    print(f"發作 Epoch 數量: {np.sum(y)}")
