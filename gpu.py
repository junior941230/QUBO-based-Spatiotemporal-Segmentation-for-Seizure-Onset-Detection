import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pipeline import preprocess_one_file, solve_qubo_seizure
from FeatureExtraction import extract_band_power
from parser import parse_seizure_file
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler # cuml 也有 Scaler，速度更快
# ---------------------------------------
from pathlib import Path 

# --- 設定 ---
SUMMARY_PATH = "DESTINATION/chb01/chb01-summary.txt"
DATA_DIR = "DESTINATION/chb01/"

# 獲取所有 EDF 檔案
seizure_files = [file.name for file in Path(DATA_DIR).iterdir() if file.suffix == '.edf']

seizures = parse_seizure_file(SUMMARY_PATH)
results = []

for test_file in seizure_files:
    print(f"\n>>> 正在測試檔案: {test_file} (使用 GPU 加速) <<<")
    
    # 1. 準備訓練集
    X_train_list, y_train_list = [], []
    for train_file in seizure_files:
        if train_file == test_file: continue
        X, y = preprocess_one_file(DATA_DIR + train_file, seizures[train_file])
        X_feat = np.array([extract_band_power(e) for e in X])
        X_train_list.append(X_feat)
        y_train_list.append(y)
    
    X_train_all = np.concatenate(X_train_list).astype('float32') # 轉為 float32
    y_train_all = np.concatenate(y_train_list).astype('float32') # cuml SVM 標籤也建議用 float32

    # 2. 訓練模型 (使用 GPU)
    scaler = StandardScaler() # 使用 GPU 加速的標標準化
    X_train_scaled = scaler.fit_transform(X_train_all)
    
    # cuml.SVC 的參數與 sklearn 幾乎一致
    # verbose=False 可以減少雜訊，想看進度可設 True
    clf = SVC(probability=True, kernel='rbf', class_weight='balanced', verbose=False)
    
    print(f"正在 GPU 上訓練 SVM (樣本數: {X_train_all.shape[0]})...")
    clf.fit(X_train_scaled, y_train_all)
    print("GPU 訓練完成！")

    # 3. 準備測試集
    X_test_raw, y_test = preprocess_one_file(DATA_DIR + test_file, seizures[test_file])
    X_test_feat = np.array([extract_band_power(e) for e in X_test_raw]).astype('float32')
    X_test_scaled = scaler.transform(X_test_feat)

    # 4. 取得分數與 QUBO 優化
    # 注意：predict_proba 在 GPU 上極快
    scores = clf.predict_proba(X_test_scaled)[:, 1]
    
    # 轉回 numpy 格式供 QUBO (CPU) 使用
    if hasattr(scores, 'get'): # 如果是 cupy 陣列則轉回 numpy
        scores = scores.get()
    
    y_baseline = (scores > 0.5).astype(int)
    y_qubo = solve_qubo_seizure(scores, lmbda=1.5, threshold=0.45)
    
    # 5. 紀錄結果
    b_f1 = f1_score(y_test, y_baseline)
    q_f1 = f1_score(y_test, y_qubo)
    results.append({
        'File': test_file,
        'Baseline_F1': b_f1,
        'QUBO_F1': q_f1,
        'Improvement': q_f1 - b_f1
    })
    
    print(f"檔案 {test_file} 完成！提升幅度: {q_f1 - b_f1:.4f}")

# --- 6. 輸出總結表 ---
df_results = pd.DataFrame(results)
print("\n" + "="*40)
print("GPU 實驗總結報告")
print(df_results)
print(f"平均提升幅度: {df_results['Improvement'].mean():.4f}")