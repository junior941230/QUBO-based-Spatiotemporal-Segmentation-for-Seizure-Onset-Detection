import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pipeline import preprocess_one_file, solve_qubo_seizure
from FeatureExtraction import extract_band_power
from parser import parse_seizure_file
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pathlib import Path 

# --- 設定 ---
SUMMARY_PATH = "DESTINATION/chb01/chb01-summary.txt"
DATA_DIR = "DESTINATION/chb01/"
# 列出 chb01 中所有包含發作的檔案 (根據 summary 決定)
seizure_files = []
for file in Path.iterdir(Path(DATA_DIR)):
    if file.suffix == '.edf':
        seizure_files.append(file.name)

seizures = parse_seizure_file(SUMMARY_PATH)
results = []

for test_file in seizure_files:
    print(f"\n>>> 正在測試檔案: {test_file} <<<")
    
    # 1. 準備訓練集：除了當前測試檔，其餘檔案合併
    X_train_list, y_train_list = [], []
    for train_file in seizure_files:
        print(f"準備訓練資料: {train_file}...")
        if train_file == test_file: continue
        X, y = preprocess_one_file(DATA_DIR + train_file, seizures[train_file])
        X_feat = np.array([extract_band_power(e) for e in X])
        X_train_list.append(X_feat)
        y_train_list.append(y)
    print("合併所有訓練資料...")
    X_train_all = np.concatenate(X_train_list)
    y_train_all = np.concatenate(y_train_list)
    print(f"訓練資料形狀: {X_train_all.shape}, 標籤形狀: {y_train_all.shape}")
    # 2. 訓練模型
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    clf = SVC(probability=True, kernel='rbf', class_weight='balanced',verbose= True)
    clf.fit(X_train_scaled, y_train_all)
    print("訓練完成！開始測試...")
    # 3. 準備測試集
    X_test_raw, y_test = preprocess_one_file(DATA_DIR + test_file, seizures[test_file])
    X_test_feat = np.array([extract_band_power(e) for e in X_test_raw])
    X_test_scaled = scaler.transform(X_test_feat)
    print(f"測試資料形狀: {X_test_scaled.shape}, 標籤形狀: {y_test.shape}")
    # 4. 取得分數與 QUBO 優化
    scores = clf.predict_proba(X_test_scaled)[:, 1]
    y_baseline = (scores > 0.5).astype(int)
    y_qubo = solve_qubo_seizure(scores, lmbda=1.5, threshold=0.45)
    print("優化完成，計算評估指標...")
    # 5. 紀錄結果
    b_f1 = f1_score(y_test, y_baseline)
    q_f1 = f1_score(y_test, y_qubo)
    results.append({
        'File': test_file,
        'Baseline_F1': b_f1,
        'QUBO_F1': q_f1,
        'Improvement': q_f1 - b_f1
    })
    print(f"完成！提升幅度: {q_f1 - b_f1:.4f}")

# --- 6. 輸出總結表 ---
df_results = pd.DataFrame(results)
print("\n" + "="*40)
print("最終實驗總結報告")
print(df_results)
print(f"平均提升幅度: {df_results['Improvement'].mean():.4f}")