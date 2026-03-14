import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pipeline import processAllFiles, solve_qubo_seizure
from FeatureExtraction import extract_band_power
from parser import parse_seizure_file
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from pathlib import Path 
import os
import datetime
import cupy as cp

# --- 設定 ---
SUMMARY_PATH = "DESTINATION/chb01/chb01-summary.txt"
DATA_DIR = "DESTINATION/chb01/"

# 獲取所有 EDF 檔案
seizure_files = [file.name for file in Path(DATA_DIR).iterdir() if file.suffix == '.edf']

seizures = parse_seizure_file(SUMMARY_PATH)

print("正在預處理所有檔案...")
allDataFeatures , allDataLabels = processAllFiles([os.path.join(DATA_DIR, f) for f in seizure_files], seizures)  # 這裡會使用多核心加速預處理


results = []

for testFile in seizure_files:
    print(f"\n>>> 正在測試檔案: {testFile} (使用 GPU 加速) <<<")
    trainFeaturesList = [cp.asarray(allDataFeatures[f]) for f in seizure_files if f != testFile]
    trainLabelsList = [cp.asarray(allDataLabels[f]) for f in seizure_files if f != testFile]
    # 使用 cupy 或 numpy 快速合併
    # 如果 preprocess_one_file 已經回傳 GPU 陣列，這裡會非常快
    xTrainAll = cp.vstack(trainFeaturesList)
    yTrainAll = cp.concatenate(trainLabelsList)
    print(f"訓練資料處理完成。特徵維度: {xTrainAll.shape}")

    # 2. 訓練模型 (使用 GPU)
    scaler = StandardScaler() # 使用 GPU 加速的標標準化
    xTrainScaled = scaler.fit_transform(xTrainAll)
    
    # cuml.SVC 的參數與 sklearn 幾乎一致
    # verbose=False 可以減少雜訊，想看進度可設 True
    clf = SVC(probability=True, kernel='rbf', class_weight='balanced', verbose=False)
    
    print(f"正在 GPU 上訓練 SVM (樣本數: {xTrainAll.shape[0]})...")
    clf.fit(xTrainScaled, yTrainAll)
    print("GPU 訓練完成！")

    # 3. 準備測試集
    xTestFeat = cp.asarray(allDataFeatures[testFile])
    yTest = allDataLabels[testFile]

    xTestScaled = scaler.transform(xTestFeat)
    # 4. 取得分數與 QUBO 優化
    # 注意：predict_proba 在 GPU 上極快
    scores = clf.predict_proba(xTestScaled)[:, 1]

    # 轉回 numpy 格式供 QUBO (CPU) 使用
    if hasattr(scores, 'get'): # 如果是 cupy 陣列則轉回 numpy
        scores = scores.get()
    
    y_baseline = (scores > 0.5).astype(int)
    y_qubo = solve_qubo_seizure(scores, lmbda=1.5, threshold=0.45)
    
    # 5. 紀錄結果
    b_f1 = f1_score(yTest, y_baseline)
    q_f1 = f1_score(yTest, y_qubo)
    results.append({
        'File': testFile,
        'Baseline_F1': b_f1,
        'QUBO_F1': q_f1,
        'Improvement': q_f1 - b_f1
    })
    
    print(f"檔案 {testFile} 完成！提升幅度: {q_f1 - b_f1:.4f}")

# --- 6. 輸出總結表 ---
df_results = pd.DataFrame(results)
output_dir = "experimentLogs"
os.makedirs(output_dir, exist_ok=True)
df_results.to_pickle(f"{output_dir}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu_results.pkl")  # 儲存結果以供後續分析
print("\n" + "="*40)
print("GPU 實驗總結報告")
print(df_results)
print(f"平均提升幅度: {df_results['Improvement'].mean():.4f}")