import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pipeline import processAllFiles, solve_qubo_seizure
from parser import parse_seizure_file
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from pathlib import Path 
import os
import datetime
import cupy as cp

# --- 設定 ---
summaryPath = "DESTINATION/chb01/chb01-summary.txt"
dataDir = "DESTINATION/chb01/"

# 獲取所有 EDF 檔案
seizureFiles = sorted([file.name for file in Path(dataDir).iterdir() if file.suffix == '.edf'])

seizures = parse_seizure_file(summaryPath)

print("正在預處理所有檔案...")
allDataFeatures, allDataLabels = processAllFiles([os.path.join(dataDir, f) for f in seizureFiles], seizures)

results = []

for testFile in seizureFiles:
    print(f"\n>>> 正在測試檔案: {testFile} (使用 GPU 加速) <<<")
    trainFeaturesList = [cp.asarray(allDataFeatures[f]) for f in seizureFiles if f != testFile]
    trainLabelsList = [cp.asarray(allDataLabels[f]) for f in seizureFiles if f != testFile]
    xTrainAll = cp.vstack(trainFeaturesList)
    yTrainAll = cp.concatenate(trainLabelsList)
    print(f"訓練資料處理完成。特徵維度: {xTrainAll.shape}")

    # 2. 訓練模型 (使用 GPU)
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrainAll)
    
    clf = SVC(probability=True, kernel='rbf', class_weight='balanced', verbose=False)
    
    print(f"正在 GPU 上訓練 SVM (樣本數: {xTrainAll.shape[0]})...")
    clf.fit(xTrainScaled, yTrainAll)
    print("GPU 訓練完成！")

    # 3. 準備測試集
    xTestFeat = cp.asarray(allDataFeatures[testFile])
    yTest = np.asarray(allDataLabels[testFile]).astype(int)

    xTestScaled = scaler.transform(xTestFeat)
    # 4. 取得分數與 QUBO 優化
    scores = clf.predict_proba(xTestScaled)[:, 1]

    # 轉回 numpy 格式供 QUBO (CPU) 使用
    if hasattr(scores, 'get'):
        scores = scores.get()
    
    yBaseline = (scores > 0.5).astype(int)
    yQubo = solve_qubo_seizure(scores, lmbda=1.5, threshold=0.45)
    
    # 5. 紀錄結果
    bF1 = f1_score(yTest, yBaseline, zero_division=0)
    qF1 = f1_score(yTest, yQubo, zero_division=0)
    results.append({
        'File': testFile,
        'Num_Test_Samples': len(yTest),
        'Num_Positive': int(np.sum(yTest)),
        'Baseline_Positive_Pred': int(np.sum(yBaseline)),
        'QUBO_Positive_Pred': int(np.sum(yQubo)),
        'Baseline_F1': bF1,
        'QUBO_F1': qF1,
        'Improvement': qF1 - bF1
    })
    print(f"檔案 {testFile} 完成！提升幅度: {qF1 - bF1:.4f}")

# --- 6. 輸出總結表 ---
dfResults = pd.DataFrame(results)
outputDir = "experimentLogs"
os.makedirs(outputDir, exist_ok=True)
dfResults.to_pickle(f"{outputDir}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu_results.pkl")
print("\n" + "="*40)
print("GPU 實驗總結報告")
print(dfResults)
print(f"平均提升幅度: {dfResults['Improvement'].mean():.4f}")