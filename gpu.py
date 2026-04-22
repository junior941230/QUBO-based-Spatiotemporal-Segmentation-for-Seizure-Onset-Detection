import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from pipeline import processAllFiles, solve_qubo_seizure
from parser import parse_seizure_file
from xgboost import XGBClassifier
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from pathlib import Path
import os
import datetime
import cupy as cp
import time


def build_validation_score_cache(trainingFiles, allDataFeatures, allDataLabels):
    cache = {}

    for valFile in trainingFiles:
        innerTrainFiles = [f for f in trainingFiles if f != valFile]

        if len(innerTrainFiles) == 0:
            continue

        print(f"  [Cache] 正在建立 validation score: {valFile}")

        scores, yVal = train_and_get_scores(
            innerTrainFiles,
            valFile,
            allDataFeatures,
            allDataLabels
        )

        cache[valFile] = {
            "scores": np.asarray(scores),
            "yVal": np.asarray(yVal).astype(int)
        }

    return cache


def build_validation_score_cache_kfold(trainingFiles, allDataFeatures, allDataLabels, n_splits=5):
    """
    用 K-Fold 取代 Leave-One-File-Out，大幅減少訓練次數。
    原本 N-1 次訓練 → 現在只需 K 次訓練。
    """
    cache = {}
    trainArray = np.array(trainingFiles)

    fileHasSeizure = np.array([
        1 if np.sum(allDataLabels[f]) > 0 else 0
        for f in trainingFiles
    ])

    minClassCount = min(np.sum(fileHasSeizure), np.sum(1 - fileHasSeizure))
    if minClassCount >= n_splits:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splitIter = kf.split(trainArray, fileHasSeizure)
    else:
        kf = KFold(n_splits=min(n_splits, len(trainingFiles)),
                   shuffle=True, random_state=42)
        splitIter = kf.split(trainArray)

    for foldIdx, (trainIdx, valIdx) in enumerate(splitIter):
        innerTrainFiles = trainArray[trainIdx].tolist()
        valFiles = trainArray[valIdx].tolist()

        print(f"  [Fold {foldIdx + 1}/{kf.n_splits}] "
              f"train={len(innerTrainFiles)} files, val={len(valFiles)} files")

        xTrain = cp.vstack([cp.asarray(allDataFeatures[f])
                           for f in innerTrainFiles])
        yTrain = cp.concatenate(
            [cp.asarray(allDataLabels[f]).astype(cp.int32) for f in innerTrainFiles])

        scaler = StandardScaler()
        xTrainScaled = scaler.fit_transform(xTrain)

        bst = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            objective='binary:logistic', device='cuda'
        )
        # ✅ 訓練時用 CuPy array，XGBoost 會自動識別為 GPU 資料
        bst.fit(xTrainScaled, yTrain)

        for valFile in valFiles:
            xVal = cp.asarray(allDataFeatures[valFile])
            yVal = np.asarray(allDataLabels[valFile]).astype(int)

            xValScaled = scaler.transform(xVal)
            # ✅ 預測時也直接傳 CuPy array，不要 .get()
            scores = bst.predict_proba(xValScaled)[:, 1]

            if hasattr(scores, 'get'):
                scores = scores.get()
            else:
                scores = np.asarray(scores)

            cache[valFile] = {
                "scores": scores,
                "yVal": yVal
            }

    print(f"  [Cache] 完成！共 {len(cache)} 個檔案的 validation scores")
    return cache


def train_and_get_scores(trainFiles, valFile, allDataFeatures, allDataLabels):
    # 組訓練集
    xTrain = cp.vstack([cp.asarray(allDataFeatures[f]) for f in trainFiles])
    yTrain = cp.concatenate(
        [cp.asarray(allDataLabels[f]).astype(cp.int32) for f in trainFiles])

    # 組驗證集
    xVal = cp.asarray(allDataFeatures[valFile])
    yVal = np.asarray(allDataLabels[valFile]).astype(int)

    # 標準化
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xValScaled = scaler.transform(xVal)

    # 訓練 SVM
    # clf = SVC(probability=True, kernel='rbf', class_weight='balanced', verbose=False)
    # clf.fit(xTrainScaled, yTrain)
    # scores = clf.predict_proba(xValScaled)[:, 1]

    # XGBoost
    bst = XGBClassifier(n_estimators=500, max_depth=6,
                        learning_rate=0.1, objective='binary:logistic', device='cuda')
    bst.fit(xTrainScaled.get(), yTrain.get())
    scores = bst.predict_proba(xValScaled)[:, 1]

    # 預測 validation 機率

    if hasattr(scores, 'get'):
        scores = scores.get()
    else:
        scores = np.asarray(scores)

    return scores, yVal


def tune_qubo_params_from_cache(scoreCache, lambdaList, thresholdList, alpha=0.2):
    bestScore = -1e9
    bestLambda = None
    bestThreshold = None

    for lmbda in lambdaList:
        for threshold in thresholdList:
            seizureF1List = []
            nonseizureFpList = []

            for valFile, data in scoreCache.items():
                scores = data["scores"]
                yVal = data["yVal"]

                yQuboVal = solve_qubo_seizure(
                    scores, lmbda=lmbda, threshold=threshold)
                yQuboVal = np.asarray(yQuboVal).astype(int)

                if np.sum(yVal) > 0:
                    seizureF1 = f1_score(yVal, yQuboVal, zero_division=0)
                    seizureF1List.append(seizureF1)
                else:
                    fpRate = np.mean(yQuboVal)
                    nonseizureFpList.append(fpRate)

            meanSeizureF1 = np.mean(seizureF1List) if len(
                seizureF1List) > 0 else 0.0
            meanNonseizureFp = np.mean(nonseizureFpList) if len(
                nonseizureFpList) > 0 else 0.0

            combinedScore = meanSeizureF1 - alpha * meanNonseizureFp

            print(
                f"[QUBO Tune] lambda={lmbda}, threshold={threshold}, "
                f"seizureF1={meanSeizureF1:.4f}, nonseizureFP={meanNonseizureFp:.4f}, "
                f"combined={combinedScore:.4f}"
            )

            if combinedScore > bestScore:
                bestScore = combinedScore
                bestLambda = lmbda
                bestThreshold = threshold

    return bestLambda, bestThreshold, bestScore


# --- 設定 ---
summaryPath = "DESTINATION/chb01/chb01-summary.txt"
dataDir = "DESTINATION/chb01/"

# 獲取所有 EDF 檔案
seizureFiles = sorted([file.name for file in Path(
    dataDir).iterdir() if file.suffix == '.edf'])

seizures = parse_seizure_file(summaryPath)

print("正在預處理所有檔案...")
preprocessStart = time.perf_counter()
allDataFeatures, allDataLabels = processAllFiles(
    [os.path.join(dataDir, f) for f in seizureFiles], seizures)
preprocessElapsed = time.perf_counter() - preprocessStart
print(f"前處理完成，耗時: {preprocessElapsed:.2f} 秒")

lambdaList = [0.5, 1.0, 1.5, 2.0, 3.0]
thresholdList = [0.3, 0.4, 0.45, 0.5, 0.6]

results = []
globalStart = time.perf_counter()

for testFile in seizureFiles:
    fileStart = time.perf_counter()
    print(f"\n>>> 正在測試檔案: {testFile} (使用 GPU 加速) <<<")
    trainingFiles = [f for f in seizureFiles if f != testFile]

    # Step 1: 建立 inner validation score cache
    print("正在建立 validation score cache...")
    cacheStart = time.perf_counter()
    # scoreCache = build_validation_score_cache_kfold(
    #     trainingFiles,
    #     allDataFeatures,
    #     allDataLabels,
    #     n_splits=5
    # )

    scoreCache = build_validation_score_cache(
        trainingFiles,
        allDataFeatures,
        allDataLabels
    )
    cacheElapsed = time.perf_counter() - cacheStart
    print(f"validation score cache 建立完成，耗時: {cacheElapsed:.2f} 秒")
    # --- 先在 training files 上做 inner validation 調參 ---
    # 從 cache 做 QUBO 參數搜尋
    print("正在調整 QUBO 參數...")
    tuneStart = time.perf_counter()
    bestLambda, bestThreshold, bestValScore = tune_qubo_params_from_cache(
        scoreCache,
        lambdaList,
        thresholdList,
        alpha=0.2
    )
    tuneElapsed = time.perf_counter() - tuneStart
    print(
        f"最佳 QUBO 參數: lambda={bestLambda}, threshold={bestThreshold}, "
        f"validation score={bestValScore:.4f}"
    )
    print(f"QUBO 參數調整完成，耗時: {tuneElapsed:.2f} 秒")

    # --- 用全部 training files 重訓最終模型 ---
    trainPrepStart = time.perf_counter()
    trainFeaturesList = [cp.asarray(allDataFeatures[f])
                         for f in seizureFiles if f != testFile]
    trainLabelsList = [cp.asarray(allDataLabels[f]).astype(
        cp.int32) for f in seizureFiles if f != testFile]
    xTrainAll = cp.vstack(trainFeaturesList)
    yTrainAll = cp.concatenate(trainLabelsList)
    trainPrepElapsed = time.perf_counter() - trainPrepStart
    print(f"訓練資料處理完成。特徵維度: {xTrainAll.shape}")
    print(f"訓練資料準備耗時: {trainPrepElapsed:.2f} 秒")

    # 2. 訓練模型 (使用 GPU)
    modelTrainStart = time.perf_counter()
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrainAll)

    # clf = SVC(probability=True, kernel='rbf', class_weight='balanced', verbose=False)
    bst = XGBClassifier(n_estimators=500, max_depth=6,
                        learning_rate=0.1, objective='binary:logistic', device='cuda')

    print(f"正在 GPU 上訓練 SVM (樣本數: {xTrainAll.shape[0]})...")
    # clf.fit(xTrainScaled, yTrainAll)
    bst.fit(xTrainScaled.get(), yTrainAll.get())
    modelTrainElapsed = time.perf_counter() - modelTrainStart
    print("GPU 訓練完成！")
    print(f"模型訓練耗時: {modelTrainElapsed:.2f} 秒")

    # 3. 準備測試集
    inferStart = time.perf_counter()
    xTestFeat = cp.asarray(allDataFeatures[testFile])
    yTest = np.asarray(allDataLabels[testFile]).astype(int)

    xTestScaled = scaler.transform(xTestFeat)
    # 4. 取得分數與 QUBO 優化
    # scores = clf.predict_proba(xTestScaled)[:, 1]
    scores = bst.predict_proba(xTestScaled)[:, 1]

    # 轉回 numpy 格式供 QUBO (CPU) 使用
    if hasattr(scores, 'get'):
        scores = scores.get()
    else:
        scores = np.asarray(scores)

    # --- Baseline 與最佳 QUBO ---
    yBaseline = (scores > 0.5).astype(int)
    yQubo = solve_qubo_seizure(
        scores, lmbda=bestLambda, threshold=bestThreshold)
    yQubo = np.asarray(yQubo).astype(int)
    inferElapsed = time.perf_counter() - inferStart

    # 5. 紀錄結果
    bF1 = f1_score(yTest, yBaseline, zero_division=0)
    qF1 = f1_score(yTest, yQubo, zero_division=0)
    fileElapsed = time.perf_counter() - fileStart
    results.append({
        'File': testFile,
        'Num_Test_Samples': len(yTest),
        'Num_Positive': int(np.sum(yTest)),
        'Baseline_Positive_Pred': int(np.sum(yBaseline)),
        'QUBO_Positive_Pred': int(np.sum(yQubo)),
        'Best_Lambda': bestLambda,
        'Best_Threshold': bestThreshold,
        'Val_Score': bestValScore,
        'Baseline_F1': bF1,
        'QUBO_F1': qF1,
        'Improvement': qF1 - bF1,
        'Time_Cache_s': cacheElapsed,
        'Time_QUBO_Tune_s': tuneElapsed,
        'Time_TrainPrep_s': trainPrepElapsed,
        'Time_ModelTrain_s': modelTrainElapsed,
        'Time_Infer_QUBO_s': inferElapsed,
        'Time_TotalFile_s': fileElapsed
    })
    print(
        f"檔案 {testFile} 完成！提升幅度: {qF1 - bF1:.4f}, "
        f"總耗時: {fileElapsed:.2f} 秒"
    )

# --- 6. 輸出總結表 ---
dfResults = pd.DataFrame(results)
globalElapsed = time.perf_counter() - globalStart
outputDir = "experimentLogs"
os.makedirs(outputDir, exist_ok=True)
dfResults.to_pickle(
    f"{outputDir}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu_results.pkl")
print("\n" + "="*40)
print("GPU 實驗總結報告")
print(dfResults)
print(f"平均提升幅度: {dfResults['Improvement'].mean():.4f}")
print("\n--- 時間統計 ---")
print(f"前處理耗時: {preprocessElapsed:.2f} 秒")
print(f"主迴圈總耗時: {globalElapsed:.2f} 秒")
print(f"整體總耗時: {preprocessElapsed + globalElapsed:.2f} 秒")
if len(dfResults) > 0:
    print(f"平均 cache 耗時: {dfResults['Time_Cache_s'].mean():.2f} 秒")
    print(f"平均 QUBO 調參耗時: {dfResults['Time_QUBO_Tune_s'].mean():.2f} 秒")
    print(f"平均訓練資料準備耗時: {dfResults['Time_TrainPrep_s'].mean():.2f} 秒")
    print(f"平均模型訓練耗時: {dfResults['Time_ModelTrain_s'].mean():.2f} 秒")
    print(f"平均測試推論與 QUBO 耗時: {dfResults['Time_Infer_QUBO_s'].mean():.2f} 秒")
    print(f"平均每檔案總耗時: {dfResults['Time_TotalFile_s'].mean():.2f} 秒")
