import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

from FeatureExtraction import extract_band_power
from parser import parse_seizure_file
from pipeline import preprocess_one_file, solve_qubo_seizure

def get_data_and_features(edf_path, seizure_intervals):
    """封裝預處理與特徵提取流程"""
    X, y = preprocess_one_file(edf_path, seizure_intervals)
    X_features = np.array([extract_band_power(e) for e in X])
    return X_features, y

if __name__ == "__main__":
    # --- 1. 初始化路徑與資料 ---
    SUMMARY_PATH = "DESTINATION/chb01/chb01-summary.txt"
    seizures = parse_seizure_file(SUMMARY_PATH)
    
    # --- 2. 訓練階段 (Train on File 03) ---
    print(">>> 正在準備訓練資料 (chb01_03)...")
    X_train_raw, y_train = get_data_and_features(
        'DESTINATION/chb01/chb01_03.edf', seizures['chb01_03.edf']
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    print(">>> 訓練 SVM 分類器...")
    clf = SVC(probability=True, kernel='rbf', class_weight='balanced')
    clf.fit(X_train_scaled, y_train)

    # --- 3. 測試階段 (Test on File 04) ---
    # 這裡假設 04 有發作，或是換成另一個有發作的檔案如 01_15
    test_file = 'chb01_04.edf' 
    print(f"\n>>> 正在準備測試資料 ({test_file})...")
    X_test_raw, y_test = get_data_and_features(
        f'DESTINATION/chb01/{test_file}', seizures.get(test_file, [])
    )
    
    X_test_scaled = scaler.transform(X_test_raw) # 必須使用訓練集的 scaler
    
    # 取得 SVM 預測機率 (Unary Scores)
    all_scores_test = clf.predict_proba(X_test_scaled)[:, 1]
    # --- 4. QUBO 優化 ---
    print(">>> 啟動 QUBO 時間平滑優化...")
    # 可以嘗試不同的 lambda 看看效果變化
    LMBDA = 1.0
    y_qubo_test = solve_qubo_seizure(all_scores_test, lmbda=LMBDA, threshold=0.4)

    # --- 5. 結果評估 ---
    # 門檻值預測 (傳統方法)
    y_baseline = (all_scores_test > 0.5).astype(int)
    
    b_f1 = f1_score(y_test, y_baseline)
    q_f1 = f1_score(y_test, y_qubo_test)
    
    print("-" * 30)
    print(f"測試檔案: {test_file}")
    print(f"Baseline F1 Score: {b_f1:.4f}")
    print(f"QUBO 優化後 F1 Score: {q_f1:.4f}")
    print("-" * 30)
    print("詳細報告 (QUBO):")
    print(classification_report(y_test, y_qubo_test))

    # --- 6. 視覺化對比圖 ---
    plt.figure(figsize=(15, 6))
    
    # 第一個子圖：原始機率與真實標籤
    plt.subplot(2, 1, 1)
    plt.plot(y_test, label='True Label (Ground Truth)', color='black', alpha=0.2, linewidth=8)
    plt.plot(all_scores_test, label='SVM Probability (Noisy)', color='blue', alpha=0.6)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    plt.title(f"Test Set ({test_file}): Raw SVM Output")
    plt.legend()
    
    # 第二個子圖：QUBO 優化後的結果
    plt.subplot(2, 1, 2)
    plt.plot(y_test, label='True Label', color='black', alpha=0.2, linewidth=8)
    plt.plot(y_qubo_test, label='QUBO Smoothed Result', color='red', linewidth=2)
    plt.title(f"Test Set ({test_file}): QUBO Optimized (Lambda={LMBDA})")
    plt.legend()
    
    plt.tight_layout()
    plt.show()