from FeatureExtraction import extract_band_power
from parser import parse_seizure_file
from pipeline import preprocess_one_file
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    FILE_PATH = "DESTINATION/chb01/chb01-summary.txt"
    seizures = parse_seizure_file(FILE_PATH)
    seizure_intervals = seizures['chb01_03.edf']
    X, y = preprocess_one_file(
        'DESTINATION/chb01/chb01_03.edf', seizure_intervals)
    print(f"資料形狀: {X.shape}, 標籤形狀: {y.shape}")
    print(f"發作 Epoch 數量: {np.sum(y)}")
    # 轉換整個 X 變成特徵矩陣
    # X 的形狀是 (Epochs, Channels, Time)
    # 我們需要變成 (Epochs, Features)
    print("開始提取特徵...")
    X_features = np.array([extract_band_power(e) for e in X])
    print(f"特徵矩陣形狀: {X_features.shape}")

    # 2. 資料分割 (這裡做簡單示範，實際要小心資料洩漏)
    # 使用 stratify=y 確保訓練集和測試集都有發作資料
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.3, stratify=y, random_state=42
    )

    # 3. 正規化 (SVM 對數值大小很敏感，必須做)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 訓練 SVM
    print("訓練 SVM 中...")
    clf = SVC(probability=True, kernel='rbf', class_weight='balanced')
    clf.fit(X_train_scaled, y_train)

    # 5. 測試與取得「一元分數 (Unary Score)」
    y_pred = clf.predict(X_test_scaled)
    print(f"Baseline F1 Score: {f1_score(y_test, y_pred):.4f}")

    # ----------------------------------------------------
    # 這是最重要的部分！這就是 QUBO 的輸入！
    # ----------------------------------------------------
    # 我們拿整段訊號（不分 train/test）來看看隨時間變化的機率
    all_scores = clf.predict_proba(scaler.transform(X_features))[:, 1]

    # 這裡 all_scores[t] 就是第 t 秒是發作的可能性
    # 你會發現這個分數在時間軸上會跳動，這就是我們要優化的地方
