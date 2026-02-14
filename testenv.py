import cuml
from cuml.svm import SVC
import cupy as cp

# 建立假數據測試
X = cp.random.rand(100, 10).astype('float32')
y = cp.random.randint(0, 2, 100).astype('float32')

clf = SVC(kernel='rbf', probability=True)
clf.fit(X, y)
print("GPU SVM 訓練成功！")