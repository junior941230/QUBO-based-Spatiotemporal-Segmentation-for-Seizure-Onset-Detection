# QUBO-based Spatiotemporal Segmentation for Seizure Onset Detection

## 專案簡介
本專案從 EEG (腦波圖) 發作可能性分數出發，構建 QUBO (Quadratic Unconstrained Binary Optimization) 模型，旨在改善癲癇發作起始偵測的時間穩定性與準確度。

## 專案架構
* `pipeline.py`: 處理整體運算流程的主要程式。
* `FeatureExtraction.py`: 負責從原始 EEG 資料中提取相關特徵。
* `cpu.py` / `gpu.py`: 提供基於 CPU 與 GPU 的核心運算邏輯與最佳化求解。
* `gradio_ui.py`: 基於 Gradio 建立的網頁互動式使用者介面。
* `parser.py`: 負責命令列參數解析與設定。
* `test.py`: 系統測試腳本。
* `requirements.txt`: 執行本專案所需的 Python 依賴套件。
* `draw/` & `images/`: 存放圖表繪製腳本與視覺化結果（如 `new_vs_old_comparison.png`）。
* `experimentLogs/` & `results/`: 存放實驗紀錄與模型輸出結果。

## 環境建置
請確保系統已安裝 Python 3.x 環境。請在專案根目錄執行以下指令以安裝依賴套件：

```bash
pip install -r requirements.txt
