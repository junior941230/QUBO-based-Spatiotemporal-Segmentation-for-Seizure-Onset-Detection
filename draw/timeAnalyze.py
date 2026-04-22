import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib_fontja
import pandas as pd
import numpy as np

# ── 設定中文字型（避免亂碼）──────────────────────────────
# plt.rcParams['font.family'] = 'Microsoft JhengHei'  # Windows
# # plt.rcParams['font.family'] = 'Heiti TC'          # macOS
# plt.rcParams['axes.unicode_minus'] = False

# ── 欄位定義 ─────────────────────────────────────────────
time_cols = [
    'Time_Cache_s',
    'Time_QUBO_Tune_s',
    'Time_TrainPrep_s',
    'Time_ModelTrain_s',
    'Time_Infer_QUBO_s',
]
labels = [
    'Cache\n快取',
    'QUBO\n調參',
    '訓練資料\n準備',
    '模型\n訓練',
    '推論 &\nQUBO',
]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
dfResults = pd.read_pickle('experimentLogs/20260421_073735_gpu_results_kfold.pkl')
if len(dfResults) > 0:

    means = [dfResults[c].mean() for c in time_cols]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('各階段耗時分析', fontsize=16, fontweight='bold', y=1.02)

    # ── 圖一：平均耗時長條圖 ──────────────────────────────
    ax1 = axes[0]
    bars = ax1.bar(labels, means, color=colors, edgecolor='white',
                   linewidth=0.8, zorder=3)
    ax1.set_title('各階段平均耗時', fontsize=13, pad=10)
    ax1.set_ylabel('耗時（秒）', fontsize=11)
    ax1.set_xlabel('處理階段', fontsize=11)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax1.set_axisbelow(True)

    # 在每個 bar 上方標數值
    for bar, val in zip(bars, means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(means) * 0.01,
            f'{val:.2f}s',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    # 總耗時參考線
    total_mean = dfResults['Time_TotalFile_s'].mean()
    ax1.axhline(total_mean, color='red', linestyle='--',
                linewidth=1.2, label=f'平均總耗時 {total_mean:.2f}s')
    ax1.legend(fontsize=10)

    # ── 圖二：每檔案堆疊長條圖 ───────────────────────────
    ax2 = axes[1]
    x = np.arange(len(dfResults))
    bottom = np.zeros(len(dfResults))

    for col, label, color in zip(time_cols, labels, colors):
        vals = dfResults[col].values
        ax2.bar(x, vals, bottom=bottom, label=label.replace('\n', ' '),
                color=color, edgecolor='white', linewidth=0.5, zorder=3)
        bottom += vals

    ax2.set_title('每檔案各階段耗時堆疊', fontsize=13, pad=10)
    ax2.set_ylabel('耗時（秒）', fontsize=11)
    ax2.set_xlabel('檔案索引', fontsize=11)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        dfResults.index if 'FileName' not in dfResults.columns
        else dfResults['FileName'],
        rotation=45, ha='right', fontsize=8
    )
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.85)

    plt.tight_layout()
    plt.savefig('timing_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 圖表已儲存為 timing_analysis.png")
