import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_pickle("experimentLogs/20260314_233207_gpu_results.pkl")
    meaningfuldf = df[df["Baseline_F1"] != 0]
    comparisonPlot = meaningfuldf[['Baseline_F1', 'QUBO_F1']].plot(kind='bar', figsize=(10, 6))
    plt.title('Comparison of Baseline $F_{1}$ and QUBO $F_{1}$')
    plt.ylabel('$F_{1}$ Score')
    plt.xlabel('File Name')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('resultVisualize/f1_comparison_bar.png')
    plt.clf()  # 清除當前圖表以繪製下一個

    # 繪製進步幅度長條圖
    improvementPlot = meaningfuldf['Improvement'].plot(kind='bar', color='green', figsize=(10, 6))
    plt.title('Improvement (QUBO - Baseline)')
    plt.ylabel('Improvement Value')
    plt.xlabel('File Name')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('resultVisualize/improvement_bar.png')