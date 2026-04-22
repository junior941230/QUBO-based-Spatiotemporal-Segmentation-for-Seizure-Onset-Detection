import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dfold = pd.read_pickle(
        "experimentLogs/20260421_073735_gpu_results_kfold.pkl")
    dfnew = pd.read_pickle(
        "experimentLogs/20260422_052115_gpu_results_opcv.pkl")
    
    df_seizureOld = dfold[dfold["Num_Positive"] > 0]
    df_seizureNew = dfnew[dfnew["Num_Positive"] > 0]
    

    print("Seizure files 平均:")
    print("New - Baseline F1:", df_seizureNew["Baseline_F1"].mean())
    print("New - QUBO F1:", df_seizureNew["QUBO_F1"].mean())
    print("New - Improvement:", df_seizureNew["Improvement"].mean())
    print("\nOld - Baseline F1:", df_seizureOld["Baseline_F1"].mean())
    print("Old - QUBO F1:", df_seizureOld["QUBO_F1"].mean())
    print("Old - Improvement:", df_seizureOld["Improvement"].mean())

    # 比較圖表
    plt.figure(figsize=(10, 5))
    categories = ["Baseline F1", "QUBO F1", "Improvement"]
    new_values = [df_seizureNew["Baseline_F1"].mean(
    ), df_seizureNew["QUBO_F1"].mean(), df_seizureNew["Improvement"].mean()]
    old_values = [df_seizureOld["Baseline_F1"].mean(
    ), df_seizureOld["QUBO_F1"].mean(), df_seizureOld["Improvement"].mean()]

    x = range(len(categories))
    plt.bar([i - 0.2 for i in x], new_values, width=0.4, label="LOFO")
    plt.bar([i + 0.2 for i in x], old_values, width=0.4, label="5fold")
    plt.xticks(x, categories)
    plt.ylabel("Score")
    plt.title("LOFO vs 5fold Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("new_vs_old_comparison.png")
