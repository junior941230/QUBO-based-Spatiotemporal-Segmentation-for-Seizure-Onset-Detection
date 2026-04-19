import pandas as pd
import matplotlib.pyplot as plt

# 讀檔

def baselineVSQubo(df):
    # 依 Improvement 排序，比較好看
    df = df.sort_values("Improvement", ascending=False)

    x = range(len(df))

    plt.figure(figsize=(14, 6))
    plt.plot(x, df["Baseline_F1"], marker='o', label="Baseline F1")
    plt.plot(x, df["QUBO_F1"], marker='s', label="QUBO F1")
    plt.xticks(x, df["File"], rotation=90)
    plt.ylabel("F1 Score")
    plt.title("Baseline vs QUBO F1 by File")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultVisualize/baseline_vs_qubo.png")

def improvement(df):
    df = df.sort_values("Improvement", ascending=False)
    plt.figure(figsize=(14, 6))
    plt.bar(df["File"], df["Improvement"])
    plt.xticks(rotation=90)
    plt.ylabel("QUBO_F1 - Baseline_F1")
    plt.title("Improvement by File")
    plt.axhline(0, linestyle="--")
    plt.tight_layout()
    plt.savefig("resultVisualize/improvement.png")

def bestParam(df):
    plt.figure(figsize=(8, 4))
    df["Best_Lambda"].value_counts().sort_index().plot(kind="bar")
    plt.title("Best Lambda Distribution")
    plt.xlabel("Lambda")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("resultVisualize/best_lambda.png")

    plt.figure(figsize=(8, 4))
    df["Best_Threshold"].value_counts().sort_index().plot(kind="bar")
    plt.title("Best Threshold Distribution")
    plt.xlabel("Threshold")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("resultVisualize/best_threshold.png")

def ValScoreVsImprovement(df):
    plt.figure(figsize=(6, 5))
    plt.scatter(df["Val_Score"], df["Improvement"])
    for _, row in df.iterrows():
        plt.text(row["Val_Score"], row["Improvement"], row["File"], fontsize=8)

    plt.xlabel("Validation Score")
    plt.ylabel("Improvement")
    plt.title("Validation Score vs Improvement")
    plt.tight_layout()
    plt.savefig("val_score_vs_improvement.png")

if __name__ == "__main__":
    dfold = pd.read_pickle("experimentLogs/20260323_032622_svm_results.pkl")
    dfnew = pd.read_pickle("experimentLogs/20260328_170826_XG_results.pkl")
    
    df_seizureNew = dfnew[dfnew["Num_Positive"] > 0]
    df_seizureOld = dfold[dfold["Num_Positive"] > 0]

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
    new_values = [df_seizureNew["Baseline_F1"].mean(), df_seizureNew["QUBO_F1"].mean(), df_seizureNew["Improvement"].mean()]
    old_values = [df_seizureOld["Baseline_F1"].mean(), df_seizureOld["QUBO_F1"].mean(), df_seizureOld["Improvement"].mean()]
    
    x = range(len(categories))
    plt.bar([i - 0.2 for i in x], new_values, width=0.4, label="XGBoost")
    plt.bar([i + 0.2 for i in x], old_values, width=0.4, label="SVM")
    plt.xticks(x, categories)
    plt.ylabel("Score")
    plt.title("New vs Old Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultVisualize/new_vs_old_comparison.png")