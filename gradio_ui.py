import os
import re
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from parser import parse_seizure_file
from pipeline import processAllFiles, solve_chain_qubo_exact, solve_qubo_seizure

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


DESTINATION_DIR = Path("DESTINATION")


def discover_subjects(base_dir=DESTINATION_DIR):
    if not base_dir.exists():
        return []
    pattern = re.compile(r"chb\d{2}")
    return sorted(
        item.name for item in base_dir.iterdir() if item.is_dir() and pattern.fullmatch(item.name)
    )


def collect_files_and_seizures(subjects, max_files_per_subject):
    file_paths = []
    seizure_times = {}
    notes = []

    for subject in subjects:
        subject_dir = DESTINATION_DIR / subject
        summary_path = subject_dir / f"{subject}-summary.txt"

        if not subject_dir.exists():
            notes.append(f"Skip {subject}: subject directory not found")
            continue

        if not summary_path.exists():
            notes.append(f"Skip {subject}: summary file not found")
            continue

        seizure_times.update(parse_seizure_file(str(summary_path)))

        edf_files = sorted(subject_dir.glob("*.edf"))
        if max_files_per_subject > 0:
            edf_files = edf_files[:max_files_per_subject]

        file_paths.extend(str(path) for path in edf_files)

    return file_paths, seizure_times, notes


def predict_scores(baseline, x_train, y_train, x_test):
    if baseline == "svm":
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        clf = SVC(
            probability=True,
            kernel="rbf",
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(x_train_scaled, y_train)
        return clf.predict_proba(x_test_scaled)[:, 1]

    if baseline == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed")
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
        clf.fit(x_train, y_train)
        return clf.predict_proba(x_test)[:, 1]

    raise ValueError(f"Unknown baseline: {baseline}")


def get_qubo_solver(name):
    if name == "solve_qubo_seizure":
        return solve_qubo_seizure
    if name == "solve_chain_qubo_exact":
        return solve_chain_qubo_exact
    raise ValueError(f"Unknown solver: {name}")


def build_summary_plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    mean_baseline = float(df["baseline_f1"].mean())
    mean_qubo = float(df["qubo_f1"].mean())

    axes[0].bar(["Baseline", "QUBO"], [mean_baseline, mean_qubo], color=["#5B8FF9", "#5AD8A6"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Mean F1")
    axes[0].set_ylabel("F1")

    axes[1].hist(df["improvement"], bins=min(15, len(df)), color="#F6BD16", edgecolor="black")
    axes[1].axvline(df["improvement"].mean(), color="red", linestyle="--", label="Mean")
    axes[1].set_title("QUBO Improvement Distribution")
    axes[1].set_xlabel("QUBO F1 - Baseline F1")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    return fig


def build_detail_plot(detail, baseline, solver_name):
    y_true = detail["y_true"]
    y_baseline = detail["y_baseline"]
    y_qubo = detail["y_qubo"]
    scores = detail["scores"]
    timeline = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(timeline, y_true, where="post", label="Ground Truth", linewidth=2)
    ax.plot(timeline, scores, label="Baseline Probability", alpha=0.55)
    ax.step(timeline, y_baseline, where="post", label=f"{baseline.upper()} Binary")
    ax.step(timeline, y_qubo, where="post", label=solver_name)

    ax.set_title(f"File Detail: {detail['file_name']}")
    ax.set_xlabel("Epoch (1 sec)")
    ax.set_ylabel("Label / Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def run_experiment(
    selected_subjects,
    baseline,
    solver_name,
    lmbda,
    threshold,
    max_files_per_subject,
    n_jobs,
    progress=gr.Progress(),
):
    if not selected_subjects:
        return "Please select at least one subject", pd.DataFrame(), None, None

    progress(0.02, desc="Collecting files")
    file_paths, seizure_times, notes = collect_files_and_seizures(
        selected_subjects, int(max_files_per_subject)
    )

    if len(file_paths) < 2:
        return "Need at least 2 EDF files to run leave-one-file-out", pd.DataFrame(), None, None

    progress(0.12, desc="Preprocessing EDF files")
    features, labels = processAllFiles(file_paths, seizure_times, nJobs=int(n_jobs))
    solver = get_qubo_solver(solver_name)

    test_files = [os.path.basename(path) for path in file_paths]
    rows = []
    detail_cache = {}
    skipped = []

    loop_total = max(1, len(test_files))
    for idx, test_file in enumerate(test_files):
        progress(0.15 + 0.8 * ((idx + 1) / loop_total), desc=f"Training/Evaluating {test_file}")

        train_files = [name for name in test_files if name != test_file]
        x_train_list = [features[name] for name in train_files if name in features]
        y_train_list = [labels[name] for name in train_files if name in labels]

        if not x_train_list:
            skipped.append(f"{test_file}: empty training set")
            continue

        x_train = np.concatenate(x_train_list)
        y_train = np.concatenate(y_train_list).astype(int)

        x_test = features[test_file]
        y_test = np.asarray(labels[test_file]).astype(int)

        if len(np.unique(y_train)) < 2:
            skipped.append(f"{test_file}: training labels only have one class")
            continue

        try:
            scores = np.asarray(predict_scores(baseline, x_train, y_train, x_test))
            y_baseline = (scores >= threshold).astype(int)
            y_qubo = np.asarray(solver(scores, lmbda=float(lmbda), threshold=float(threshold))).astype(int)
        except Exception as exc:
            skipped.append(f"{test_file}: {exc}")
            continue

        baseline_f1 = f1_score(y_test, y_baseline, zero_division=0)
        qubo_f1 = f1_score(y_test, y_qubo, zero_division=0)

        rows.append(
            {
                "file": test_file,
                "baseline_f1": baseline_f1,
                "qubo_f1": qubo_f1,
                "improvement": qubo_f1 - baseline_f1,
                "baseline_precision": precision_score(y_test, y_baseline, zero_division=0),
                "qubo_precision": precision_score(y_test, y_qubo, zero_division=0),
                "baseline_recall": recall_score(y_test, y_baseline, zero_division=0),
                "qubo_recall": recall_score(y_test, y_qubo, zero_division=0),
                "epochs": int(len(y_test)),
                "seizure_epochs": int(y_test.sum()),
            }
        )

        detail_cache[test_file] = {
            "file_name": test_file,
            "y_true": y_test,
            "y_baseline": y_baseline,
            "y_qubo": y_qubo,
            "scores": scores,
        }

    if not rows:
        note_text = "\n".join(notes + skipped) if (notes or skipped) else "No valid result"
        return f"Run failed\n\n{note_text}", pd.DataFrame(), None, None

    result_df = pd.DataFrame(rows).sort_values("improvement", ascending=False).reset_index(drop=True)
    summary_fig = build_summary_plot(result_df)

    best_file = result_df.iloc[0]["file"]
    detail_fig = build_detail_plot(detail_cache[best_file], baseline, solver_name)

    progress(1.0, desc="Done")

    note_lines = []
    if notes:
        note_lines.append("Notes:")
        note_lines.extend(f"- {note}" for note in notes)
    if skipped:
        note_lines.append("Skipped:")
        note_lines.extend(f"- {item}" for item in skipped[:10])
        if len(skipped) > 10:
            note_lines.append(f"- ... and {len(skipped) - 10} more")

    summary_text = (
        f"Finished with {len(result_df)} evaluated files\n"
        f"Selected subjects: {', '.join(selected_subjects)}\n"
        f"Baseline={baseline}, Solver={solver_name}, lambda={lmbda}, threshold={threshold}\n"
        f"Mean baseline F1={result_df['baseline_f1'].mean():.4f}, "
        f"Mean QUBO F1={result_df['qubo_f1'].mean():.4f}, "
        f"Mean improvement={result_df['improvement'].mean():.4f}"
    )

    if note_lines:
        summary_text += "\n\n" + "\n".join(note_lines)

    return summary_text, result_df, summary_fig, detail_fig


def build_ui():
    subjects = discover_subjects()

    with gr.Blocks(title="QUBO Seizure UI") as demo:
        gr.Markdown("# QUBO Seizure Experiment Dashboard")
        gr.Markdown(
            "Select subjects, baseline model, and QUBO solver. "
            "The app runs leave-one-file-out evaluation and shows summary visualizations."
        )

        with gr.Row():
            selected_subjects = gr.CheckboxGroup(
                choices=subjects,
                value=subjects[:1],
                label="Subjects (multi-select)",
            )

        with gr.Row():
            baseline = gr.Radio(
                choices=["svm", "xgboost"],
                value="svm",
                label="Baseline",
            )
            solver_name = gr.Radio(
                choices=["solve_qubo_seizure", "solve_chain_qubo_exact"],
                value="solve_chain_qubo_exact",
                label="QUBO Solver",
            )

        with gr.Row():
            lmbda = gr.Slider(0.1, 5.0, value=1.5, step=0.1, label="Lambda")
            threshold = gr.Slider(0.1, 0.9, value=0.45, step=0.01, label="Threshold")

        with gr.Row():
            max_files_per_subject = gr.Slider(
                0, 30, value=5, step=1, label="Max EDF files per subject (0=all)"
            )
            n_jobs = gr.Slider(-1, 16, value=-1, step=1, label="Preprocess parallel jobs")

        run_button = gr.Button("Run Experiment", variant="primary")

        summary_output = gr.Textbox(label="Run Summary", lines=10)
        result_table = gr.Dataframe(label="Per-file Metrics")
        summary_plot = gr.Plot(label="Overall Visualization")
        detail_plot = gr.Plot(label="Best-file Detail Visualization")

        run_button.click(
            fn=run_experiment,
            inputs=[
                selected_subjects,
                baseline,
                solver_name,
                lmbda,
                threshold,
                max_files_per_subject,
                n_jobs,
            ],
            outputs=[summary_output, result_table, summary_plot, detail_plot],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
