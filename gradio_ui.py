import os
import re
import time
from pathlib import Path
from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from parser import parse_seizure_file
from pipeline import processAllFiles, solve_chain_qubo_exact, solve_qubo_seizure

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


DESTINATION_DIR = Path("DESTINATION")
LAMBDA_LIST = [0.5, 1.0, 1.5, 2.0, 3.0]
THRESHOLD_LIST = [0.3, 0.4, 0.45, 0.5, 0.6]
TUNE_ALPHA = 0.2
BASELINE_THRESHOLD = 0.5


def log_step(message):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


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


def build_validation_score_cache_lofo(training_files, features, labels, baseline):
    cache = {}
    log_step(f"[Tune-LOFO] start, files={len(training_files)}")

    for val_file in training_files:
        log_step(f"[Tune-LOFO] validation file={val_file}")
        inner_train_files = [name for name in training_files if name != val_file]
        if not inner_train_files:
            continue

        x_train = np.concatenate([features[f] for f in inner_train_files])
        y_train = np.concatenate([labels[f] for f in inner_train_files]).astype(int)
        if len(np.unique(y_train)) < 2:
            continue

        x_val = features[val_file]
        y_val = np.asarray(labels[val_file]).astype(int)
        scores = np.asarray(predict_scores(baseline, x_train, y_train, x_val))
        cache[val_file] = {"scores": scores, "y_val": y_val}

    log_step(f"[Tune-LOFO] done, cached_files={len(cache)}")
    return cache


def build_validation_score_cache_kfold(training_files, features, labels, baseline, n_splits=5):
    cache = {}
    train_array = np.array(training_files)
    log_step(f"[Tune-NFold] start, files={len(training_files)}, requested_splits={n_splits}")

    file_has_seizure = np.array([1 if np.sum(labels[f]) > 0 else 0 for f in training_files])
    min_class_count = min(np.sum(file_has_seizure), np.sum(1 - file_has_seizure))

    if min_class_count >= n_splits and n_splits >= 2:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(train_array, file_has_seizure)
        log_step("[Tune-NFold] using StratifiedKFold")
    else:
        safe_splits = max(2, min(n_splits, len(training_files)))
        splitter = KFold(n_splits=safe_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(train_array)
        log_step(f"[Tune-NFold] using KFold, effective_splits={safe_splits}")

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        inner_train_files = train_array[train_idx].tolist()
        val_files = train_array[val_idx].tolist()
        log_step(
            f"[Tune-NFold] fold={fold_idx}, train_files={len(inner_train_files)}, val_files={len(val_files)}"
        )

        x_train = np.concatenate([features[f] for f in inner_train_files])
        y_train = np.concatenate([labels[f] for f in inner_train_files]).astype(int)

        if len(np.unique(y_train)) < 2:
            continue

        for val_file in val_files:
            x_val = features[val_file]
            y_val = np.asarray(labels[val_file]).astype(int)
            scores = np.asarray(predict_scores(baseline, x_train, y_train, x_val))
            cache[val_file] = {"scores": scores, "y_val": y_val}

    log_step(f"[Tune-NFold] done, cached_files={len(cache)}")
    return cache


def build_validation_score_cache(training_files, features, labels, baseline, tune_mode, n_splits=5):
    if tune_mode == "lofo":
        return build_validation_score_cache_lofo(training_files, features, labels, baseline)
    if tune_mode == "nfold":
        return build_validation_score_cache_kfold(training_files, features, labels, baseline, n_splits)
    raise ValueError(f"Unknown tuning mode: {tune_mode}")


def tune_qubo_params_from_cache(score_cache, solver, lambda_list, threshold_list, alpha=TUNE_ALPHA):
    if not lambda_list or not threshold_list:
        raise ValueError("lambda_list and threshold_list must not be empty")

    best_score = -1e9
    best_lambda = float(lambda_list[0])
    best_threshold = float(threshold_list[0])

    log_step(
        f"[QUBO-Tune] start, cache_files={len(score_cache)}, "
        f"lambda_count={len(lambda_list)}, threshold_count={len(threshold_list)}"
    )

    for lmbda in lambda_list:
        for threshold in threshold_list:
            seizure_f1_list = []
            nonseizure_fp_list = []

            for data in score_cache.values():
                scores = data["scores"]
                y_val = data["y_val"]

                y_qubo_val = np.asarray(
                    solver(scores, lmbda=float(lmbda), threshold=float(threshold))
                ).astype(int)

                if np.sum(y_val) > 0:
                    seizure_f1_list.append(f1_score(y_val, y_qubo_val, zero_division=0))
                else:
                    nonseizure_fp_list.append(float(np.mean(y_qubo_val)))

            mean_seizure_f1 = float(np.mean(seizure_f1_list)) if seizure_f1_list else 0.0
            mean_nonseizure_fp = float(np.mean(nonseizure_fp_list)) if nonseizure_fp_list else 0.0
            combined_score = mean_seizure_f1 - alpha * mean_nonseizure_fp

            log_step(
                "[QUBO-Tune] "
                f"lambda={lmbda}, threshold={threshold}, "
                f"seizure_f1={mean_seizure_f1:.4f}, nonseizure_fp={mean_nonseizure_fp:.4f}, "
                f"combined={combined_score:.4f}"
            )

            if combined_score > best_score:
                best_score = combined_score
                best_lambda = float(lmbda)
                best_threshold = float(threshold)
                log_step(
                    f"[QUBO-Tune] new_best lambda={best_lambda}, "
                    f"threshold={best_threshold}, score={best_score:.4f}"
                )

    log_step(
        f"[QUBO-Tune] done, best_lambda={best_lambda}, "
        f"best_threshold={best_threshold}, best_score={best_score:.4f}"
    )
    return best_lambda, best_threshold, float(best_score)


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
    tune_mode,
    tune_n_splits,
    max_files_per_subject,
    n_jobs,
    progress=gr.Progress(),
):
    if not selected_subjects:
        return "Please select at least one subject", pd.DataFrame(), None, None

    run_start = time.perf_counter()
    log_step(
        "[Run] start "
        f"subjects={selected_subjects}, baseline={baseline}, solver={solver_name}, "
        f"tune_mode={tune_mode}, tune_n_splits={int(tune_n_splits)}, "
        f"max_files_per_subject={int(max_files_per_subject)}, n_jobs={int(n_jobs)}"
    )

    progress(0.02, desc="Collecting files")
    file_paths, seizure_times, notes = collect_files_and_seizures(
        selected_subjects, int(max_files_per_subject)
    )
    log_step(
        f"[Run] collected files={len(file_paths)}, seizure_mapping_files={len(seizure_times)}"
    )

    if len(file_paths) < 2:
        log_step("[Run] stop: less than 2 files")
        return "Need at least 2 EDF files to run leave-one-file-out", pd.DataFrame(), None, None

    progress(0.12, desc="Preprocessing EDF files")
    preprocess_start = time.perf_counter()
    features, labels = processAllFiles(file_paths, seizure_times, nJobs=int(n_jobs))
    preprocess_elapsed = time.perf_counter() - preprocess_start
    log_step(f"[Run] preprocessing done, elapsed={preprocess_elapsed:.2f}s")
    solver = get_qubo_solver(solver_name)

    test_files = [os.path.basename(path) for path in file_paths]
    rows = []
    detail_cache = {}
    skipped = []

    loop_total = max(1, len(test_files))
    for idx, test_file in enumerate(test_files):
        progress(0.15 + 0.8 * ((idx + 1) / loop_total), desc=f"Training/Evaluating {test_file}")
        file_start = time.perf_counter()
        log_step(f"[File] start test_file={test_file} ({idx + 1}/{len(test_files)})")

        train_files = [name for name in test_files if name != test_file]
        train_files = [name for name in train_files if name in features and name in labels]
        log_step(f"[File] available train_files={len(train_files)}")

        if len(train_files) < 2 and tune_mode == "nfold":
            skipped.append(f"{test_file}: not enough files for inner validation tuning")
            log_step(f"[File] skip {test_file}: not enough files for nfold")
            continue

        try:
            cache_start = time.perf_counter()
            score_cache = build_validation_score_cache(
                train_files,
                features,
                labels,
                baseline,
                tune_mode=tune_mode,
                n_splits=min(int(tune_n_splits), len(train_files)),
            )
            cache_elapsed = time.perf_counter() - cache_start
            log_step(
                f"[File] cache built for {test_file}, cached_files={len(score_cache)}, "
                f"elapsed={cache_elapsed:.2f}s"
            )
        except Exception as exc:
            skipped.append(f"{test_file}: failed to build tuning cache ({exc})")
            log_step(f"[File] cache build failed for {test_file}: {exc}")
            continue

        if not score_cache:
            skipped.append(f"{test_file}: tuning cache is empty")
            log_step(f"[File] skip {test_file}: empty cache")
            continue

        try:
            tune_start = time.perf_counter()
            best_lambda, best_threshold, best_val_score = tune_qubo_params_from_cache(
                score_cache,
                solver,
                LAMBDA_LIST,
                THRESHOLD_LIST,
                alpha=TUNE_ALPHA,
            )
            tune_elapsed = time.perf_counter() - tune_start
            log_step(
                f"[File] tuned {test_file}, best_lambda={best_lambda}, "
                f"best_threshold={best_threshold}, val_score={best_val_score:.4f}, "
                f"elapsed={tune_elapsed:.2f}s"
            )
        except Exception as exc:
            skipped.append(f"{test_file}: failed during QUBO tuning ({exc})")
            log_step(f"[File] tuning failed for {test_file}: {exc}")
            continue

        x_train_list = [features[name] for name in train_files]
        y_train_list = [labels[name] for name in train_files]

        if not x_train_list:
            skipped.append(f"{test_file}: empty training set")
            log_step(f"[File] skip {test_file}: empty training list")
            continue

        x_train = np.concatenate(x_train_list)
        y_train = np.concatenate(y_train_list).astype(int)
        log_step(f"[File] train shape={x_train.shape}, positive_labels={int(np.sum(y_train))}")

        x_test = features[test_file]
        y_test = np.asarray(labels[test_file]).astype(int)
        log_step(f"[File] test shape={x_test.shape}, positive_labels={int(np.sum(y_test))}")

        if len(np.unique(y_train)) < 2:
            skipped.append(f"{test_file}: training labels only have one class")
            log_step(f"[File] skip {test_file}: single class training labels")
            continue

        try:
            infer_start = time.perf_counter()
            scores = np.asarray(predict_scores(baseline, x_train, y_train, x_test))
            y_baseline = (scores >= BASELINE_THRESHOLD).astype(int)
            y_qubo = np.asarray(
                solver(scores, lmbda=float(best_lambda), threshold=float(best_threshold))
            ).astype(int)
            infer_elapsed = time.perf_counter() - infer_start
            log_step(f"[File] inference done for {test_file}, elapsed={infer_elapsed:.2f}s")
        except Exception as exc:
            skipped.append(f"{test_file}: {exc}")
            log_step(f"[File] inference failed for {test_file}: {exc}")
            continue

        baseline_f1 = f1_score(y_test, y_baseline, zero_division=0)
        qubo_f1 = f1_score(y_test, y_qubo, zero_division=0)
        log_step(
            f"[File] result {test_file}: baseline_f1={baseline_f1:.4f}, "
            f"qubo_f1={qubo_f1:.4f}, improvement={qubo_f1 - baseline_f1:.4f}"
        )

        rows.append(
            {
                "file": test_file,
                "baseline_f1": baseline_f1,
                "qubo_f1": qubo_f1,
                "improvement": qubo_f1 - baseline_f1,
                "best_lambda": best_lambda,
                "best_threshold": best_threshold,
                "val_score": best_val_score,
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

        file_elapsed = time.perf_counter() - file_start
        log_step(f"[File] done {test_file}, total_elapsed={file_elapsed:.2f}s")

    if not rows:
        note_text = "\n".join(notes + skipped) if (notes or skipped) else "No valid result"
        log_step("[Run] failed: no valid rows")
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
        f"Baseline={baseline}, Solver={solver_name}, "
        f"TuningMode={tune_mode}, Nfold={int(tune_n_splits)}, "
        f"QUBO tuning grid(lambda={LAMBDA_LIST}, threshold={THRESHOLD_LIST})\n"
        f"Mean baseline F1={result_df['baseline_f1'].mean():.4f}, "
        f"Mean QUBO F1={result_df['qubo_f1'].mean():.4f}, "
        f"Mean improvement={result_df['improvement'].mean():.4f}"
    )

    if note_lines:
        summary_text += "\n\n" + "\n".join(note_lines)

    run_elapsed = time.perf_counter() - run_start
    log_step(
        f"[Run] done, evaluated_files={len(result_df)}, skipped={len(skipped)}, "
        f"total_elapsed={run_elapsed:.2f}s"
    )

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
            gr.Markdown(
                "QUBO lambda/threshold are auto-tuned per test file "
                "using inner-validation (same idea as gpu.py)."
            )

        with gr.Row():
            tune_mode = gr.Radio(
                choices=["lofo", "nfold"],
                value="nfold",
                label="Tuning Strategy",
            )
            tune_n_splits = gr.Slider(2, 10, value=5, step=1, label="Nfold Splits (for nfold)")

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
                tune_mode,
                tune_n_splits,
                max_files_per_subject,
                n_jobs,
            ],
            outputs=[summary_output, result_table, summary_plot, detail_plot],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
