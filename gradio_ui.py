import os
import re
import time
import pickle
from pathlib import Path
from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
from cuml.preprocessing import StandardScaler
from cuml.svm import SVC


from parser import parse_seizure_file
from pipeline import processAllFiles, solve_chain_qubo_exact, solve_qubo_seizure

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DESTINATION_DIR = Path("DESTINATION")
RESULTS_DIR = Path("results")
DEFAULT_LAMBDA_LIST = [0.5, 1.0, 1.5, 2.0, 3.0]
DEFAULT_THRESHOLD_LIST = [0.3, 0.4, 0.45, 0.5, 0.6]
TUNE_ALPHA = 0.2
BASELINE_THRESHOLD = 0.5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log_step(message):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def parse_float_list(text, default):
    """Parse a comma-separated float list from UI input."""
    if not text or not text.strip():
        return list(default)
    try:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError:
        log_step(f"[Parse] invalid float list '{text}', fallback to default {default}")
        return list(default)


def discover_subjects(base_dir=DESTINATION_DIR):
    if not base_dir.exists():
        return []
    pattern = re.compile(r"chb\d{2}")
    return sorted(
        item.name for item in base_dir.iterdir()
        if item.is_dir() and pattern.fullmatch(item.name)
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


# ---------------------------------------------------------------------------
# Model & Solver
# ---------------------------------------------------------------------------

def predict_scores(baseline, x_train, y_train, x_test):
    if baseline == "svm":
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        clf = SVC(
            probability=True,
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_SEED,
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
            random_state=RANDOM_SEED,
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


def safe_solver_call(solver, scores, lmbda, threshold):
    """Call QUBO solver with output validation."""
    out = solver(scores, lmbda=float(lmbda), threshold=float(threshold))
    out = np.asarray(out)
    if out.ndim != 1 or out.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Solver output shape {out.shape} does not match scores shape {scores.shape}"
        )
    return (out > 0).astype(int)


# ---------------------------------------------------------------------------
# Validation Cache
# ---------------------------------------------------------------------------

def build_validation_score_cache_lofo(candidate_files, features, labels, baseline):
    cache = {}
    log_step(f"[Cache-LOFO] start, files={len(candidate_files)}")

    for val_file in candidate_files:
        log_step(f"[Cache-LOFO] processing val={val_file}")
        inner_train_files = [name for name in candidate_files if name != val_file]
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

    log_step(f"[Cache-LOFO] done, cached_files={len(cache)}")
    return cache


def build_validation_score_cache_kfold(candidate_files, features, labels, baseline, n_splits=5):
    cache = {}
    arr = np.array(candidate_files)
    effective_splits = max(2, min(int(n_splits), len(candidate_files)))
    log_step(
        f"[Cache-NFold] start, files={len(candidate_files)}, "
        f"requested={n_splits}, effective={effective_splits}"
    )

    file_has_seizure = np.array([1 if np.sum(labels[f]) > 0 else 0 for f in candidate_files])
    min_class_count = min(np.sum(file_has_seizure), np.sum(1 - file_has_seizure))

    if min_class_count >= effective_splits:
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(arr, file_has_seizure)
        log_step("[Cache-NFold] using StratifiedKFold")
    else:
        splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(arr)
        log_step("[Cache-NFold] using KFold")

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        inner_train_files = arr[train_idx].tolist()
        val_files = arr[val_idx].tolist()
        log_step(
            f"[Cache-NFold] fold={fold_idx}, "
            f"train={len(inner_train_files)}, val={len(val_files)}"
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

    log_step(f"[Cache-NFold] done, cached_files={len(cache)}")
    return cache


def build_validation_score_cache(candidate_files, features, labels, baseline, tune_mode, n_splits=5):
    if tune_mode == "lofo":
        return build_validation_score_cache_lofo(candidate_files, features, labels, baseline)
    if tune_mode == "nfold":
        return build_validation_score_cache_kfold(candidate_files, features, labels, baseline, n_splits)
    raise ValueError(f"Unknown tuning mode: {tune_mode}")


# ---------------------------------------------------------------------------
# QUBO Tuning
# ---------------------------------------------------------------------------

def tune_qubo_params_from_cache(score_cache, solver, lambda_list, threshold_list, alpha=TUNE_ALPHA):
    if not lambda_list or not threshold_list:
        raise ValueError("lambda_list and threshold_list must not be empty")
    if not score_cache:
        raise ValueError("score_cache is empty, cannot tune")

    best_score = -1e9
    best_lambda = float(lambda_list[0])
    best_threshold = float(threshold_list[0])

    for lmbda in lambda_list:
        for threshold in threshold_list:
            seizure_f1s = []
            nonseizure_fps = []
            for data in score_cache.values():
                try:
                    y_qubo_val = safe_solver_call(solver, data["scores"], lmbda, threshold)
                except Exception as exc:
                    log_step(f"[QUBO-Tune] solver failed λ={lmbda},θ={threshold}: {exc}")
                    continue
                y_val = data["y_val"]
                if np.sum(y_val) > 0:
                    seizure_f1s.append(f1_score(y_val, y_qubo_val, zero_division=0))
                else:
                    nonseizure_fps.append(float(np.mean(y_qubo_val)))

            mean_f1 = float(np.mean(seizure_f1s)) if seizure_f1s else 0.0
            mean_fp = float(np.mean(nonseizure_fps)) if nonseizure_fps else 0.0
            combined = mean_f1 - alpha * mean_fp

            if combined > best_score:
                best_score = combined
                best_lambda = float(lmbda)
                best_threshold = float(threshold)

    log_step(f"[QUBO-Tune] best λ={best_lambda}, θ={best_threshold}, score={best_score:.4f}")
    return best_lambda, best_threshold, float(best_score)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def build_summary_plot(df):
    seizure_df = df[df["has_seizure"]]
    nonseizure_df = df[~df["has_seizure"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- (1) Seizure files: Mean F1 ---
    if len(seizure_df) > 0:
        means_f1 = [seizure_df["baseline_f1"].mean(), seizure_df["qubo_f1"].mean()]
        axes[0].bar(["Baseline", "QUBO"], means_f1, color=["#5B8FF9", "#5AD8A6"])
        axes[0].set_ylim(0, 1)
        axes[0].set_title(f"Mean F1 on Seizure Files (n={len(seizure_df)})")
        axes[0].set_ylabel("F1")
        for i, v in enumerate(means_f1):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center")
    else:
        axes[0].text(0.5, 0.5, "No seizure files", ha="center", va="center")
        axes[0].set_title("Mean F1 on Seizure Files")
        axes[0].axis("off")

    # --- (2) Non-seizure files: FP rate ---
    if len(nonseizure_df) > 0:
        means_fp = [
            nonseizure_df["baseline_fp_rate"].mean(),
            nonseizure_df["qubo_fp_rate"].mean(),
        ]
        axes[1].bar(["Baseline", "QUBO"], means_fp, color=["#5B8FF9", "#5AD8A6"])
        axes[1].set_title(f"Mean FP Rate on Non-seizure Files (n={len(nonseizure_df)})")
        axes[1].set_ylabel("False Positive Rate")
        top = max(means_fp) if max(means_fp) > 0 else 0.05
        axes[1].set_ylim(0, top * 1.3)
        for i, v in enumerate(means_fp):
            axes[1].text(i, v + top * 0.03, f"{v:.4f}", ha="center")
    else:
        axes[1].text(0.5, 0.5, "No non-seizure files", ha="center", va="center")
        axes[1].set_title("Mean FP Rate on Non-seizure Files")
        axes[1].axis("off")

    # --- (3) Improvement distribution (seizure files only) ---
    target_df = seizure_df if len(seizure_df) > 0 else df
    bins = max(1, min(15, len(target_df)))
    axes[2].hist(target_df["improvement"], bins=bins, color="#F6BD16", edgecolor="black")
    axes[2].axvline(target_df["improvement"].mean(), color="red", linestyle="--", label="Mean")
    axes[2].set_title("QUBO Improvement (Seizure Files)")
    axes[2].set_xlabel("QUBO F1 - Baseline F1")
    axes[2].set_ylabel("Count")
    axes[2].legend()

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


# ---------------------------------------------------------------------------
# Result Saving
# ---------------------------------------------------------------------------

def save_results_pkl(
    result_df, detail_cache, meta, output_dir=RESULTS_DIR
):
    """Save a complete experiment snapshot as pickle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qubo_run_{timestamp}.pkl"
    filepath = output_dir / filename

    payload = {
        "meta": meta,
        "result_df": result_df,
        "detail_cache": detail_cache,
        "saved_at": timestamp,
    }

    with open(filepath, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)

    log_step(f"[Save] results written to {filepath}")
    return str(filepath)


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    selected_subjects,
    baseline,
    solver_name,
    tune_mode,
    tune_n_splits,
    max_files_per_subject,
    n_jobs,
    lambda_text,
    threshold_text,
    reuse_global_cache,
    save_pkl,
    progress=gr.Progress(),
):
    if not selected_subjects:
        return (
            "Please select at least one subject",
            pd.DataFrame(),
            None,
            None,
            "",
        )

    # --- Parameter sanitize ---
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        n_jobs = 1
    tune_n_splits = int(tune_n_splits)
    max_files_per_subject = int(max_files_per_subject)
    lambda_list = parse_float_list(lambda_text, DEFAULT_LAMBDA_LIST)
    threshold_list = parse_float_list(threshold_text, DEFAULT_THRESHOLD_LIST)

    run_start = time.perf_counter()
    log_step(
        f"[Run] start subjects={selected_subjects}, baseline={baseline}, "
        f"solver={solver_name}, mode={tune_mode}, splits={tune_n_splits}, "
        f"max_files={max_files_per_subject}, n_jobs={n_jobs}, "
        f"reuse_cache={reuse_global_cache}, save_pkl={save_pkl}"
    )

    # --- Collect files ---
    progress(0.02, desc="Collecting files")
    file_paths, seizure_times, notes = collect_files_and_seizures(
        selected_subjects, max_files_per_subject
    )
    log_step(f"[Run] files={len(file_paths)}, seizure_map={len(seizure_times)}")

    if len(file_paths) < 2:
        return (
            "Need at least 2 EDF files to run leave-one-file-out",
            pd.DataFrame(), None, None, "",
        )

    # --- Preprocess ---
    progress(0.12, desc="Preprocessing EDF files")
    t0 = time.perf_counter()
    features, labels = processAllFiles(file_paths, seizure_times, nJobs=n_jobs)
    log_step(f"[Run] preprocess done, elapsed={time.perf_counter() - t0:.2f}s")

    test_files = [os.path.basename(path) for path in file_paths]
    missing = [f for f in test_files if f not in features or f not in labels]
    if missing:
        log_step(f"[Run] WARNING missing features for {len(missing)} files, dropping them")
        test_files = [f for f in test_files if f not in missing]
        notes.append(f"Dropped {len(missing)} files with missing features")

    if len(test_files) < 2:
        return (
            "Not enough files after preprocessing",
            pd.DataFrame(), None, None, "",
        )

    solver = get_qubo_solver(solver_name)

    # --- Optional global cache ---
    global_cache = None
    if reuse_global_cache:
        progress(0.14, desc="Building global validation cache")
        try:
            global_cache = build_validation_score_cache(
                test_files, features, labels, baseline,
                tune_mode=tune_mode, n_splits=tune_n_splits,
            )
            log_step(f"[Run] global cache built, size={len(global_cache)}")
        except Exception as exc:
            log_step(f"[Run] global cache failed: {exc}, fallback to per-file cache")
            global_cache = None

    # --- LOFO Loop ---
    rows = []
    detail_cache = {}
    skipped = []
    loop_total = max(1, len(test_files))

    for idx, test_file in enumerate(test_files):
        progress(
            0.15 + 0.8 * ((idx + 1) / loop_total),
            desc=f"Evaluating {test_file}",
        )
        file_start = time.perf_counter()
        log_step(f"[File] {idx + 1}/{len(test_files)} test={test_file}")

        train_files = [f for f in test_files if f != test_file]
        if len(train_files) < 2:
            skipped.append(f"{test_file}: not enough training files")
            continue

        try:
            if global_cache is not None:
                score_cache = {k: v for k, v in global_cache.items() if k != test_file}
                log_step(f"[File] reuse global cache, size={len(score_cache)}")
            else:
                score_cache = build_validation_score_cache(
                    train_files, features, labels, baseline,
                    tune_mode=tune_mode, n_splits=min(tune_n_splits, len(train_files)),
                )
        except Exception as exc:
            skipped.append(f"{test_file}: cache build failed ({exc})")
            continue

        if not score_cache:
            skipped.append(f"{test_file}: empty score cache")
            continue

        try:
            best_lambda, best_threshold, best_val_score = tune_qubo_params_from_cache(
                score_cache, solver, lambda_list, threshold_list, alpha=TUNE_ALPHA,
            )
        except Exception as exc:
            skipped.append(f"{test_file}: QUBO tuning failed ({exc})")
            continue

        x_train = np.concatenate([features[f] for f in train_files])
        y_train = np.concatenate([labels[f] for f in train_files]).astype(int)
        x_test = features[test_file]
        y_test = np.asarray(labels[test_file]).astype(int)

        if len(np.unique(y_train)) < 2:
            skipped.append(f"{test_file}: single class in training labels")
            continue

        try:
            scores = np.asarray(predict_scores(baseline, x_train, y_train, x_test))
            y_baseline = (scores >= BASELINE_THRESHOLD).astype(int)
            y_qubo = safe_solver_call(solver, scores, best_lambda, best_threshold)
        except Exception as exc:
            skipped.append(f"{test_file}: inference failed ({exc})")
            continue

        has_seizure = bool(y_test.sum() > 0)
        baseline_f1 = f1_score(y_test, y_baseline, zero_division=0)
        qubo_f1 = f1_score(y_test, y_qubo, zero_division=0)

        rows.append({
            "file": test_file,
            "has_seizure": has_seizure,
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
            "baseline_fp_rate": float(np.mean(y_baseline)),
            "qubo_fp_rate": float(np.mean(y_qubo)),
            "epochs": int(len(y_test)),
            "seizure_epochs": int(y_test.sum()),
        })

        detail_cache[test_file] = {
            "file_name": test_file,
            "has_seizure": has_seizure,
            "y_true": y_test,
            "y_baseline": y_baseline,
            "y_qubo": y_qubo,
            "scores": scores,
            "best_lambda": best_lambda,
            "best_threshold": best_threshold,
        }

        log_step(
            f"[File] done {test_file}, seizure={has_seizure}, "
            f"baseline_f1={baseline_f1:.4f}, qubo_f1={qubo_f1:.4f}, "
            f"Δ={qubo_f1 - baseline_f1:.4f}, elapsed={time.perf_counter() - file_start:.2f}s"
        )

    # --- Aggregate & Report ---
    if not rows:
        note_text = "\n".join(notes + skipped) or "No valid result"
        return f"Run failed\n\n{note_text}", pd.DataFrame(), None, None, ""

    result_df = (
        pd.DataFrame(rows)
        .sort_values(["has_seizure", "improvement"], ascending=[False, False])
        .reset_index(drop=True)
    )

    seizure_df = result_df[result_df["has_seizure"]]
    nonseizure_df = result_df[~result_df["has_seizure"]]

    summary_fig = build_summary_plot(result_df)

    # Prefer seizure file for detail view
    if len(seizure_df) > 0:
        top_file = seizure_df.sort_values("improvement", ascending=False).iloc[0]["file"]
    else:
        top_file = result_df.iloc[0]["file"]
    detail_fig = build_detail_plot(detail_cache[top_file], baseline, solver_name)

    progress(0.96, desc="Saving results")

    # --- Save pkl ---
    meta = {
        "timestamp": datetime.now().isoformat(),
        "subjects": list(selected_subjects),
        "baseline": baseline,
        "solver_name": solver_name,
        "tune_mode": tune_mode,
        "tune_n_splits": tune_n_splits,
        "max_files_per_subject": max_files_per_subject,
        "n_jobs": n_jobs,
        "lambda_list": lambda_list,
        "threshold_list": threshold_list,
        "reuse_global_cache": reuse_global_cache,
        "tune_alpha": TUNE_ALPHA,
        "baseline_threshold": BASELINE_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "notes": notes,
        "skipped": skipped,
        "total_elapsed_sec": time.perf_counter() - run_start,
    }

    saved_path = ""
    if save_pkl:
        try:
            saved_path = save_results_pkl(result_df, detail_cache, meta)
        except Exception as exc:
            log_step(f"[Save] failed: {exc}")
            saved_path = f"(save failed: {exc})"

    progress(1.0, desc="Done")

    # --- Summary text ---
    summary_text = (
        f"Finished {len(result_df)} files "
        f"(seizure={len(seizure_df)}, non-seizure={len(nonseizure_df)})\n"
        f"Subjects: {', '.join(selected_subjects)}\n"
        f"Baseline={baseline}, Solver={solver_name}, "
        f"TuningMode={tune_mode}, Nfold={tune_n_splits}, ReuseCache={reuse_global_cache}\n"
        f"λ grid={lambda_list}, θ grid={threshold_list}"
    )

    summary_text += "\n\n[Seizure files]"
    if len(seizure_df) > 0:
        summary_text += (
            f"\n  Mean baseline F1 = {seizure_df['baseline_f1'].mean():.4f}"
            f"\n  Mean QUBO F1     = {seizure_df['qubo_f1'].mean():.4f}"
            f"\n  Mean Δ F1        = {seizure_df['improvement'].mean():.4f}"
            f"\n  Mean baseline Recall = {seizure_df['baseline_recall'].mean():.4f}"
            f"\n  Mean QUBO Recall     = {seizure_df['qubo_recall'].mean():.4f}"
        )
    else:
        summary_text += "\n  (none)"

    summary_text += "\n\n[Non-seizure files]"
    if len(nonseizure_df) > 0:
        summary_text += (
            f"\n  Mean baseline FP rate = {nonseizure_df['baseline_fp_rate'].mean():.4f}"
            f"\n  Mean QUBO FP rate     = {nonseizure_df['qubo_fp_rate'].mean():.4f}"
        )
    else:
        summary_text += "\n  (none)"

    note_lines = []
    if notes:
        note_lines.append("Notes:")
        note_lines.extend(f"- {n}" for n in notes)
    if skipped:
        note_lines.append("Skipped:")
        note_lines.extend(f"- {s}" for s in skipped[:10])
        if len(skipped) > 10:
            note_lines.append(f"- ... and {len(skipped) - 10} more")
    if note_lines:
        summary_text += "\n\n" + "\n".join(note_lines)

    log_step(
        f"[Run] done, evaluated={len(result_df)}, skipped={len(skipped)}, "
        f"total={time.perf_counter() - run_start:.2f}s"
    )

    return summary_text, result_df, summary_fig, detail_fig, saved_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    subjects = discover_subjects()

    with gr.Blocks(title="QUBO Seizure UI") as demo:
        gr.Markdown("# QUBO Seizure Experiment Dashboard")
        gr.Markdown(
            "Leave-one-file-out evaluation with inner-validation QUBO tuning. "
            "Metrics are reported separately for seizure vs non-seizure files."
        )

        if not subjects:
            gr.Markdown(
                "⚠️ **No subjects found under `DESTINATION/`.** "
                "Please check that the directory exists and contains `chbXX` folders."
            )

        with gr.Row():
            selected_subjects = gr.CheckboxGroup(
                choices=subjects,
                value=subjects[:1] if subjects else [],
                label="Subjects (multi-select)",
            )

        with gr.Row():
            baseline = gr.Radio(choices=["svm", "xgboost"], value="svm", label="Baseline")
            solver_name = gr.Radio(
                choices=["solve_qubo_seizure", "solve_chain_qubo_exact"],
                value="solve_chain_qubo_exact",
                label="QUBO Solver",
            )

        with gr.Row():
            tune_mode = gr.Radio(
                choices=["lofo", "nfold"], value="nfold", label="Tuning Strategy"
            )
            tune_n_splits = gr.Slider(
                2, 10, value=5, step=1,
                label="Nfold Splits (used only when tune_mode=nfold)",
            )

        with gr.Row():
            max_files_per_subject = gr.Slider(
                0, 30, value=5, step=1, label="Max EDF files per subject (0=all)"
            )
            n_jobs = gr.Slider(
                -1, 16, value=-1, step=1,
                label="Preprocess parallel jobs (0 auto-adjusted to 1)",
            )

        with gr.Row():
            lambda_text = gr.Textbox(
                value=", ".join(str(x) for x in DEFAULT_LAMBDA_LIST),
                label="Lambda grid (comma-separated)",
            )
            threshold_text = gr.Textbox(
                value=", ".join(str(x) for x in DEFAULT_THRESHOLD_LIST),
                label="Threshold grid (comma-separated)",
            )

        with gr.Row():
            reuse_global_cache = gr.Checkbox(
                value=True,
                label="Reuse global validation cache (faster; slight leakage in nfold mode)",
            )
            save_pkl = gr.Checkbox(
                value=True,
                label="Save results as .pkl into ./results/",
            )

        run_button = gr.Button("Run Experiment", variant="primary")

        summary_output = gr.Textbox(label="Run Summary", lines=14)
        result_table = gr.Dataframe(label="Per-file Metrics")
        summary_plot = gr.Plot(label="Overall Visualization")
        detail_plot = gr.Plot(label="Top-Improvement Seizure File Detail")
        saved_path_output = gr.Textbox(label="Saved .pkl Path", lines=1)

        run_button.click(
            fn=run_experiment,
            inputs=[
                selected_subjects, baseline, solver_name,
                tune_mode, tune_n_splits,
                max_files_per_subject, n_jobs,
                lambda_text, threshold_text,
                reuse_global_cache, save_pkl,
            ],
            outputs=[
                summary_output, result_table,
                summary_plot, detail_plot,
                saved_path_output,
            ],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
