"""
Aggregate results across all experiments and seeds.
Produces:
  - summary_table.csv  (mean ± std across seeds)
  - answer_quality.png (EM/F1 bar chart)
  - retrieval_metrics.png (recall line chart)
  - training_curves.png (reward over steps per experiment)
  - ctx_utilization.png (correct vs incorrect predictions)

Usage:
    python scripts/aggregate_results.py
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config import RUNS_DIR


# ── Map run folder names to paper-friendly labels ─────────────────────────────

RUN_TYPE_LABELS = {
    "baseline":         "Baseline",
    "reranker_ce":      "CE Reranker + Frozen Gen",
    "grpo_softgate":    "GRPO Soft-Gate (fixed retriever)",
    "grpo_nofaith":     "GRPO No-Faith (ablation)",
    "grpo_hardgate":    "GRPO Hard-Gate (ablation)",
    "grpo_soft":        "Stable Combined (CE + Soft-Gate GRPO)",
}

PAPER_ORDER = [
    "Baseline",
    "CE Reranker + Frozen Gen",
    "GRPO Soft-Gate (fixed retriever)",
    "GRPO No-Faith (ablation)",
    "GRPO Hard-Gate (ablation)",
    "Stable Combined (CE + Soft-Gate GRPO)",
]


def load_metrics_files(runs_dir: str) -> list:
    """Load all metrics.json files from run directories."""
    records = []
    for metrics_path in glob.glob(os.path.join(runs_dir, "**", "metrics.json"), recursive=True):
        with open(metrics_path) as f:
            m = json.load(f)
        m["_path"] = metrics_path
        m["_run_dir"] = os.path.dirname(metrics_path)
        records.append(m)
    return records


def infer_experiment_type(run_name: str) -> str:
    """Infer experiment type from run directory name."""
    rn = run_name.lower()
    if "baseline" in rn and "reranker" not in rn:
        return "Baseline"
    if "reranker_ce" in rn or ("reranker" in rn and "grpo" not in rn):
        return "CE Reranker + Frozen Gen"
    if "grpo" in rn and "nofaith" in rn:
        return "GRPO No-Faith (ablation)"
    if "grpo" in rn and "hardgate" in rn:
        return "GRPO Hard-Gate (ablation)"
    if "grpo" in rn and "softgate" in rn and "reranker" not in rn:
        return "GRPO Soft-Gate (fixed retriever)"
    if "grpo" in rn and "soft" in rn and "reranker" in rn:
        return "Stable Combined (CE + Soft-Gate GRPO)"
    return "Other"


def build_summary_table(records: list) -> pd.DataFrame:
    """Build mean ± std table across seeds for each experiment type."""
    from collections import defaultdict
    grouped = defaultdict(list)

    for r in records:
        run_name = r.get("run_name", os.path.basename(r["_run_dir"]))
        exp_type = infer_experiment_type(run_name)
        if exp_type == "Other":
            continue
        grouped[exp_type].append(r)

    rows = []
    for exp_type in PAPER_ORDER:
        if exp_type not in grouped:
            continue
        recs = grouped[exp_type]
        row = {"System": exp_type}
        for metric in ["em", "f1", "title_recall", "sentence_recall", "hit_at_k"]:
            vals = [r[metric] for r in recs if metric in r]
            if vals:
                row[metric + "_mean"] = np.mean(vals)
                row[metric + "_std"] = np.std(vals)
                row[metric + "_str"] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
            else:
                row[metric + "_mean"] = None
                row[metric + "_std"] = None
                row[metric + "_str"] = "N/A"

        for metric in ["ctx_util_correct", "ctx_util_incorrect"]:
            vals = [r[metric] for r in recs if metric in r]
            if vals:
                row[metric + "_mean"] = np.mean(vals)
                row[metric + "_str"] = f"{np.mean(vals):.4f}"

        rows.append(row)

    return pd.DataFrame(rows)


def plot_answer_quality(df: pd.DataFrame, out_dir: str):
    """Bar chart: EM and F1 per system."""
    systems = df["System"].tolist()
    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_em = ax.bar(x - width/2, df["em_mean"], width, label="EM",
                     yerr=df["em_std"], capsize=4, color="#2196F3")
    bars_f1 = ax.bar(x + width/2, df["f1_mean"], width, label="F1",
                     yerr=df["f1_std"], capsize=4, color="#FF9800")

    ax.set_xlabel("System")
    ax.set_ylabel("Score")
    ax.set_title("Answer Quality Across Systems (EM, F1)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" (", "\n(") for s in systems], rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, max(df["f1_mean"].max() * 1.2, 0.5))
    plt.tight_layout()
    path = os.path.join(out_dir, "answer_quality.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_retrieval_metrics(df: pd.DataFrame, out_dir: str):
    """Line chart: retrieval metrics per system."""
    fig, ax = plt.subplots(figsize=(14, 6))
    systems = df["System"].tolist()
    x = np.arange(len(systems))

    for metric, label, color in [
        ("title_recall_mean", "Title Recall", "#1976D2"),
        ("sentence_recall_mean", "Sentence Recall", "#FF5722"),
        ("hit_at_k_mean", "Hit@3", "#4CAF50"),
    ]:
        if metric in df.columns:
            ax.plot(x, df[metric], marker="o", label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" (", "\n(") for s in systems], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Metrics Across Systems")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "retrieval_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_ctx_utilization(df: pd.DataFrame, out_dir: str):
    """Bar chart showing context utilization for correct vs incorrect preds."""
    has_ctx = "ctx_util_correct_mean" in df.columns and df["ctx_util_correct_mean"].notna().any()
    if not has_ctx:
        print("No context utilization data found, skipping plot.")
        return

    systems = df["System"].tolist()
    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    correct_vals = df["ctx_util_correct_mean"].fillna(0).tolist()
    incorrect_vals = df.get("ctx_util_incorrect_mean", pd.Series([0]*len(df))).fillna(0).tolist()

    ax.bar(x - width/2, correct_vals, width, label="Correct predictions", color="#4CAF50")
    ax.bar(x + width/2, incorrect_vals, width, label="Incorrect predictions", color="#F44336")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" (", "\n(") for s in systems], rotation=15, ha="right")
    ax.set_ylabel("Context Utilization")
    ax.set_title("Context Utilization: Correct vs Incorrect Predictions\n(Higher gap = soft-gate is working)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "ctx_utilization.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_training_curves(runs_dir: str, out_dir: str):
    """Plot reward curves from train_log.jsonl files."""
    log_files = glob.glob(os.path.join(runs_dir, "grpo", "**", "train_log.jsonl"), recursive=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for log_path in log_files:
        run_name = os.path.basename(os.path.dirname(log_path))
        exp_type = infer_experiment_type(run_name)
        if exp_type == "Other":
            continue

        steps, rewards, r_ans = [], [], []
        try:
            with open(log_path) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    steps.append(rec["step"])
                    rewards.append(rec.get("reward_mean", 0))
                    r_ans.append(rec.get("r_ans_mean", 0))
        except Exception:
            continue

        if not steps:
            continue

        label = exp_type[:30]
        axes[0].plot(steps, rewards, label=label, alpha=0.8)
        axes[1].plot(steps, r_ans, label=label, alpha=0.8)

    axes[0].set_title("Total Reward During Training")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean Reward (group)")
    axes[0].legend(fontsize=7)

    axes[1].set_title("Answer Reward (r_ans) During Training")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean F1 (group)")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    out_dir = os.path.join(RUNS_DIR, "summary")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading all metrics files...")
    records = load_metrics_files(RUNS_DIR)
    print(f"Found {len(records)} metrics files")

    if not records:
        print("No results found. Run experiments first.")
        return

    print("\nBuilding summary table...")
    df = build_summary_table(records)

    if df.empty:
        print("No recognized experiment types found.")
        return

    # Print table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    display_cols = ["System"] + [c for c in df.columns if c.endswith("_str")]
    print(df[display_cols].to_string(index=False))

    # Save CSV
    csv_path = os.path.join(out_dir, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Plots
    print("\nGenerating plots...")
    plot_answer_quality(df, out_dir)
    plot_retrieval_metrics(df, out_dir)
    plot_ctx_utilization(df, out_dir)
    plot_training_curves(RUNS_DIR, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
