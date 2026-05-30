"""
Master experiment runner.
Runs all experiments in the correct order for the paper.

Priority order (as discussed):
1. Baseline (no training)
2. CE Reranker only (frozen generator)
3. GRPO soft-gate only (baseline retriever)
4. GRPO no-faith ablation
5. GRPO hard-gate ablation
6. Stable Combined: CE reranker + GRPO soft-gate (your main system)

Runs each experiment for all 3 seeds.

Usage:
    python run_all_experiments.py --seeds 42 123 456
    python run_all_experiments.py --seeds 42          # single seed for quick test
    python run_all_experiments.py --skip_baseline     # if baseline already done
"""
import os
import sys
import json
import argparse
import subprocess
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import RUNS_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--val_size", type=int, default=1000)
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--grpo_steps", type=int, default=3000)
    p.add_argument("--reranker_steps", type=int, default=2000)
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running them")
    return p.parse_args()


def run_cmd(cmd: str, dry_run: bool = False):
    print(f"\n{'='*60}")
    print(f"RUNNING: {cmd}")
    print(f"{'='*60}")
    if not dry_run:
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"ERROR: Command failed with code {result.returncode}")
            return False
    return True


def find_latest_run(run_type: str, seed: int) -> str:
    """Find the most recent run directory for a given type and seed."""
    base = os.path.join(RUNS_DIR, run_type)
    if not os.path.exists(base):
        return None
    runs = [d for d in os.listdir(base) if f"seed{seed}" in d]
    if not runs:
        return None
    runs.sort()
    return os.path.join(base, runs[-1])


def find_reranker_w(seed: int) -> str:
    """Find the reranker weight file for a given seed."""
    run_dir = find_latest_run("reranker_ce", seed)
    if run_dir is None:
        return None
    w_path = os.path.join(run_dir, "w.pt")
    return w_path if os.path.exists(w_path) else None


def find_grpo_ckpt(gate: str, seed: int) -> str:
    """Find the final GRPO checkpoint for a given gate type and seed."""
    run_dir = find_latest_run("grpo", seed)
    if run_dir is None:
        return None
    final = os.path.join(run_dir, "checkpoints", "final")
    return final if os.path.exists(final) else None


def main():
    args = parse_args()
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(scripts_dir, "src")

    common = (
        f"--train_size {args.train_size} "
        f"--val_size {args.val_size}"
    )

    summary = []

    # ── Step 1: Baseline (once, no seeds needed) ──────────────────────────────
    if not args.skip_baseline:
        print("\n\n" + "="*60)
        print("EXPERIMENT 1: BASELINE")
        print("="*60)
        cmd = f"cd {scripts_dir} && PYTHONPATH={src_dir} python scripts/eval_baseline.py {common}"
        run_cmd(cmd, args.dry_run)
        summary.append("baseline: done")

    # ── Step 2: CE Reranker per seed ──────────────────────────────────────────
    reranker_paths = {}
    print("\n\n" + "="*60)
    print("EXPERIMENT 2: CE RERANKER")
    print("="*60)
    for seed in args.seeds:
        cmd = (
            f"cd {scripts_dir} && PYTHONPATH={src_dir} "
            f"python scripts/train_reranker.py {common} "
            f"--steps {args.reranker_steps} --seed {seed}"
        )
        run_cmd(cmd, args.dry_run)
        if not args.dry_run:
            w_path = find_reranker_w(seed)
            reranker_paths[seed] = w_path
            summary.append(f"reranker seed={seed}: {w_path}")

    # ── Step 3: GRPO soft-gate only (baseline retriever) ─────────────────────
    print("\n\n" + "="*60)
    print("EXPERIMENT 3: GRPO SOFT-GATE (fixed retriever)")
    print("="*60)
    for seed in args.seeds:
        cmd = (
            f"cd {scripts_dir} && PYTHONPATH={src_dir} "
            f"python scripts/train_grpo.py {common} "
            f"--gate soft --seed {seed} --steps {args.grpo_steps}"
        )
        run_cmd(cmd, args.dry_run)
        summary.append(f"grpo_softgate seed={seed}: done")

    # ── Step 4: GRPO no-faith ablation ───────────────────────────────────────
    print("\n\n" + "="*60)
    print("EXPERIMENT 4: GRPO NO-FAITH ABLATION")
    print("="*60)
    for seed in args.seeds:
        cmd = (
            f"cd {scripts_dir} && PYTHONPATH={src_dir} "
            f"python scripts/train_grpo.py {common} "
            f"--gate none --seed {seed} --steps {args.grpo_steps}"
        )
        run_cmd(cmd, args.dry_run)
        summary.append(f"grpo_nofaith seed={seed}: done")

    # ── Step 5: GRPO hard-gate ablation ──────────────────────────────────────
    print("\n\n" + "="*60)
    print("EXPERIMENT 5: GRPO HARD-GATE ABLATION")
    print("="*60)
    for seed in args.seeds:
        cmd = (
            f"cd {scripts_dir} && PYTHONPATH={src_dir} "
            f"python scripts/train_grpo.py {common} "
            f"--gate hard --seed {seed} --steps {args.grpo_steps}"
        )
        run_cmd(cmd, args.dry_run)
        summary.append(f"grpo_hardgate seed={seed}: done")

    # ── Step 6: Stable Combined (CE reranker + GRPO soft-gate) ───────────────
    print("\n\n" + "="*60)
    print("EXPERIMENT 6: STABLE COMBINED (main system)")
    print("="*60)
    for seed in args.seeds:
        reranker_path = reranker_paths.get(seed)
        if reranker_path is None and not args.dry_run:
            print(f"WARNING: No reranker found for seed={seed}, skipping combined.")
            continue

        reranker_arg = f"--reranker_path {reranker_path}" if reranker_path else ""
        cmd = (
            f"cd {scripts_dir} && PYTHONPATH={src_dir} "
            f"python scripts/train_grpo.py {common} "
            f"--gate soft --seed {seed} --steps {args.grpo_steps} "
            f"{reranker_arg}"
        )
        run_cmd(cmd, args.dry_run)
        summary.append(f"stable_combined seed={seed}: done")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    for s in summary:
        print(f"  ✓ {s}")
    print(f"\nResults in: {RUNS_DIR}")
    print("Run: python scripts/aggregate_results.py to build summary table")


if __name__ == "__main__":
    main()
