"""
Baseline evaluation: vanilla RAG with no RL training.
Run this first to establish your baseline numbers.

Usage:
    python eval_baseline.py
    python eval_baseline.py --with_reranker --reranker_path runs/reranker_ce/.../w.pt
"""
import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config import RUNS_DIR, GEN_MODEL, TOP_K
from data import setup_data
from evaluate import evaluate
import datetime
import uuid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=2000)
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--model", type=str, default=GEN_MODEL)
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--with_reranker", action="store_true")
    p.add_argument("--reranker_path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Setting up data...")
    data = setup_data(device, args.train_size, args.val_size)

    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    reranker_w = None
    run_name = "baseline"

    if args.with_reranker and args.reranker_path:
        rer = torch.load(args.reranker_path, map_location="cpu")
        reranker_w = rer["w"].to(device)
        run_name = "baseline_with_reranker"
        print(f"Using reranker: {args.reranker_path}")

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RUNS_DIR, "baseline", f"{run_name}_{ts}_{uuid.uuid4().hex[:6]}")

    metrics = evaluate(
        model=model,
        tokenizer=tok,
        val_data_dict=data,
        run_dir=run_dir,
        run_name=run_name,
        reranker_w=reranker_w,
        device=device,
    )

    print(f"\nBaseline results saved to: {run_dir}")
    return metrics


if __name__ == "__main__":
    main()
