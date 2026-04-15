"""
Baseline evaluation: vanilla RAG with Mistral-7B-Instruct, no RL training.
Supports 4-GPU distributed inference via torchrun.

Usage (single GPU):
    python eval_baseline.py

Usage (all 4 GPUs):
    torchrun --nproc_per_node=4 scripts/eval_baseline.py

Usage (with reranker):
    torchrun --nproc_per_node=4 scripts/eval_baseline.py \
        --with_reranker --reranker_path runs/reranker_ce/.../w.pt
"""
import os
import sys
import json
import argparse
import datetime
import uuid
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config import RUNS_DIR, GEN_MODEL, TOP_K, MAX_NEW_TOKENS, MAX_INPUT_LENGTH
from data import setup_data
from utils import (
    em_score, f1_score, title_recall, sentence_recall,
    hit_at_k, context_utilization, build_prompt, aggregate_metrics
)
from evaluate import retrieve_topk
import datetime
import uuid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--val_size",   type=int, default=1000)
    p.add_argument("--model",      type=str, default=GEN_MODEL)
    p.add_argument("--top_k",      type=int, default=TOP_K)
    p.add_argument("--with_reranker", action="store_true")
    p.add_argument("--reranker_path", type=str, default=None)
    p.add_argument("--dataset",    type=str, default="hotpotqa",
                   choices=["hotpotqa", "2wikimultihop", "musique"])
    return p.parse_args()


def main():
    args = parse_args()

    # ── Distributed setup ─────────────────────────────────────────────────────
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1

    device  = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    is_main = local_rank == 0

    if is_main:
        print(f"Running baseline eval | model={args.model} | dataset={args.dataset}")
        print(f"GPUs: {world_size} | val_size={args.val_size}")

    # ── Data setup (all ranks, embeddings are cached) ────────────────────────
    if is_main:
        print("\nSetting up data...")
    data = setup_data(device, args.train_size, args.val_size, dataset_name=args.dataset)

    # ── Load model on each rank's GPU ─────────────────────────────────────────
    if is_main:
        print(f"\nLoading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_w = None
    if args.with_reranker and args.reranker_path:
        rer = torch.load(args.reranker_path, map_location="cpu")
        reranker_w = rer["w"].to(device)
        if is_main:
            print(f"Using reranker: {args.reranker_path}")

    # ── Split val examples across ranks ──────────────────────────────────────
    all_ex_ids = data["val_ex_ids"]
    # Each rank takes every world_size-th example: rank 0 → [0,4,8...], rank 1 → [1,5,9...] etc.
    rank_ex_ids = all_ex_ids[local_rank::world_size]
    if is_main:
        print(f"\nVal examples: {len(all_ex_ids)} total | {len(rank_ex_ids)} per GPU")

    # ── Inference on this rank's slice ────────────────────────────────────────
    local_predictions = []

    for ex_id in rank_ex_ids:
        chosen_rows = retrieve_topk(
            ex_id,
            data["val_group_emb"],
            data["val_group_rows"],
            data["val_q_vecs"],
            args.top_k,
            reranker_w,
        )
        retrieved_titles = [r["title"] for r in chosen_rows]
        prompt = build_prompt(data["questions_val"][ex_id], chosen_rows)

        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(device)

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        pred = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()

        gold = data["gold_answers_val"][ex_id]
        gm   = data["gold_support_val"][ex_id]

        local_predictions.append({
            "id":               ex_id,
            "question":         data["questions_val"][ex_id],
            "gold":             gold,
            "pred":             pred,
            "em":               em_score(pred, gold),
            "f1":               f1_score(pred, gold),
            "ctx_util":         context_utilization(pred, chosen_rows),
            "title_recall":     title_recall(retrieved_titles, gm),
            "sent_recall":      sentence_recall(retrieved_titles, gm),
            "hit_at_k":         hit_at_k(retrieved_titles, gm),
            "retrieved_titles": retrieved_titles,
        })

    # ── Gather all predictions to rank 0 ─────────────────────────────────────
    if is_distributed:
        all_preds_gathered = [None] * world_size
        dist.all_gather_object(all_preds_gathered, local_predictions)
        all_predictions = [p for rank_preds in all_preds_gathered for p in rank_preds]
    else:
        all_predictions = local_predictions

    # ── Aggregate metrics and save (main rank only) ───────────────────────────
    if is_main:
        ems          = [p["em"]           for p in all_predictions]
        f1s          = [p["f1"]           for p in all_predictions]
        t_recalls    = [p["title_recall"] for p in all_predictions]
        s_recalls    = [p["sent_recall"]  for p in all_predictions]
        hits         = [p["hit_at_k"]     for p in all_predictions]
        ctx_correct  = [p["ctx_util"] for p in all_predictions if p["em"] > 0]
        ctx_incorrect= [p["ctx_util"] for p in all_predictions if p["em"] == 0]

        metrics = aggregate_metrics(ems, f1s, t_recalls, s_recalls, hits,
                                    ctx_correct, ctx_incorrect)
        metrics["n_examples"] = len(all_predictions)
        metrics["model"]      = args.model
        metrics["dataset"]    = args.dataset

        run_name = f"baseline_{args.dataset}"
        ts       = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir  = os.path.join(RUNS_DIR, "baseline",
                                f"{run_name}_{ts}_{uuid.uuid4().hex[:6]}")
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(run_dir, "predictions.jsonl"), "w") as f:
            for p in all_predictions:
                f.write(json.dumps(p) + "\n")

        print(f"\n{'='*50}")
        print(f"BASELINE — {args.dataset.upper()} ({args.model})")
        print(f"  EM:               {metrics['em']:.4f}")
        print(f"  F1:               {metrics['f1']:.4f}")
        print(f"  Title Recall:     {metrics['title_recall']:.4f}")
        print(f"  Sentence Recall:  {metrics['sentence_recall']:.4f}")
        print(f"  Hit@{args.top_k}:          {metrics['hit_at_k']:.4f}")
        if "ctx_util_correct" in metrics:
            print(f"  CtxUtil(correct):   {metrics['ctx_util_correct']:.4f}")
            print(f"  CtxUtil(incorrect): {metrics['ctx_util_incorrect']:.4f}")
        print(f"{'='*50}")
        print(f"\nResults saved to: {run_dir}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
