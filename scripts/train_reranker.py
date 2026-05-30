"""
Train the CE (cross-entropy) reranker.
Supervised: maximizes probability mass on gold paragraphs.
This is your stable retriever used in Stable Combined system.

Usage:
    python train_reranker.py --steps 2000 --lr 0.01 --seed 42
"""
import os
import sys
import json
import argparse
import datetime
import uuid
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RUNS_DIR, TOP_K, RERANKER_STEPS, RERANKER_LR
from data import setup_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=RERANKER_STEPS)
    p.add_argument("--lr", type=float, default=RERANKER_LR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_size", type=int, default=2000)
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--top_k", type=int, default=TOP_K)
    return p.parse_args()


def train_reranker(args, data: dict, device: str) -> torch.Tensor:
    """Train CE reranker. Returns learned weight vector w."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_group_emb = data["train_group_emb"]
    train_group_rows = data["train_group_rows"]
    train_q_vecs = data["train_q_vecs"]
    gold_support_train = data["gold_support_train"]

    dim = next(iter(train_group_emb.values())).shape[1]
    w = torch.zeros(dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([w], lr=args.lr)

    train_ex_ids = list(train_group_emb.keys())

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"reranker_ce_seed{args.seed}_{ts}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(RUNS_DIR, "reranker_ce", run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.jsonl")

    print(f"Training CE Reranker | steps={args.steps} lr={args.lr} seed={args.seed}")
    print(f"Saving to: {run_dir}")

    with open(log_path, "w") as logf:
        for step in range(1, args.steps + 1):
            ex_id = rng.choice(train_ex_ids)
            P = train_group_emb[ex_id]       # [n, D]
            rows = train_group_rows[ex_id]
            qv = train_q_vecs[ex_id]         # [D]

            gm = gold_support_train[ex_id]
            gold_titles = set(gm.keys())
            mask = torch.tensor(
                [1 if r["title"] in gold_titles else 0 for r in rows],
                device=device, dtype=torch.float32
            )
            if mask.sum() == 0:
                continue

            feats = P * qv.unsqueeze(0)      # [n, D]
            scores = feats @ w               # [n]
            logp = torch.log_softmax(scores, dim=0)

            loss = -(logp[mask.bool()].mean())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 200 == 0:
                with torch.no_grad():
                    probs = torch.softmax(scores, dim=0)
                    pos_mass = float(probs[mask.bool()].sum())
                rec = {
                    "step": step,
                    "loss": float(loss.item()),
                    "pos_mass": pos_mass
                }
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                print(f"  step {step:5d} | loss={rec['loss']:.4f} | pos_mass={pos_mass:.4f}")

    # Save weights
    w_path = os.path.join(run_dir, "w.pt")
    torch.save({"w": w.detach().cpu(), "dim": dim, "seed": args.seed}, w_path)
    print(f"Saved reranker weights: {w_path}")

    return w.detach(), run_dir


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Setting up data...")
    data = setup_data(device, args.train_size, args.val_size)

    # Train reranker
    w, run_dir = train_reranker(args, data, device)

    # Evaluate retrieval quality
    from evaluate import evaluate
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from config import GEN_MODEL

    print("\nLoading baseline generator for retrieval eval...")
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.bfloat16).to(device)

    metrics = evaluate(
        model=model,
        tokenizer=tok,
        val_data_dict=data,
        run_dir=run_dir,
        run_name=f"reranker_ce_seed{args.seed}",
        reranker_w=w.to(device),
        device=device,
    )

    print(f"Reranker eval complete. Results saved to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
