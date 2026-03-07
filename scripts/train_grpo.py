"""
GRPO training for the RAG generator.
Supports all ablation variants via --gate flag.

Usage:
    # Your best system (soft gate)
    python train_grpo.py --gate soft --seed 42

    # Ablation: no faith reward
    python train_grpo.py --gate none --seed 42

    # Ablation: hard gate
    python train_grpo.py --gate hard --seed 42

    # With CE reranker providing retrieval
    python train_grpo.py --gate soft --reranker_path runs/reranker_ce/.../w.pt --seed 42
"""
import os
import sys
import json
import argparse
import datetime
import uuid
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config import (
    RUNS_DIR, GEN_MODEL, TOP_K,
    GRPO_STEPS, GRPO_GROUP_SIZE, GRPO_LR,
    GRPO_LOG_EVERY, GRPO_CKPT_EVERY,
    GRPO_TEMPERATURE, GRPO_TOP_P,
    W_ANS, W_FAITH, LAMBDA_COST,
    MAX_NEW_TOKENS, MAX_INPUT_LENGTH,
)
from data import setup_data, build_train_pack
from utils import build_prompt, compute_reward
from evaluate import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gate", type=str, default="soft",
                   choices=["soft", "hard", "none"],
                   help="Faithfulness gate type. 'soft'=your contribution, 'none'=ablation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=GRPO_STEPS)
    p.add_argument("--group_size", type=int, default=GRPO_GROUP_SIZE)
    p.add_argument("--lr", type=float, default=GRPO_LR)
    p.add_argument("--w_ans", type=float, default=W_ANS)
    p.add_argument("--w_faith", type=float, default=W_FAITH)
    p.add_argument("--lambda_cost", type=float, default=LAMBDA_COST)
    p.add_argument("--train_size", type=int, default=2000)
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--model", type=str, default=GEN_MODEL)
    p.add_argument("--reranker_path", type=str, default=None,
                   help="Path to reranker w.pt. If set, uses CE reranker for retrieval.")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint dir to resume from.")
    return p.parse_args()


def seq2seq_logprob(model, input_ids, attention_mask, output_ids):
    """Compute total log probability of output sequence."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=output_ids
    )
    T = output_ids.shape[1]
    return -out.loss * T


def retrieve_for_training(
    ex_id: str,
    data: dict,
    reranker_w: torch.Tensor,
    top_k: int,
    device: str,
) -> list:
    """Retrieve paragraphs using reranker or baseline cosine sim."""
    P = data["train_group_emb"][ex_id]
    qv = data["train_q_vecs"][ex_id]
    rows = data["train_group_rows"][ex_id]

    if reranker_w is not None:
        feats = P * qv.unsqueeze(0)
        scores = (feats @ reranker_w).detach()
    else:
        scores = (P @ qv).detach()

    topk_idx = torch.topk(scores, k=min(top_k, scores.numel())).indices.tolist()
    return [rows[i] for i in topk_idx]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: gate={args.gate} seed={args.seed} steps={args.steps} model={args.model}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ── Setup data ────────────────────────────────────────────────────────────
    print("\nSetting up data...")
    data = setup_data(device, args.train_size, args.val_size)

    # ── Load reranker if provided ─────────────────────────────────────────────
    reranker_w = None
    if args.reranker_path:
        rer = torch.load(args.reranker_path, map_location="cpu")
        reranker_w = rer["w"].to(device)
        print(f"Loaded reranker: {args.reranker_path}")

    # ── Build train pack ──────────────────────────────────────────────────────
    print("\nBuilding train pack...")
    train_pack = build_train_pack(
        data["train_data"],
        data["train_rows"],
        data["train_by_ex"],
        data["train_para_emb"],
        data["train_ex_ids"],
        top_k=args.top_k,
    )
    print(f"Train pack: {len(train_pack)} examples")

    # ── Setup run directory ───────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"grpo_{args.gate}gate_seed{args.seed}_{ts}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(RUNS_DIR, "grpo", run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.jsonl")

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saving to: {run_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        tok = AutoTokenizer.from_pretrained(args.resume_from)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.resume_from).to(device)
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting GRPO training | {args.steps} steps | group_size={args.group_size}")

    with open(log_path, "w") as logf:
        for step in range(1, args.steps + 1):
            ex = train_pack[rng.integers(0, len(train_pack))]
            ex_id = ex["id"]
            gold = ex["gold"]

            # Use reranker-based retrieval if available, else precomputed
            if reranker_w is not None:
                chosen_rows = retrieve_for_training(
                    ex_id, data, reranker_w, args.top_k, device
                )
                prompt = build_prompt(ex["question"], chosen_rows)
                retrieved_titles = [r["title"] for r in chosen_rows]
            else:
                prompt = ex["prompt"]
                retrieved_titles = ex["retrieved_titles"]

            gm = data["gold_support_train"][ex_id]

            enc = tok(
                prompt, return_tensors="pt",
                truncation=True, max_length=MAX_INPUT_LENGTH
            ).to(device)
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]

            # Generate group of samples
            preds, out_ids_list = [], []
            for _ in range(args.group_size):
                with torch.no_grad():
                    gen_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=GRPO_TEMPERATURE,
                        top_p=GRPO_TOP_P,
                    )
                pred = tok.decode(gen_ids[0], skip_special_tokens=True).strip()
                preds.append(pred)
                out_ids_list.append(gen_ids)

            # Compute rewards
            reward_dicts = [
                compute_reward(
                    pred=p,
                    gold=gold,
                    retrieved_titles=retrieved_titles,
                    gold_map=gm,
                    tokenizer=tok,
                    max_new_tokens=MAX_NEW_TOKENS,
                    w_ans=args.w_ans,
                    w_faith=args.w_faith,
                    lambda_cost=args.lambda_cost,
                    gate=args.gate,
                )
                for p in preds
            ]

            rewards = [d["R"] for d in reward_dicts]
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

            # GRPO advantage = reward - mean(group rewards)
            adv = rewards_t - rewards_t.mean()

            # Compute log probs and loss
            logps = [
                seq2seq_logprob(model, input_ids, attn, out_ids)
                for out_ids in out_ids_list
            ]
            logps_t = torch.stack(logps)
            loss = -(adv.detach() * logps_t).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Logging
            if step % args.steps // 100 == 0 or step % GRPO_LOG_EVERY == 0:
                rec = {
                    "step": step,
                    "reward_mean": float(rewards_t.mean()),
                    "reward_std": float(rewards_t.std()),
                    "reward_max": float(rewards_t.max()),
                    "r_ans_mean": float(np.mean([d["r_ans"] for d in reward_dicts])),
                    "r_faith_mean": float(np.mean([d["r_faith_base"] for d in reward_dicts])),
                    "r_faith_eff_mean": float(np.mean([d["r_faith_eff"] for d in reward_dicts])),
                    "loss": float(loss.detach()),
                }
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                print(
                    f"step {step:5d} | "
                    f"R={rec['reward_mean']:.4f}±{rec['reward_std']:.4f} | "
                    f"r_ans={rec['r_ans_mean']:.4f} | "
                    f"r_faith={rec['r_faith_eff_mean']:.4f} | "
                    f"loss={rec['loss']:.4f}"
                )

            # Checkpointing
            if step % GRPO_CKPT_EVERY == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step_{step:07d}")
                os.makedirs(ckpt_path, exist_ok=True)
                model.save_pretrained(ckpt_path)
                tok.save_pretrained(ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

    # Save final model
    final_ckpt = os.path.join(ckpt_dir, "final")
    os.makedirs(final_ckpt, exist_ok=True)
    model.save_pretrained(final_ckpt)
    tok.save_pretrained(final_ckpt)
    print(f"\nFinal model saved: {final_ckpt}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    metrics = evaluate(
        model=model,
        tokenizer=tok,
        val_data_dict=data,
        run_dir=run_dir,
        run_name=run_name,
        reranker_w=reranker_w,
        device=device,
    )

    print(f"\nDone. Results: {run_dir}")
    return run_dir, metrics


if __name__ == "__main__":
    main()
