"""
GRPO training for the RAG generator.
Supports all ablation variants via --gate flag.

Usage (single GPU):
    python train_grpo.py --gate soft --seed 42

Usage (4 GPUs):
    python -m torch.distributed.run --nproc_per_node=4 scripts/train_grpo.py --gate soft --seed 42
"""
import os
import sys
import json
import argparse
import datetime
import uuid
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    p.add_argument("--gate",        type=str,   default="soft",
                   choices=["soft", "hard", "none"])
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--steps",       type=int,   default=GRPO_STEPS)
    p.add_argument("--group_size",  type=int,   default=GRPO_GROUP_SIZE)
    p.add_argument("--lr",          type=float, default=GRPO_LR)
    p.add_argument("--w_ans",       type=float, default=W_ANS)
    p.add_argument("--w_faith",     type=float, default=W_FAITH)
    p.add_argument("--lambda_cost", type=float, default=LAMBDA_COST)
    p.add_argument("--train_size",  type=int,   default=10000)
    p.add_argument("--val_size",    type=int,   default=1000)
    p.add_argument("--top_k",       type=int,   default=TOP_K)
    p.add_argument("--model",       type=str,   default=GEN_MODEL)
    p.add_argument("--dataset",     type=str,   default="hotpotqa",
                   choices=["hotpotqa", "2wikimultihop", "musique"])
    p.add_argument("--reranker_path", type=str, default=None)
    p.add_argument("--resume_from",   type=str, default=None)
    return p.parse_args()


def causal_logprob(model, input_ids, attention_mask, answer_ids):
    full_ids  = torch.cat([input_ids, answer_ids], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(answer_ids)], dim=1)
    labels    = torch.full_like(full_ids, -100)
    labels[:, input_ids.shape[1]:] = answer_ids
    out = model(input_ids=full_ids, attention_mask=full_mask, labels=labels)
    return -out.loss * answer_ids.shape[1]


def retrieve_for_training(ex_id, data, reranker_w, top_k, device):
    P    = data["train_group_emb"][ex_id]
    qv   = data["train_q_vecs"][ex_id]
    rows = data["train_group_rows"][ex_id]
    if reranker_w is not None:
        scores = (P * qv.unsqueeze(0) @ reranker_w).detach()
    else:
        scores = (P @ qv).detach()
    topk_idx = torch.topk(scores, k=min(top_k, scores.numel())).indices.tolist()
    return [rows[i] for i in topk_idx]


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

    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    rng = np.random.default_rng(args.seed + local_rank)

    if is_main:
        print(f"Config: gate={args.gate} seed={args.seed} steps={args.steps} gpus={world_size}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if is_main:
        print("\nSetting up data...")
    data = setup_data(device, args.train_size, args.val_size, dataset_name=args.dataset)

    reranker_w = None
    if args.reranker_path:
        rer = torch.load(args.reranker_path, map_location="cpu")
        reranker_w = rer["w"].to(device)

    if is_main:
        print("\nBuilding train pack...")
    train_pack = build_train_pack(
        data["train_data"], data["train_rows"], data["train_by_ex"],
        data["train_para_emb"], data["train_ex_ids"], top_k=args.top_k,
    )
    if is_main:
        print(f"Train pack: {len(train_pack)} examples")

    # ── Run directory ─────────────────────────────────────────────────────────
    ts       = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"grpo_{args.gate}gate_{args.dataset}_seed{args.seed}_{ts}_{uuid.uuid4().hex[:6]}"
    run_dir  = os.path.join(RUNS_DIR, "grpo", run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Saving to: {run_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    src = args.resume_from if args.resume_from else args.model
    if is_main:
        print(f"\nLoading model: {src}")

    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype=torch.bfloat16,
    ).to(device)

    raw_model = model
    if is_distributed:
        model = DDP(raw_model, device_ids=[local_rank])

    model.train()
    opt = torch.optim.AdamW(raw_model.parameters(), lr=args.lr)

    # ── Training loop ─────────────────────────────────────────────────────────
    if is_main:
        print(f"\nStarting GRPO | {args.steps} steps | group_size={args.group_size} | {args.group_size // world_size} samples/GPU")

    log_file = open(os.path.join(run_dir, "train_log.jsonl"), "w") if is_main else None

    for step in range(1, args.steps + 1):
        ex    = train_pack[rng.integers(0, len(train_pack))]
        ex_id = ex["id"]
        gold  = ex["gold"]

        if reranker_w is not None:
            chosen_rows      = retrieve_for_training(ex_id, data, reranker_w, args.top_k, device)
            prompt           = build_prompt(ex["question"], chosen_rows)
            retrieved_titles = [r["title"] for r in chosen_rows]
        else:
            prompt           = ex["prompt"]
            retrieved_titles = ex["retrieved_titles"]

        gm  = data["gold_support_train"][ex_id]
        enc = tok(prompt, return_tensors="pt",
                  truncation=True, max_length=MAX_INPUT_LENGTH).to(device)
        input_ids = enc["input_ids"]
        attn      = enc["attention_mask"]
        input_len = input_ids.shape[1]

        # Generate samples (each rank generates its share)
        samples_this_rank = args.group_size // world_size
        preds, answer_ids_list = [], []

        for _ in range(samples_this_rank):
            with torch.no_grad():
                gen_ids = raw_model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=GRPO_TEMPERATURE,
                    top_p=GRPO_TOP_P,
                    pad_token_id=tok.eos_token_id,
                )
            ans_ids = gen_ids[:, input_len:]
            pred    = tok.decode(ans_ids[0], skip_special_tokens=True).strip()
            preds.append(pred)
            answer_ids_list.append(ans_ids)

        # Compute local rewards
        local_rewards = [
            compute_reward(
                pred=p, gold=gold,
                retrieved_titles=retrieved_titles, gold_map=gm,
                tokenizer=tok, max_new_tokens=MAX_NEW_TOKENS,
                w_ans=args.w_ans, w_faith=args.w_faith,
                lambda_cost=args.lambda_cost, gate=args.gate,
            )["R"]
            for p in preds
        ]

        # Gather rewards across ranks for group-relative advantage
        if is_distributed:
            all_gathered = [None] * world_size
            dist.all_gather_object(all_gathered, local_rewards)
            flat_rewards = [r for g in all_gathered for r in g]
        else:
            flat_rewards = local_rewards

        mean_r = float(np.mean(flat_rewards))
        adv    = torch.tensor(
            [r - mean_r for r in local_rewards],
            dtype=torch.float32, device=device
        )

        # Log probs and loss
        logps = torch.stack([
            causal_logprob(model, input_ids, attn, ans_ids)
            for ans_ids in answer_ids_list
        ])
        loss = -(adv.detach() * logps).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        opt.step()

        # Logging
        if is_main and step % GRPO_LOG_EVERY == 0:
            rewards_t = torch.tensor(flat_rewards)
            rec = {
                "step":        step,
                "reward_mean": float(rewards_t.mean()),
                "reward_std":  float(rewards_t.std()),
                "reward_max":  float(rewards_t.max()),
                "loss":        float(loss.detach()),
            }
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            print(f"step {step:5d} | R={rec['reward_mean']:.4f}±{rec['reward_std']:.4f} | loss={rec['loss']:.4f}")

        # Checkpoint
        if is_main and step % GRPO_CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:07d}")
            os.makedirs(ckpt_path, exist_ok=True)
            raw_model.save_pretrained(ckpt_path)
            tok.save_pretrained(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    if log_file:
        log_file.close()

    # ── Save final ────────────────────────────────────────────────────────────
    if is_main:
        final_ckpt = os.path.join(ckpt_dir, "final")
        os.makedirs(final_ckpt, exist_ok=True)
        raw_model.save_pretrained(final_ckpt)
        tok.save_pretrained(final_ckpt)
        print(f"\nFinal model saved: {final_ckpt}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if is_main:
        print("\nRunning evaluation...")
        raw_model.eval()
        evaluate(
            model=raw_model, tokenizer=tok,
            val_data_dict=data, run_dir=run_dir, run_name=run_name,
            reranker_w=reranker_w, device=device,
        )
        print(f"\nDone. Results: {run_dir}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()