"""
Shared evaluation loop used by all experiment scripts.
Evaluates a (retriever, generator) pair on validation set.
"""
import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Callable
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils import (
    em_score, f1_score, title_recall, sentence_recall,
    hit_at_k, context_utilization, build_prompt, aggregate_metrics
)
from config import TOP_K, MAX_NEW_TOKENS, MAX_INPUT_LENGTH


def retrieve_topk(
    ex_id: str,
    val_group_emb: Dict,
    val_group_rows: Dict,
    val_q_vecs: Dict,
    top_k: int,
    reranker_w: Optional[torch.Tensor] = None,
) -> List[Dict]:
    """
    Retrieve top-k paragraphs.
    If reranker_w is provided, uses learned reranker.
    Otherwise uses cosine similarity baseline.
    """
    P = val_group_emb[ex_id]      # [n_paras, D]
    qv = val_q_vecs[ex_id]        # [D]
    rows = val_group_rows[ex_id]

    if reranker_w is not None:
        feats = P * qv.unsqueeze(0)
        scores = (feats @ reranker_w).detach()
    else:
        scores = (P @ qv).detach()

    topk_idx = torch.topk(scores, k=min(top_k, scores.numel())).indices.tolist()
    return [rows[i] for i in topk_idx]


def evaluate(
    model,
    tokenizer,
    val_data_dict: Dict,
    run_dir: str,
    run_name: str,
    reranker_w: Optional[torch.Tensor] = None,
    top_k: int = TOP_K,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = "cuda",
    save_predictions: bool = True,
) -> Dict[str, float]:
    """
    Full evaluation loop. Returns aggregated metrics dict.
    Saves metrics.json and predictions.jsonl to run_dir.
    """
    model.eval()

    val_ex_ids = val_data_dict["val_ex_ids"]
    val_group_emb = val_data_dict["val_group_emb"]
    val_group_rows = val_data_dict["val_group_rows"]
    val_q_vecs = val_data_dict["val_q_vecs"]
    questions_val = val_data_dict["questions_val"]
    gold_answers_val = val_data_dict["gold_answers_val"]
    gold_support_val = val_data_dict["gold_support_val"]

    ems, f1s = [], []
    title_recalls, sent_recalls, hit_ks = [], [], []
    ctx_utils_correct, ctx_utils_incorrect = [], []
    predictions = []

    for ex_id in val_ex_ids:
        chosen_rows = retrieve_topk(
            ex_id, val_group_emb, val_group_rows,
            val_q_vecs, top_k, reranker_w
        )
        retrieved_titles = [r["title"] for r in chosen_rows]

        prompt = build_prompt(questions_val[ex_id], chosen_rows)
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_INPUT_LENGTH
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        gold = gold_answers_val[ex_id]
        gm = gold_support_val[ex_id]

        em = em_score(pred, gold)
        f1 = f1_score(pred, gold)
        ctx_util = context_utilization(pred, chosen_rows)

        ems.append(em)
        f1s.append(f1)
        title_recalls.append(title_recall(retrieved_titles, gm))
        sent_recalls.append(sentence_recall(retrieved_titles, gm))
        hit_ks.append(hit_at_k(retrieved_titles, gm))

        if em > 0:
            ctx_utils_correct.append(ctx_util)
        else:
            ctx_utils_incorrect.append(ctx_util)

        predictions.append({
            "id": ex_id,
            "question": questions_val[ex_id],
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": f1,
            "ctx_util": ctx_util,
            "retrieved_titles": retrieved_titles,
        })

    metrics = aggregate_metrics(
        ems, f1s, title_recalls, sent_recalls, hit_ks,
        ctx_utils_correct, ctx_utils_incorrect
    )
    metrics["n_examples"] = len(val_ex_ids)
    metrics["run_name"] = run_name

    # Save
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if save_predictions:
        with open(os.path.join(run_dir, "predictions.jsonl"), "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

    print(f"\n{'='*50}")
    print(f"EVAL: {run_name}")
    print(f"  EM:               {metrics['em']:.4f}")
    print(f"  F1:               {metrics['f1']:.4f}")
    print(f"  Title Recall:     {metrics['title_recall']:.4f}")
    print(f"  Sentence Recall:  {metrics['sentence_recall']:.4f}")
    print(f"  Hit@{top_k}:          {metrics['hit_at_k']:.4f}")
    if "ctx_util_correct" in metrics:
        print(f"  CtxUtil(correct):   {metrics['ctx_util_correct']:.4f}")
        print(f"  CtxUtil(incorrect): {metrics['ctx_util_incorrect']:.4f}")
    print(f"{'='*50}\n")

    return metrics
