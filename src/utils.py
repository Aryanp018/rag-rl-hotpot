"""
Shared utilities: metrics, reward functions, prompt building.
All functions are pure (no side effects) and importable anywhere.
"""
import re
from collections import Counter
from typing import List, Dict, Any


# ── Text normalization ────────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


# ── Answer quality metrics ────────────────────────────────────────────────────

def em_score(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    pc, gc = Counter(pred_toks), Counter(gold_toks)
    num_same = sum((pc & gc).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def title_recall(retrieved_titles: List[str], gold_map: Dict[str, Any]) -> float:
    gold_titles = set(gold_map.keys())
    ret = set(retrieved_titles)
    return len(ret & gold_titles) / len(gold_titles) if gold_titles else 0.0


def sentence_recall(retrieved_titles: List[str], gold_map: Dict[str, Any]) -> float:
    total = sum(len(s) for s in gold_map.values())
    if total == 0:
        return 0.0
    got = 0
    ret = set(retrieved_titles)
    for t, sids in gold_map.items():
        if t in ret:
            got += len(sids)
    return got / total


def hit_at_k(retrieved_titles: List[str], gold_map: Dict[str, Any]) -> float:
    gold_titles = set(gold_map.keys())
    return 1.0 if set(retrieved_titles) & gold_titles else 0.0


# ── Context utilization (new mechanistic metric) ──────────────────────────────

def context_utilization(pred: str, chosen_rows: List[Dict]) -> float:
    """
    Fraction of prediction tokens that appear in retrieved context.
    Higher = generator is using the context.
    Key metric for showing soft-gate is working mechanistically.
    """
    context_tokens = set()
    for r in chosen_rows:
        context_tokens.update(normalize_text(r["text"]).split())
    pred_tokens = normalize_text(pred).split()
    if not pred_tokens:
        return 0.0
    overlap = sum(1 for t in pred_tokens if t in context_tokens)
    return overlap / len(pred_tokens)


# ── Reward functions ──────────────────────────────────────────────────────────

def soft_gate(r_faith: float, r_ans: float, eps: float = 0.05, alpha: float = 0.5) -> float:
    """Faith reward gated by answer quality. Core contribution."""
    return r_faith * float((eps + r_ans) ** alpha)


def hard_gate(r_faith: float, r_ans: float) -> float:
    """Faith reward only when answer is non-zero."""
    return r_faith if r_ans > 0 else 0.0


def compute_reward(
    pred: str,
    gold: str,
    retrieved_titles: List[str],
    gold_map: Dict[str, Any],
    tokenizer,
    max_new_tokens: int,
    w_ans: float = 2.0,
    w_faith: float = 1.0,
    lambda_cost: float = 0.01,
    gate: str = "soft",   # "soft" | "hard" | "none"
) -> Dict[str, float]:
    r_ans = f1_score(pred, gold)
    r_faith_base = sentence_recall(retrieved_titles, gold_map)

    if gate == "soft":
        r_faith_eff = soft_gate(r_faith_base, r_ans)
    elif gate == "hard":
        r_faith_eff = hard_gate(r_faith_base, r_ans)
    else:  # none
        r_faith_eff = r_faith_base

    n_toks = len(tokenizer.encode(pred))
    r_cost = n_toks / max(1, max_new_tokens)

    R = w_ans * r_ans + w_faith * r_faith_eff - lambda_cost * r_cost

    return {
        "R": float(R),
        "r_ans": float(r_ans),
        "r_faith_base": float(r_faith_base),
        "r_faith_eff": float(r_faith_eff),
        "r_cost": float(r_cost),
    }


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(question: str, chosen_rows: List[Dict]) -> str:
    ctx = "\n".join([f"[{r['title']}] {r['text']}" for r in chosen_rows])
    return (
        f"<s>[INST] Answer the question using only the provided context. Be concise.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question} [/INST]"
    )


# ── Aggregated eval dict ──────────────────────────────────────────────────────

def aggregate_metrics(
    ems, f1s, title_recalls, sent_recalls, hit_ks,
    ctx_utils_correct=None, ctx_utils_incorrect=None
) -> Dict[str, float]:
    import numpy as np
    d = {
        "em": float(np.mean(ems)),
        "f1": float(np.mean(f1s)),
        "title_recall": float(np.mean(title_recalls)),
        "sentence_recall": float(np.mean(sent_recalls)),
        "hit_at_k": float(np.mean(hit_ks)),
    }
    if ctx_utils_correct is not None and ctx_utils_correct:
        d["ctx_util_correct"] = float(np.mean(ctx_utils_correct))
    if ctx_utils_incorrect is not None and ctx_utils_incorrect:
        d["ctx_util_incorrect"] = float(np.mean(ctx_utils_incorrect))
    return d
