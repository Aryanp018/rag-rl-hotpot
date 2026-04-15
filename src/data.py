"""
Data loading and preprocessing for HotpotQA.
Handles: loading, flattening, embedding, caching.
"""
import os
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Any

import torch
from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer

from config import (
    CACHE_DIR, EMB_DIR, EMB_MODEL,
    DATASET_NAME, DATASET_CONFIG,
    TRAIN_SIZE, VAL_SIZE
)


# ── Load raw dataset ──────────────────────────────────────────────────────────

def load_hotpot(train_size: int = TRAIN_SIZE, val_size: int = VAL_SIZE):
    """Load HotpotQA with caching to disk."""
    train_cache = os.path.join(CACHE_DIR, f"hotpot_train_{train_size}")
    val_cache = os.path.join(CACHE_DIR, f"hotpot_val_{val_size}")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("Loading cached dataset splits...")
        train_data = load_from_disk(train_cache)
        val_data = load_from_disk(val_cache)
    else:
        print(f"Downloading HotpotQA (train={train_size}, val={val_size})...")
        os.makedirs(CACHE_DIR, exist_ok=True)
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
        train_data = dataset["train"].select(range(train_size))
        val_data = dataset["validation"].select(range(val_size))
        train_data.save_to_disk(train_cache)
        val_data.save_to_disk(val_cache)
        print(f"Saved to {CACHE_DIR}")

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    return train_data, val_data


# ── Flatten into paragraph rows ───────────────────────────────────────────────

def flatten_hotpot(ds) -> List[Dict]:
    """
    Convert HotpotQA examples into flat paragraph rows.
    Each row = one paragraph with its question/answer/gold labels.
    """
    rows = []
    for ex in ds:
        ex_id = ex["id"]
        q = ex["question"]
        a = ex["answer"]

        # gold map: title -> set of supporting sentence ids
        gold_map = defaultdict(set)
        for t, sid in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]):
            gold_map[t].add(int(sid))

        titles = ex["context"]["title"]
        sentences_lists = ex["context"]["sentences"]

        for p_idx, (title, sents) in enumerate(zip(titles, sentences_lists)):
            gold_sids = sorted(list(gold_map.get(title, set())))
            rows.append({
                "ex_id": ex_id,
                "p_idx": p_idx,
                "para_id": f"{ex_id}::{p_idx}",
                "title": title,
                "sentences": sents,
                "text": " ".join(sents),
                "is_gold_para": int(title in gold_map),
                "gold_sent_ids": gold_sids,
                "question": q,
                "answer": a,
            })
    return rows


# ── Build grouping structures ─────────────────────────────────────────────────

def build_grouping(rows: List[Dict]) -> Tuple[Dict, List[str]]:
    """
    Returns:
        by_ex: ex_id -> [row indices]
        ex_ids: ordered list of example ids
    """
    by_ex = defaultdict(list)
    for i, r in enumerate(rows):
        by_ex[r["ex_id"]].append(i)
    return dict(by_ex), list(by_ex.keys())


def build_gold_support(ds) -> Dict[str, Dict]:
    """Returns ex_id -> {title: set(sent_ids)}"""
    gold_support = {}
    for ex in ds:
        gm = defaultdict(set)
        for t, sid in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]):
            gm[t].add(int(sid))
        gold_support[ex["id"]] = dict(gm)
    return gold_support


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embeddings(
    texts: List[str],
    cache_path: str,
    device: str,
    batch_size: int = 128,
    force_recompute: bool = False,
) -> np.ndarray:
    """Compute or load cached sentence embeddings."""
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached embeddings: {cache_path}")
        return np.load(cache_path)

    print(f"Computing embeddings for {len(texts)} texts...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    embedder = SentenceTransformer(EMB_MODEL, device=device)
    emb = embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(cache_path, emb)
    print(f"Saved embeddings: {cache_path}")
    return emb


# ── Build grouped torch tensors ───────────────────────────────────────────────

def build_group_tensors(
    rows: List[Dict],
    by_ex: Dict[str, List[int]],
    para_emb: np.ndarray,
    device: str,
) -> Tuple[Dict, Dict]:
    """
    Returns:
        group_emb:  ex_id -> FloatTensor [n_paras, D]
        group_rows: ex_id -> list of row dicts
    """
    para_emb_t = torch.tensor(para_emb, dtype=torch.float32, device=device)
    group_emb = {}
    group_rows = {}
    for ex_id, idxs in by_ex.items():
        group_emb[ex_id] = para_emb_t[idxs]
        group_rows[ex_id] = [rows[i] for i in idxs]
    return group_emb, group_rows


def build_q_vecs(
    ex_ids: List[str],
    questions: Dict[str, str],
    q_emb: np.ndarray,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Returns ex_id -> question embedding tensor."""
    return {
        eid: torch.tensor(q_emb[i], dtype=torch.float32, device=device)
        for i, eid in enumerate(ex_ids)
    }


# ── Train pack (precomputed prompts for GRPO) ─────────────────────────────────

def build_train_pack(
    train_data,
    train_rows: List[Dict],
    train_by_ex: Dict[str, List[int]],
    train_para_emb: np.ndarray,
    train_ex_ids: List[str],
    top_k: int = 3,
) -> List[Dict]:
    """
    Precompute baseline retrieval + prompts for training.
    These are fixed-retrieval prompts used during GRPO training.
    """
    from utils import build_prompt

    questions = {ex["id"]: ex["question"] for ex in train_data}
    gold_answers = {ex["id"]: ex["answer"] for ex in train_data}

    # Compute question embeddings
    embedder = SentenceTransformer(EMB_MODEL)
    q_texts = [questions[eid] for eid in train_ex_ids]
    q_emb = embedder.encode(
        q_texts, batch_size=128, convert_to_numpy=True,
        show_progress_bar=True, normalize_embeddings=True
    )

    train_pack = []
    for qi, ex_id in enumerate(train_ex_ids):
        idxs = train_by_ex[ex_id]
        sims = train_para_emb[idxs] @ q_emb[qi]
        top_local = np.argsort(-sims)[:top_k]
        chosen_rows = [train_rows[idxs[j]] for j in top_local]

        train_pack.append({
            "id": ex_id,
            "question": questions[ex_id],
            "gold": gold_answers[ex_id],
            "retrieved_titles": [r["title"] for r in chosen_rows],
            "prompt": build_prompt(questions[ex_id], chosen_rows),
        })

    return train_pack


# ── Full data setup (called by all training scripts) ─────────────────────────

def setup_data(device: str, train_size: int = TRAIN_SIZE, val_size: int = VAL_SIZE,
               dataset_name: str = "hotpotqa"):
    """
    One-stop function that returns everything needed for training/eval.
    Caches embeddings to disk.
    """
    train_data, val_data = load_hotpot(train_size, val_size)

    train_rows = flatten_hotpot(train_data)
    val_rows = flatten_hotpot(val_data)

    train_by_ex, train_ex_ids = build_grouping(train_rows)
    val_by_ex, val_ex_ids = build_grouping(val_rows)

    gold_support_train = build_gold_support(train_data)
    gold_support_val = build_gold_support(val_data)

    questions_train = {ex["id"]: ex["question"] for ex in train_data}
    questions_val = {ex["id"]: ex["question"] for ex in val_data}
    gold_answers_val = {ex["id"]: ex["answer"] for ex in val_data}
    gold_answers_train = {ex["id"]: ex["answer"] for ex in train_data}

    # Paragraph embeddings
    train_para_emb = get_embeddings(
        [r["text"] for r in train_rows],
        os.path.join(EMB_DIR, f"train_para_{train_size}.npy"),
        device,
    )
    val_para_emb = get_embeddings(
        [r["text"] for r in val_rows],
        os.path.join(EMB_DIR, f"val_para_{val_size}.npy"),
        device,
    )

    # Question embeddings
    train_q_emb = get_embeddings(
        [questions_train[eid] for eid in train_ex_ids],
        os.path.join(EMB_DIR, f"train_q_{train_size}.npy"),
        device,
    )
    val_q_emb = get_embeddings(
        [questions_val[eid] for eid in val_ex_ids],
        os.path.join(EMB_DIR, f"val_q_{val_size}.npy"),
        device,
    )

    # Grouped tensors
    train_group_emb, train_group_rows = build_group_tensors(train_rows, train_by_ex, train_para_emb, device)
    val_group_emb, val_group_rows = build_group_tensors(val_rows, val_by_ex, val_para_emb, device)

    # Question vectors
    train_q_vecs = build_q_vecs(train_ex_ids, questions_train, train_q_emb, device)
    val_q_vecs = build_q_vecs(val_ex_ids, questions_val, val_q_emb, device)

    return {
        # raw data
        "train_data": train_data,
        "val_data": val_data,
        "train_rows": train_rows,
        "val_rows": val_rows,
        # grouping
        "train_by_ex": train_by_ex,
        "val_by_ex": val_by_ex,
        "train_ex_ids": train_ex_ids,
        "val_ex_ids": val_ex_ids,
        # gold maps
        "gold_support_train": gold_support_train,
        "gold_support_val": gold_support_val,
        # text maps
        "questions_train": questions_train,
        "questions_val": questions_val,
        "gold_answers_train": gold_answers_train,
        "gold_answers_val": gold_answers_val,
        # embeddings (numpy)
        "train_para_emb": train_para_emb,
        "val_para_emb": val_para_emb,
        "train_q_emb": train_q_emb,
        "val_q_emb": val_q_emb,
        # grouped tensors
        "train_group_emb": train_group_emb,
        "train_group_rows": train_group_rows,
        "val_group_emb": val_group_emb,
        "val_group_rows": val_group_rows,
        # question vectors
        "train_q_vecs": train_q_vecs,
        "val_q_vecs": val_q_vecs,
    }
