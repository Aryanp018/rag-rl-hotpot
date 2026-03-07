# RAG-RL: Stable Joint Training of Retriever and Generator via Soft-Gated Faithfulness Reward

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
rag_rl/
├── src/
│   ├── config.py          # All hyperparameters in one place
│   ├── data.py            # Data loading, preprocessing, embeddings
│   ├── utils.py           # Metrics, reward functions, prompt building
│   └── evaluate.py        # Shared evaluation loop
├── scripts/
│   ├── eval_baseline.py   # Baseline (no training)
│   ├── train_reranker.py  # CE Reranker training
│   ├── train_grpo.py      # GRPO training (all variants)
│   └── aggregate_results.py  # Build summary table + plots
├── run_all_experiments.py # Master runner
├── requirements.txt
└── README.md
```

## Quick Start: Run Everything

```bash
# Run all experiments with 3 seeds (takes ~12-15 hours on A6000)
python run_all_experiments.py --seeds 42 123 456

# Quick test with 1 seed
python run_all_experiments.py --seeds 42 --grpo_steps 500

# See what commands will run without executing
python run_all_experiments.py --dry_run
```

## Run Individual Experiments

### 1. Baseline
```bash
PYTHONPATH=src python scripts/eval_baseline.py
```

### 2. CE Reranker
```bash
PYTHONPATH=src python scripts/train_reranker.py --seed 42
```

### 3. GRPO Soft-Gate (your main contribution)
```bash
PYTHONPATH=src python scripts/train_grpo.py --gate soft --seed 42
```

### 4. GRPO No-Faith Ablation
```bash
PYTHONPATH=src python scripts/train_grpo.py --gate none --seed 42
```

### 5. GRPO Hard-Gate Ablation
```bash
PYTHONPATH=src python scripts/train_grpo.py --gate hard --seed 42
```

### 6. Stable Combined (CE Reranker + GRPO Soft-Gate)
```bash
# First train the reranker, then pass its path here
PYTHONPATH=src python scripts/train_grpo.py \
    --gate soft \
    --seed 42 \
    --reranker_path runs/reranker_ce/YOUR_RUN/w.pt
```

## Aggregate Results

```bash
PYTHONPATH=src python scripts/aggregate_results.py
```

This produces `runs/summary/`:
- `summary_table.csv` — mean ± std across seeds
- `answer_quality.png` — EM/F1 bar chart
- `retrieval_metrics.png` — recall line chart
- `training_curves.png` — reward curves (shows instability)
- `ctx_utilization.png` — mechanistic evidence for soft-gate

## Key Config Options (src/config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GEN_MODEL` | `google/flan-t5-xl` | Generator model |
| `TRAIN_SIZE` | 2000 | Training examples |
| `VAL_SIZE` | 500 | Validation examples |
| `GRPO_STEPS` | 2000 | GRPO training steps |
| `GRPO_GROUP_SIZE` | 8 | Samples per GRPO step |
| `W_ANS` | 2.0 | Answer reward weight |
| `W_FAITH` | 1.0 | Faithfulness reward weight |
| `LAMBDA_COST` | 0.01 | Length penalty weight |
| `FAITH_EPS` | 0.05 | Soft gate epsilon |
| `FAITH_ALPHA` | 0.5 | Soft gate power |

## Reward Function (core contribution)

```python
# Standard faithfulness reward
r_faith = sentence_recall(retrieved_titles, gold_map)

# Soft gate: faith only counts when answer is also correct
# This is the key contribution — prevents rewarding faithfulness to irrelevant context
r_faith_eff = r_faith * (eps + r_ans) ** alpha

# Total reward
R = W_ANS * r_ans + W_FAITH * r_faith_eff - LAMBDA_COST * length_penalty
```

## Paper Experiments Checklist

- [ ] Baseline on HotpotQA (2000 train / 500 val)
- [ ] CE Reranker + Frozen Gen
- [ ] GRPO Soft-Gate only (fixed retriever)
- [ ] GRPO No-Faith ablation
- [ ] GRPO Hard-Gate ablation
- [ ] Stable Combined (main system)
- [ ] All above on Natural Questions (second dataset)
- [ ] All above with 3 seeds
- [ ] Training curve analysis (instability evidence)
- [ ] Context utilization analysis (mechanistic evidence)
