# RAG-RL Experiment Context

## What This Project Is
Research implementation of **RAG-RL: Stable Joint Training of Retriever and Generator via Soft-Gated Faithfulness Reward**.
Trains a RAG system using GRPO (Group Relative Policy Optimization) with a soft-gate reward that prevents the generator
from being rewarded for hallucinating answers not supported by retrieved context.

Primary benchmark: **HotpotQA** (multi-hop QA). Also benchmarking on **MuSiQue** and **2WikiMultiHopQA**.

---

## Hardware
- **4x NVIDIA A6000 48GB GPUs** (192GB total VRAM)
- ~500GB free storage
- Training window: typically 8 PM – 6 AM (~10 hours)
- Strategy: **run one experiment at a time using all 4 GPUs via DDP** (not 4 experiments in parallel)

---

## Current Model Decisions

### Generator (being changed)
| | Old | New |
|--|-----|-----|
| Model | `google/flan-t5-xl` (3B) | `mistralai/Mistral-7B-Instruct-v0.3` (7B) |
| Architecture | Encoder-Decoder (seq2seq) | Decoder-only (causal LM) |
| Class | `AutoModelForSeq2SeqLM` | `AutoModelForCausalLM` |
| Reason for change | High ctx_util_incorrect — model reads context but can't reason over it (multi-hop bottleneck) | Stronger multi-hop reasoning, full attention over context at every step |

### Retriever (unchanged)
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- Method: cosine similarity by default; optional learned CE reranker (`w.pt`)
- TOP_K: 3 (consider bumping to 4)

---

## Config Changes From Baseline
All in `src/config.py`:

```python
# Generator
GEN_MODEL        = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS   = 64          # was 32
MAX_INPUT_LENGTH = 2048        # was 512

# Data — scaled up from tiny 2K/500 split
TRAIN_SIZE       = 10000       # was 2000 (was only 2.2% of HotpotQA)
VAL_SIZE         = 1000        # was 500

# GRPO — tuned for 4-GPU DDP
GRPO_STEPS       = 3000        # was 2000; fits in ~1.7 hrs on 4x A6000
GRPO_GROUP_SIZE  = 32          # was 8; 8 samples per GPU across 4 GPUs
GRPO_CKPT_EVERY  = 3001        # only saves final checkpoint — saves ~200GB storage
GRPO_LR          = 5e-6
GRPO_TEMPERATURE = 0.7
```

---

## Code Changes Required (Decoder-Only Migration)

### 1. `src/config.py`
- Update `GEN_MODEL` and `GRPO_GROUP_SIZE` as above.

### 2. `src/evaluate.py`
- `AutoModelForSeq2SeqLM` → `AutoModelForCausalLM`
- Slice prompt tokens from generated output:
  ```python
  input_len = inputs["input_ids"].shape[1]
  pred = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
  ```

### 3. `scripts/train_grpo.py`
- `AutoModelForSeq2SeqLM` → `AutoModelForCausalLM`
- Replace `seq2seq_logprob()` with `causal_logprob()`:
  ```python
  def causal_logprob(model, input_ids, attention_mask, answer_ids):
      full_ids  = torch.cat([input_ids, answer_ids], dim=1)
      full_mask = torch.cat([attention_mask, torch.ones_like(answer_ids)], dim=1)
      labels    = torch.full_like(full_ids, -100)
      labels[:, input_ids.shape[1]:] = answer_ids
      out = model(input_ids=full_ids, attention_mask=full_mask, labels=labels)
      return -out.loss * answer_ids.shape[1]
  ```
- Add DDP init and `torchrun` support (see DDP section below)
- Slice answer from generated output: `answer_ids = gen_ids[:, input_len:]`

### 4. `src/utils.py` — `build_prompt()`
Update to Mistral instruction format:
```python
def build_prompt(question: str, chosen_rows: List[Dict]) -> str:
    ctx = "\n".join([f"[{r['title']}] {r['text']}" for r in chosen_rows])
    return (
        f"<s>[INST] Answer the question using only the provided context. Be concise.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question} [/INST]"
    )
```

### 5. `scripts/eval_baseline.py`
- Same import swap + output slicing as evaluate.py

---

## Multi-GPU DDP Setup (All 4 GPUs Per Run)

Add to `scripts/train_grpo.py`:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# In main():
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device     = f"cuda:{local_rank}"
is_main    = local_rank == 0

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
model = DDP(model, device_ids=[local_rank])

# Each rank generates group_size // world_size samples
samples_per_rank = args.group_size // dist.get_world_size()

# After local reward computation, gather across all ranks:
all_rewards = [None] * dist.get_world_size()
dist.all_gather_object(all_rewards, local_rewards)
flat_rewards = [r for group in all_rewards for r in group]
mean_r = np.mean(flat_rewards)

# Save only from main process:
if is_main:
    model.module.save_pretrained(ckpt_path)
```

Launch command:
```bash
torchrun --nproc_per_node=4 scripts/train_grpo.py \
    --dataset hotpotqa --gate soft --seed 42 \
    --steps 3000 --train_size 10000 --val_size 1000
```

---

## Dataset Support

Three datasets, all multi-hop QA:

| Dataset | HF name | HF config | Schema |
|---------|---------|-----------|--------|
| HotpotQA | `hotpot_qa` | `distractor` | `context.title/sentences` + `supporting_facts.title/sent_id` |
| 2WikiMultiHopQA | `2wikimultihop` | none | Same as HotpotQA but id field is `_id` |
| MuSiQue | `musique` | `musique_ans_v1.0` | `paragraphs[].title/paragraph_text/is_supporting` (paragraph-level gold only) |

Dataset-specific loaders added to `src/data.py`:
- `flatten_2wiki()` — same as HotpotQA logic, uses `_id`
- `flatten_musique()` — uses `is_supporting` boolean, no sentence-level gold
- `build_gold_support_2wiki()` — uses `_id`
- `build_gold_support_musique()` — returns `title -> set()` (empty, paragraph-level only)
- `setup_data(... dataset_name="hotpotqa")` — accepts `hotpotqa | 2wikimultihop | musique`

Add `--dataset` arg to all training/eval scripts:
```python
p.add_argument("--dataset", type=str, default="hotpotqa",
               choices=["hotpotqa", "2wikimultihop", "musique"])
```

---

## Typical Overnight Schedule (8 PM – 6 AM)

Each experiment takes ~1.7 hrs on 4x A6000 with 3000 steps, group_size=32.
Run sequentially, all 4 GPUs per run:

```
8:00 PM  → HotpotQA      seed 42   (~1.7 hrs)
9:45 PM  → HotpotQA      seed 123  (~1.7 hrs)
11:30 PM → HotpotQA      seed 456  (~1.7 hrs)
1:15 AM  → 2WikiMultiHop seed 42   (~1.7 hrs)
3:00 AM  → MuSiQue       seed 42   (~1.7 hrs)
4:45 AM  → buffer / eval
6:00 AM  → Done
```

Launch script template (`run_tonight.sh`):
```bash
#!/bin/bash
torchrun --nproc_per_node=4 scripts/train_grpo.py \
    --dataset $1 --gate soft --seed $2 \
    --steps 3000 --train_size 10000 --val_size 1000
```

---

## Key Metric to Watch: `ctx_util_incorrect`
- **Problem**: FLAN-T5-XL had high `ctx_util_incorrect` — model reads context but still gets wrong answers
- **Root cause**: 3B encoder-decoder can't do multi-hop reasoning over compressed context
- **Fix**: Mistral 7B decoder-only attends directly over all context tokens at every generation step
- **Expected**: `ctx_util_incorrect` should drop; `ctx_util_correct` should stay high or rise

---

## Storage Budget

| Item | Size |
|------|------|
| Mistral 7B weights (bf16) | ~14 GB |
| Final checkpoint per run | ~14 GB |
| 5 runs (tonight) | ~70 GB |
| Embedding caches (all 3 datasets) | ~500 MB |
| Predictions + logs | ~200 MB |
| **Total tonight** | **~71 GB / 500 GB available** ✅ |

---

## File Map

```
src/
  config.py       — all hyperparameters, model names, dataset registry
  data.py         — data loading, flattening, embeddings, setup_data()
  evaluate.py     — shared eval loop: retrieve → prompt → generate → score
  utils.py        — metrics (EM/F1/recall), rewards (soft/hard gate), build_prompt()

scripts/
  train_grpo.py   — GRPO training loop, DDP support
  train_reranker.py — CE reranker training (supervised, frozen generator)
  eval_baseline.py  — baseline evaluation script
  aggregate_results.py — loads all metrics.json, produces tables + plots

run_all_experiments.py  — orchestrates full suite across seeds
```

---

## Ablation Variants (gate parameter)
| Gate | Behavior |
|------|----------|
| `soft` | `r_faith *= (eps + r_ans)^alpha` — main contribution |
| `hard` | `r_faith = r_faith if r_ans > 0 else 0` — binary gate |
| `none` | `r_faith` unchanged — baseline faithfulness reward |

Reward formula: `R = 2.0 * r_ans + 1.0 * r_faith_eff - 0.01 * length_penalty`
