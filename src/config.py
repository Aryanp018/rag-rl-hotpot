"""
Central config for all experiments.
Edit this file to change hyperparameters.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
CACHE_DIR = os.path.join(ARTIFACTS_DIR, "caches")
EMB_DIR = os.path.join(ARTIFACTS_DIR, "embeddings")

# ── Data ─────────────────────────────────────────────────────────────────────
TRAIN_SIZE = 2000
VAL_SIZE = 500
DATASET_NAME = "hotpot_qa"
DATASET_CONFIG = "distractor"

# ── Retrieval ─────────────────────────────────────────────────────────────────
EMB_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# ── Generator ─────────────────────────────────────────────────────────────────
GEN_MODEL = "google/flan-t5-xl"   # upgrade from base to xl for A6000
MAX_NEW_TOKENS = 32
MAX_INPUT_LENGTH = 512

# ── GRPO ──────────────────────────────────────────────────────────────────────
GRPO_STEPS = 2000
GRPO_GROUP_SIZE = 8
GRPO_LR = 5e-6
GRPO_LOG_EVERY = 20
GRPO_CKPT_EVERY = 200
GRPO_TEMPERATURE = 0.7
GRPO_TOP_P = 0.9

# ── Reward ────────────────────────────────────────────────────────────────────
W_ANS = 2.0
W_FAITH = 1.0
LAMBDA_COST = 0.01
FAITH_EPS = 0.05      # soft gate epsilon
FAITH_ALPHA = 0.5     # soft gate power (sqrt)

# ── CE Reranker ───────────────────────────────────────────────────────────────
RERANKER_STEPS = 2000
RERANKER_LR = 1e-2

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456]

# ── Ablation variants ─────────────────────────────────────────────────────────
# These are passed via argparse in train_grpo.py
ABLATION_CONFIGS = {
    "softgate":  {"w_ans": 2.0, "w_faith": 1.0, "lambda_cost": 0.01, "gate": "soft"},
    "nofaith":   {"w_ans": 2.0, "w_faith": 0.0, "lambda_cost": 0.01, "gate": "none"},
    "hardgate":  {"w_ans": 2.0, "w_faith": 1.0, "lambda_cost": 0.01, "gate": "hard"},
}
