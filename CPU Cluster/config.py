# =============================================================
#  CPU CLUSTER LAB — CONFIGURATION
#  University Research Project | Distributed LLM Training
# =============================================================
#
#  THIS IS THE ONLY FILE YOU NEED TO EDIT.
#  Change MASTER_ADDR to Node 01's IP, same value on every node.
#  Run `ip addr` on Node 01 (Ubuntu) to find the IP.
#
# =============================================================

# ── CHANGE THIS (same value on all nodes) ────────────────────
MASTER_ADDR = "192.168.1.101"   # Node 01 (master) IP address

# ── NETWORK ──────────────────────────────────────────────────
MASTER_PORT  = "29500"          # Port used for node communication
BACKEND      = "gloo"           # CPU-native backend (no GPU required)
TIMEOUT_MIN  = 30               # Minutes to wait before cluster times out
RDZV_ID      = "lab-cluster"    # Unique run identifier (change per experiment)

# ── PROJECT PATHS ─────────────────────────────────────────────
RESULTS_DIR  = "results"        # Benchmark/metric CSVs saved here
LOG_DIR      = "logs"           # Training logs saved here
DATA_DIR     = "data"           # Datasets downloaded here automatically
MODEL_DIR    = "models"         # Model checkpoints saved here

# ── HARDWARE PROFILE (your specific setup) ───────────────────
# Node 01 — Acer Aspire | Intel i5-10th Gen | 20 GB RAM
# Node 02 — HP Pavilion | Intel i7-7th Gen  | 12 GB RAM | NVIDIA 940MX
# Bottleneck node is HP (12 GB) — batch sizes are tuned to fit

# ── TRAINING DEFAULTS ─────────────────────────────────────────
DEFAULT_BATCH_SIZE = 32         # Per-node batch (safe for 12 GB RAM)
DEFAULT_EPOCHS     = 5          # Epochs for CIFAR-10 demo
DEFAULT_LR         = 0.01       # Learning rate (SGD for ResNet, AdamW for GPT-2)
GPT2_SEQ_LEN       = 512        # Token sequence length for GPT-2 training
GPT2_BATCH_SIZE    = 4          # Per-node GPT-2 batch (larger model = smaller batch)
