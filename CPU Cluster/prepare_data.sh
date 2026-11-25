#!/bin/bash
# =============================================================
#  PRE-DOWNLOAD DATASETS
#  Run this BEFORE going to the university lab.
#  University networks may block Hugging Face / PyTorch CDN.
#  Datasets are saved to data/ and cached for offline use.
#
#  Usage: bash prepare_data.sh           (all 4 datasets)
#         bash prepare_data.sh --voice   (SpeechCommands + Whisper only)
#  Time : ~5 minutes | Size: ~200 MB - 2.5 GB
# =============================================================

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

VOICE_ONLY=0
if [ "${1:-}" = "--voice" ]; then
    VOICE_ONLY=1
fi

echo ""
echo "=================================================="
echo "  DATASET PREPARATION — CPU Cluster Lab"
echo "=================================================="

if [ "$VOICE_ONLY" -eq 0 ]; then
# ── CIFAR-10 ─────────────────────────────────────────────────
echo ""
echo "[1/4] Downloading CIFAR-10 (~170 MB)..."
python3 - <<'EOF'
import torchvision
torchvision.datasets.CIFAR10(root="data", train=True,  download=True)
torchvision.datasets.CIFAR10(root="data", train=False, download=True)
print("  CIFAR-10 ready ✓")
EOF

# ── WikiText-2 (for GPT-2) ───────────────────────────────────
echo ""
echo "[2/4] Downloading WikiText-2 (~5 MB)..."
python3 - <<'EOF'
from datasets import load_dataset
try:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
except Exception:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
print(f"  WikiText-2 ready ✓  ({len(ds['train'])} train samples)")
EOF

# ── GPT-2 tokenizer ─────────────────────────────────────────
echo ""
echo "[3/4] Caching GPT-2 tokenizer..."
python3 - <<'EOF'
from transformers import GPT2Tokenizer
GPT2Tokenizer.from_pretrained("gpt2")
print("  GPT-2 tokenizer ready ✓")
EOF
fi

# ── Speech Commands & Whisper-Tiny tokenizer (for Voice) ────
echo ""
echo "[4/4] Downloading Speech Commands + Whisper tokenizer (~2.3 GB)..."
python3 - <<'EOF'
import torchaudio
from transformers import WhisperTokenizer
WhisperTokenizer.from_pretrained("openai/whisper-tiny")
torchaudio.datasets.SPEECHCOMMANDS(root="data", download=True)
print("  SpeechCommands + Whisper tokenizer ready ✓")
EOF

echo ""
echo "=================================================="
echo "  ALL DATASETS READY — Safe to go offline"
echo "  Saved to: data/  and  ~/.cache/huggingface/"
echo "=================================================="
echo ""
