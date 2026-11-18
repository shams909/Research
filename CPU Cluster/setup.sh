#!/bin/bash
# =============================================================
#  CPU CLUSTER LAB — UBUNTU SERVER SETUP SCRIPT
#  Run this on EVERY NODE after cloning the repo
#  Usage: bash setup.sh
# =============================================================

set -e  # Exit immediately on any error

echo ""
echo "=================================================="
echo "  CPU CLUSTER LAB — Node Setup"
echo "=================================================="

# ── 1. System Info ────────────────────────────────────────────
echo ""
echo "[1/7] System information"
echo "  Hostname : $(hostname)"
echo "  OS       : $(lsb_release -d | cut -f2)"
echo "  IP       : $(hostname -I | awk '{print $1}')"
echo "  CPU      : $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "  RAM      : $(free -h | awk '/^Mem/{print $2}')"

# ── 2. System packages ────────────────────────────────────────
echo ""
echo "[2/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-venv python3-dev python3-pip \
    git curl wget net-tools iperf3 htop
echo "  Done"

# ── 3. Python virtual environment ─────────────────────────────
echo ""
echo "[3/7] Setting up Python virtual environment..."
if [ -d ".venv" ]; then
    echo "  .venv already exists — skipping"
else
    python3 -m venv .venv
    echo "  .venv created at $(pwd)/.venv"
fi
source .venv/bin/activate
echo "  Python: $(python3 --version)"

# ── 4. Install packages ───────────────────────────────────────
echo ""
echo "[4/7] Installing Python packages (may take a few minutes)..."
pip install --upgrade pip --quiet
# CPU-only PyTorch — no CUDA needed
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu --quiet
pip install -r requirements.txt --quiet
echo "  All packages installed"

# ── 5. Create project directories ─────────────────────────────
echo ""
echo "[5/7] Creating project directories..."
mkdir -p results logs data models
echo "  results/  logs/  data/  models/ — ready"

# ── 6. Firewall ───────────────────────────────────────────────
echo ""
echo "[6/7] Configuring firewall..."
sudo ufw allow 29500/tcp comment "Lab Cluster torchrun port" 2>/dev/null || true
sudo ufw allow ssh 2>/dev/null || true
echo "  Port 29500 (torchrun) and 22 (SSH) allowed"

# ── 7. Verify ─────────────────────────────────────────────────
echo ""
echo "[7/7] Verifying installation..."
python3 - <<'EOF'
import torch
import sys

ok = True
checks = {
    "PyTorch version"  : torch.__version__,
    "Distributed"      : torch.distributed.is_available(),
    "Gloo backend"     : torch.distributed.is_gloo_available(),
}

for name, val in checks.items():
    status = "✓" if val else "✗"
    print(f"  {status}  {name}: {val}")
    if val is False:
        ok = False

sys.exit(0 if ok else 1)
EOF

echo ""
echo "=================================================="
echo "  SETUP COMPLETE"
echo "  Node IP : $(hostname -I | awk '{print $1}')"
echo ""
echo "  NEXT STEP:"
echo "  Edit config.py and set MASTER_ADDR to Node 01's IP"
echo "=================================================="
echo ""
