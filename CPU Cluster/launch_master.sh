#!/bin/bash
# =============================================================
#  LAUNCH MASTER NODE (Node 01)
#  Run this on Node 01 FIRST, then start workers on other nodes.
#
#  Usage:
#    bash launch_master.sh <script.py> [nnodes]
#
#  Examples:
#    bash launch_master.sh 01_smoke_test.py        # 2-node cluster
#    bash launch_master.sh 02_train_cifar.py 4     # 4-node cluster
#    bash launch_master.sh 03_train_gpt2.py 2
# =============================================================

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

SCRIPT=${1:-"01_smoke_test.py"}
NNODES=${2:-2}
MASTER_ADDR=$(python3 -c "import config; print(config.MASTER_ADDR)")
MASTER_PORT=$(python3 -c "import config; print(config.MASTER_PORT)")

echo ""
echo "=============================================="
echo "  MASTER NODE | rank=0 | world_size=$NNODES"
echo "  Script  : src/$SCRIPT"
echo "  Address : $MASTER_ADDR:$MASTER_PORT"
echo "  Waiting for $((NNODES-1)) worker(s)..."
echo "=============================================="
echo ""

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=1 \
  --node-rank=0 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv-id=lab-cluster \
  --rdzv-conf=timeout=1800 \
  src/$SCRIPT
