#!/bin/bash
# =============================================================
#  LAUNCH SINGLE NODE (Baseline — No Cluster)
#  Run this on ONE laptop ALONE to get your single-node baseline.
#  Critical for research: compare single vs multi-node throughput.
#
#  Scaling Efficiency = (N-node throughput) / (N × single throughput)
#
#  Usage:
#    bash launch_single.sh 02_train_cifar.py
#    bash launch_single.sh 03_train_gpt2.py
#    bash launch_single.sh 04_benchmark.py
# =============================================================

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

SCRIPT=${1:-"02_train_cifar.py"}
shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

echo ""
echo "=============================================="
echo "  SINGLE NODE BASELINE | rank=0 | world_size=1"
echo "  Script : src/$SCRIPT"
echo "  Args   : $EXTRA_ARGS"
echo "  Run this ALONE — no other nodes needed"
echo "=============================================="
echo ""

torchrun \
  --nnodes=1 \
  --nproc-per-node=1 \
  --node-rank=0 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:29500 \
  --rdzv-id=lab-cluster-single \
  src/$SCRIPT $EXTRA_ARGS
