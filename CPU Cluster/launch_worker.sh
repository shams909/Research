#!/bin/bash
# =============================================================
#  LAUNCH WORKER NODE (Node 02, 03, 04 ...)
#  Run AFTER the master has started on Node 01.
#
#  Usage:
#    bash launch_worker.sh <script.py> <node-rank> [nnodes]
#
#  Examples:
#    bash launch_worker.sh 01_smoke_test.py 1       # rank=1, 2 nodes
#    bash launch_worker.sh 02_train_cifar.py 1 4    # rank=1, 4 nodes
#    bash launch_worker.sh 02_train_cifar.py 2 4    # rank=2, 4 nodes
#    bash launch_worker.sh 02_train_cifar.py 3 4    # rank=3, 4 nodes
#
#  For 15-node university cluster (run on nodes 02 → 15):
#    bash launch_worker.sh 02_train_cifar.py $NODE_RANK 15
# =============================================================

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

SCRIPT=${1:-"01_smoke_test.py"}
NODE_RANK=${2:-1}
NNODES=${3:-2}
shift 3 2>/dev/null || shift 2 2>/dev/null || shift 1 2>/dev/null || true
EXTRA_ARGS="$@"
MASTER_ADDR=$(python3 -c "import config; print(config.MASTER_ADDR)")
MASTER_PORT=$(python3 -c "import config; print(config.MASTER_PORT)")

# Force Gloo to use the physical Ethernet interface (enp2s0) — NEVER lo!
export GLOO_SOCKET_IFNAME="enp2s0"
export TP_SOCKET_IFNAME="enp2s0"

echo ""
echo "=============================================="
echo "  WORKER NODE | rank=$NODE_RANK | world_size=$NNODES"
echo "  Script    : src/$SCRIPT"
echo "  Args      : $EXTRA_ARGS"
echo "  Master    : $MASTER_ADDR:$MASTER_PORT"
echo "  Interface : $GLOO_SOCKET_IFNAME"
echo "  Connecting..."
echo "=============================================="
echo ""

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=1 \
  --node-rank=$NODE_RANK \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  src/$SCRIPT $EXTRA_ARGS
