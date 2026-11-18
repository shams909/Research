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

# Force Gloo to use the physical Ethernet interface instead of 127.0.0.1 loopback
IFACE=$(ip route get "$MASTER_ADDR" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1); exit}')
if [ -z "$IFACE" ]; then
    IFACE=$(ip -br link | grep -v 'lo' | awk '{print $1; exit}')
fi
export GLOO_SOCKET_IFNAME="$IFACE"
export TP_SOCKET_IFNAME="$IFACE"

echo ""
echo "=============================================="
echo "  MASTER NODE | rank=0 | world_size=$NNODES"
echo "  Script    : src/$SCRIPT"
echo "  Address   : $MASTER_ADDR:$MASTER_PORT"
echo "  Interface : $GLOO_SOCKET_IFNAME"
echo "  Waiting for $((NNODES-1)) worker(s)..."
echo "=============================================="
echo ""

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=1 \
  --node-rank=0 \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  src/$SCRIPT
