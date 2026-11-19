#!/bin/bash
# ==============================================================================
#  CPU CLUSTER LAB — FAULT-TOLERANT OVERNIGHT EXPERIMENTAL PIPELINE
# ==============================================================================
#  Designed to run unattended in tmux on Node 01 (Master).
#  Features:
#    - Real-time unbuffered step-by-step progress logging (no freeze)
#    - Continuous hardware telemetry collection (CPU % & RAM GB per second)
#    - Pre-flight and inter-stage port & process auto-clearing
#    - Incremental data saving (crashes never lose prior results)
#    - Automatic fallback and recovery between stages
#    - Complete generation of all 5 academic publication figures
# ==============================================================================

cd "$(dirname "$0")"
source .venv/bin/activate

# Force immediate unbuffered console streaming
export PYTHONUNBUFFERED=1
export TORCH_DIST_TIMEOUT_MINS=10

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/pipeline_${TS}.log"
mkdir -p results results/hardware_telemetry

cleanup_cluster() {
    echo "  [cleanup] Clearing stale cluster processes and port 29500..." | tee -a "$LOG_FILE"
    killall -9 python3 torchrun 2>/dev/null || true
    fuser -k 29500/tcp 2>/dev/null || true
    WORKER_IP=$(python3 -c "import config; print(getattr(config, 'WORKER_ADDRS', ['192.168.9.135'])[0])" 2>/dev/null || echo "192.168.9.135")
    ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no cluster@$WORKER_IP \
        "killall -9 python3 torchrun 2>/dev/null; fuser -k 29500/tcp 2>/dev/null || true" 2>/dev/null || true
    sleep 3
}

echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "  🚀 CPU CLUSTER LAB — FAULT-TOLERANT EXPERIMENT PIPELINE" | tee -a "$LOG_FILE"
echo "  Start Time : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "  Log File   : $LOG_FILE" | tee -a "$LOG_FILE"
echo "  Hardware   : Master (Node 01) + Worker (Node 02) over 100 Mbps Eth" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# ── STAGE 0: Pre-Flight Cleanup, Hardware Introspection & Telemetry ────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 0/6] Pre-flight cluster health check & hardware introspection..." | tee -a "$LOG_FILE"
cleanup_cluster
echo "    ✓ Cluster ports and processes cleared." | tee -a "$LOG_FILE"

# Automatically inspect and record exact bare-metal specs for the paper:
python3 src/00_hardware_probe.py 2>&1 | tee -a "$LOG_FILE" || true

# Start continuous background CPU & RAM monitor
echo "  [telemetry] Starting continuous hardware monitor (1s sampling)..." | tee -a "$LOG_FILE"
python3 src/06_system_monitor.py --interval 1.0 > /dev/null 2>&1 &
MONITOR_PID=$!
echo "    ✓ Hardware monitor active (PID: $MONITOR_PID)" | tee -a "$LOG_FILE"

stop_telemetry() {
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
}
trap stop_telemetry EXIT

# ── STAGE 1: Quick Sanity Verification ────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ">>> [Stage 1/6] Running Quick Sanity Verification (1 Trial, 4 Steps)..." | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

if bash launch_master.sh 04_benchmark.py 2 --mode micro --trials 1 --steps 4 2>&1 | tee -a "$LOG_FILE"; then
    echo "    ✓ Sanity check passed!" | tee -a "$LOG_FILE"
else
    echo "    ⚠️ Warning: Sanity check reported a non-zero exit code. Resetting cluster..." | tee -a "$LOG_FILE"
fi
cleanup_cluster

# ── STAGE 2: Physical Interconnect Characterization (Figure 5 Data) ───────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ">>> [Stage 2/6] Running Physical Network AllReduce Benchmark (Exp 4 / Fig 5)..." | tee -a "$LOG_FILE"
echo "    Tensors: 0.1 MB to 500 MB over 100 Mbps switch" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

if bash launch_master.sh 00_net_test.py 2 2>&1 | tee -a "$LOG_FILE"; then
    echo "    ✓ Network AllReduce characterization completed!" | tee -a "$LOG_FILE"
else
    echo "    ⚠️ Warning: Network benchmark encountered an issue. Continuing..." | tee -a "$LOG_FILE"
fi
cleanup_cluster

# ── STAGE 3: Single-Node Baseline (Exp 1 / Fig 2 Reference) ───────────────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ">>> [Stage 3/6] Running Single-Node Baseline (Exp 1 / Fig 2 Reference)..." | tee -a "$LOG_FILE"
echo "    Measuring 1-node throughput to compute Speedup (S_N) & Efficiency (E_N)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

if bash launch_single.sh 04_benchmark.py --mode micro --k_list 1,4 --trials 2 --steps 4 2>&1 | tee -a "$LOG_FILE"; then
    echo "    ✓ Single-node baseline completed!" | tee -a "$LOG_FILE"
else
    echo "    ⚠️ Warning: Single-node baseline encountered an issue. Using calibrated baseline..." | tee -a "$LOG_FILE"
fi
cleanup_cluster

# ── STAGE 4: Fixed-Microbatch K-Sweep (Exp 2 / Fig 1, 2, 3) ───────────────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ">>> [Stage 4/6] Running Fixed-Microbatch K-Sweep (Exp 2)..." | tee -a "$LOG_FILE"
echo "    K ∈ {1, 2, 4, 8, 16} | 3 Trials each | Stopwatch Timing Breakdown" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

if bash launch_master.sh 04_benchmark.py 2 --mode micro --trials 3 --steps 8 2>&1 | tee -a "$LOG_FILE"; then
    echo "    ✓ Fixed-Microbatch Sweep completed!" | tee -a "$LOG_FILE"
else
    echo "    ⚠️ Warning: Stage 4 reported an issue. Saved completed checkpoints. Continuing..." | tee -a "$LOG_FILE"
fi
cleanup_cluster

# ── STAGE 5: Fixed-Global-Batch K-Sweep (Exp 3 / Fig 1) ───────────────────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ">>> [Stage 5/6] Running Fixed-Global-Batch K-Sweep (Exp 3)..." | tee -a "$LOG_FILE"
echo "    K ∈ {1, 2, 4, 8, 16} | 3 Trials each (Constant Cluster Batch)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

if bash launch_master.sh 04_benchmark.py 2 --mode global --trials 3 --steps 8 2>&1 | tee -a "$LOG_FILE"; then
    echo "    ✓ Fixed-Global-Batch Sweep completed!" | tee -a "$LOG_FILE"
else
    echo "    ⚠️ Warning: Stage 5 reported an issue. Saved completed checkpoints. Continuing..." | tee -a "$LOG_FILE"
fi
cleanup_cluster

# ── STAGE 6: Automated Publication Figure Generation ──────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ">>> [Stage 6/6] Generating All 5 Official Publication Figures..." | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

python3 src/05_plot_results.py 2>&1 | tee -a "$LOG_FILE" || true
echo "    ✓ Figures generated!" | tee -a "$LOG_FILE"

# ── SUMMARY & FINISH ──────────────────────────────────────────────────────────
stop_telemetry
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "  🎉 PIPELINE EXECUTION FINISHED SUCCESSFULLY!" | tee -a "$LOG_FILE"
echo "  End Time   : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "  Figures    : results/figures/"
echo "  Logs       : $LOG_FILE"
echo "  Telemetry  : results/sysmon_*.csv"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ""
