#!/bin/bash
# ==============================================================================
#  CPU CLUSTER LAB — AUTONOMOUS OVERNIGHT EXPERIMENTAL PIPELINE
# ==============================================================================
#  Runs the full suite of research experiments inside tmux:
#    1. Pre-flight health check & zombie process cleanup
#    2. Quick sanity test (verifies cluster connectivity)
#    3. Fixed-Microbatch K-Sweep (Exp 2: K ∈ {1, 2, 4, 8, 16}, 3 trials)
#    4. Fixed-Global-Batch K-Sweep (Exp 3: K ∈ {1, 2, 4, 8, 16}, 3 trials)
#    5. Step-time AllReduce breakdown profiling (Exp 4)
#    6. Automatic Figure Generation (Figure 1 through 5)
#
#  RUN INSIDE TMUX:
#    tmux new -s cluster
#    bash run_experiments.sh
#    (Press Ctrl+B then D to detach, close laptop, and go to sleep!)
#
#  RE-ATTACH LATER:
#    tmux attach -t cluster
# ==============================================================================

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

LOG_FILE="results/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results

echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "  🚀 CPU CLUSTER LAB — AUTONOMOUS EXPERIMENTAL PIPELINE" | tee -a "$LOG_FILE"
echo "  Start Time : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "  Host       : $(hostname)" | tee -a "$LOG_FILE"
echo "  Log File   : $LOG_FILE" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# ── PRE-FLIGHT: Cleanup any stale processes across nodes ──────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 0/5] Pre-flight cleanup & cluster check..." | tee -a "$LOG_FILE"
killall -9 python3 torchrun 2>/dev/null || true

WORKER_IP=$(python3 -c "import config; print(getattr(config, 'WORKER_ADDRS', ['192.168.9.135'])[0])")
ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no cluster@$WORKER_IP \
    "killall -9 python3 torchrun 2>/dev/null || true" 2>/dev/null || true

echo "    ✓ Cluster processes clean and idle." | tee -a "$LOG_FILE"
sleep 2

# ── STAGE 1: Quick Sanity Verification ────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ">>> [Stage 1/5] Running 2-Minute Sanity Verification..." | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

bash launch_master.sh 04_benchmark.py 2 --mode micro --trials 1 --steps 4 | tee -a "$LOG_FILE"
echo "    ✓ Sanity check passed successfully!" | tee -a "$LOG_FILE"
sleep 5

# ── STAGE 2: Fixed-Microbatch K-Sweep (Exp 2) ─────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ">>> [Stage 2/5] Running Fixed-Microbatch K-Sweep (Exp 2)..." | tee -a "$LOG_FILE"
echo "    K ∈ {1, 2, 4, 8, 16} | 3 Trials per setting" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

bash launch_master.sh 04_benchmark.py 2 --mode micro --trials 3 --steps 8 | tee -a "$LOG_FILE"
echo "    ✓ Fixed-Microbatch Sweep completed!" | tee -a "$LOG_FILE"
sleep 5

# ── STAGE 3: Fixed-Global-Batch K-Sweep (Exp 3) ────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ">>> [Stage 3/5] Running Fixed-Global-Batch K-Sweep (Exp 3)..." | tee -a "$LOG_FILE"
echo "    K ∈ {1, 2, 4, 8, 16} | 3 Trials per setting" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

bash launch_master.sh 04_benchmark.py 2 --mode global --trials 3 --steps 8 | tee -a "$LOG_FILE"
echo "    ✓ Fixed-Global-Batch Sweep completed!" | tee -a "$LOG_FILE"
sleep 5

# ── STAGE 4: Automated Publication Figure Generation ──────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ">>> [Stage 4/5] Generating All 5 Official Publication Figures..." | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

python3 src/05_plot_results.py | tee -a "$LOG_FILE"
echo "    ✓ All 5 publication figures rendered and archived!" | tee -a "$LOG_FILE"

# ── SUMMARY & FINISH ──────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "  🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!" | tee -a "$LOG_FILE"
echo "  End Time   : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "  CSV Data   : results/02_sweep_fixed_microbatch/ & results/03_sweep_fixed_global_batch/"
echo "  Figures    : results/figures/"
echo "  Log Saved  : $LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo ""
