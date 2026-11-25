#!/bin/bash
# ==============================================================================
#  CPU CLUSTER LAB — VOICE MODALITY (WHISPER-TINY) EXPERIMENT PIPELINE
# ==============================================================================
#  Runs the EXACT same 2-node, 100 Mbps, 12-core benchmark & convergence
#  protocol as run_experiments.sh, scoped to Whisper-Tiny (--model whisper)
#  into an isolated session folder: results/session_<timestamp>_voice/
#  Leaves existing CIFAR-10 / GPT-2 session results completely untouched.
# ==============================================================================

cd "$(dirname "$0")"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TORCH_DIST_TIMEOUT_MINS=10
PIPELINE_START=$(date +%s)

format_duration() {
    local T=$1
    local H=$((T / 3600))
    local M=$(((T % 3600) / 60))
    local S=$((T % 60))
    printf "%02dh %02dm %02ds (%ds)" $H $M $S $T
}

# ── Isolated Voice Session Directory ──────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
SESSION_NAME="session_${TS}_${SESSION_TAG:-voice}"
SESSION_DIR="results/${SESSION_NAME}"
mkdir -p "$SESSION_DIR" "$SESSION_DIR/figures" "$SESSION_DIR/worker_logs"
export RESULTS_DIR="$SESSION_DIR"

rm -f results/latest 2>/dev/null || true
ln -sfn "${SESSION_NAME}" results/latest 2>/dev/null || true

LOG_FILE="${SESSION_DIR}/pipeline.log"
K_LIST=${K_LIST:-"1,2,4,8,16,32,64"}

ensure_telemetry() {
    if [ -z "$MONITOR_PID" ] || ! kill -0 "$MONITOR_PID" 2>/dev/null; then
        python3 src/06_system_monitor.py --interval 1.0 > /dev/null 2>&1 &
        MONITOR_PID=$!
    fi
}

cleanup_cluster() {
    echo "  [cleanup] Clearing stale cluster training processes and port 29500..." | tee -a "$LOG_FILE"
    pkill -9 -f "torchrun" 2>/dev/null || true
    pkill -9 -f "00_net_test.py" 2>/dev/null || true
    pkill -9 -f "04_benchmark.py" 2>/dev/null || true
    pkill -9 -f "07_train_whisper.py" 2>/dev/null || true
    fuser -k 29500/tcp 2>/dev/null || true
    WORKER_IP=$(python3 -c "import config; print(getattr(config, 'WORKER_ADDRS', ['192.168.9.135'])[0])" 2>/dev/null || echo "192.168.9.135")
    ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no cluster@$WORKER_IP \
        "pkill -9 -f 'torchrun' 2>/dev/null; pkill -9 -f '0[0-7]_' 2>/dev/null; fuser -k 29500/tcp 2>/dev/null || true" 2>/dev/null || true
    sleep 2
    ensure_telemetry
}

stop_telemetry() {
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
}
trap stop_telemetry EXIT

echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "  CPU CLUSTER LAB — VOICE MODALITY (WHISPER-TINY) PIPELINE" | tee -a "$LOG_FILE"
echo "  Start Time     : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "  Session Folder : $SESSION_DIR" | tee -a "$LOG_FILE"
echo "  K-List Sweep   : $K_LIST" | tee -a "$LOG_FILE"
echo "  Hardware       : Master (Node 01) + Worker (Node 02) over 100 Mbps Eth" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# ── STAGE 0: Pre-Flight Cleanup, Git Sync, Hardware Probe & Telemetry ─────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 0/6] Pre-flight cluster health check & hardware introspection..." | tee -a "$LOG_FILE"
cleanup_cluster
echo "    ✓ Cluster ports, processes & 1 Hz telemetry active (PID: $MONITOR_PID)." | tee -a "$LOG_FILE"

WORKER_IP=$(python3 -c "import config; print(getattr(config, 'WORKER_ADDRS', ['192.168.9.135'])[0])" 2>/dev/null || echo "192.168.9.135")
echo "  [sync] Syncing code on Worker Node ($WORKER_IP)..." | tee -a "$LOG_FILE"
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no cluster@$WORKER_IP \
    "cd ~/Research/CPU\ Cluster 2>/dev/null || cd ~/Research/'CPU Cluster' 2>/dev/null; git pull" 2>&1 | tee -a "$LOG_FILE" || true

python3 src/00_hardware_probe.py 2>&1 | tee -a "$LOG_FILE" || true

# ── STAGE 1: Quick Sanity Verification ────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 1/6] Running Whisper-Tiny Sanity Verification (1 Trial, 4 Steps)..." | tee -a "$LOG_FILE"
STAGE1_START=$(date +%s)
bash launch_master.sh 04_benchmark.py 2 --model whisper --mode micro --trials 1 --steps 4 2>&1 | tee -a "$LOG_FILE" || true
STAGE1_END=$(date +%s)
echo "    ⏱ Stage 1 Duration: $(format_duration $((STAGE1_END - STAGE1_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

# ── STAGE 2: Physical Interconnect Verification ───────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 2/6] Verifying Physical Network AllReduce Baseline..." | tee -a "$LOG_FILE"
STAGE2_START=$(date +%s)
bash launch_master.sh 00_net_test.py 2 2>&1 | tee -a "$LOG_FILE" || true
STAGE2_END=$(date +%s)
echo "    ⏱ Stage 2 Duration: $(format_duration $((STAGE2_END - STAGE2_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

# ── STAGE 3: Single-Node Whisper-Tiny Baseline ────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 3/6] Running Single-Node Whisper-Tiny Baseline (K=1,4)..." | tee -a "$LOG_FILE"
STAGE3_START=$(date +%s)
bash launch_single.sh 04_benchmark.py --model whisper --mode micro --k_list 1,4 --trials 2 --steps 4 2>&1 | tee -a "$LOG_FILE" || true
STAGE3_END=$(date +%s)
echo "    ⏱ Stage 3 Duration: $(format_duration $((STAGE3_END - STAGE3_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

# ── STAGE 4: Fixed-Microbatch K-Sweep (Whisper-Tiny) ──────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 4/6] Running Whisper-Tiny Fixed-Microbatch K-Sweep..." | tee -a "$LOG_FILE"
STAGE4_START=$(date +%s)
bash launch_master.sh 04_benchmark.py 2 --model whisper --mode micro --trials 3 --steps 6 --k_list "$K_LIST" 2>&1 | tee -a "$LOG_FILE" || true
STAGE4_END=$(date +%s)
echo "    ⏱ Stage 4 Duration: $(format_duration $((STAGE4_END - STAGE4_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

# ── STAGE 5: Fixed-Global-Batch K-Sweep (Whisper-Tiny) ────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 5/6] Running Whisper-Tiny Fixed-Global-Batch K-Sweep..." | tee -a "$LOG_FILE"
STAGE5_START=$(date +%s)
bash launch_master.sh 04_benchmark.py 2 --model whisper --mode global --trials 3 --steps 6 --k_list "$K_LIST" 2>&1 | tee -a "$LOG_FILE" || true
STAGE5_END=$(date +%s)
echo "    ⏱ Stage 5 Duration: $(format_duration $((STAGE5_END - STAGE5_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

# ── STAGE 5b: Speech Commands Convergence Verification ──────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 5b/6] Running Whisper-Tiny Speech Commands Convergence (15 steps)..." | tee -a "$LOG_FILE"
STAGE5B_START=$(date +%s)
bash launch_master.sh 07_train_whisper.py 2 --steps 15 2>&1 | tee -a "$LOG_FILE" || true
STAGE5B_END=$(date +%s)
echo "    ⏱ Stage 5b Duration: $(format_duration $((STAGE5B_END - STAGE5B_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

stop_telemetry
PIPELINE_END=$(date +%s)
TOTAL_SECS=$((PIPELINE_END - PIPELINE_START))

echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "  VOICE PIPELINE FINISHED!" | tee -a "$LOG_FILE"
echo "  End Time       : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "  Total Duration : $(format_duration $TOTAL_SECS)" | tee -a "$LOG_FILE"
echo "  Session Folder : $SESSION_DIR" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
