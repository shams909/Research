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
set -o pipefail

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

# ── STAGE 0: Pre-Flight Cleanup, Git Sync, Dataset Check & Hardware Probe ─────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 0/6] Pre-flight cluster health check & dataset verification..." | tee -a "$LOG_FILE"
WORKER_IP=$(python3 -c "import config; print(getattr(config, 'WORKER_ADDRS', ['192.168.9.135'])[0])" 2>/dev/null || echo "192.168.9.135")
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no cluster@"$WORKER_IP" "echo ok" >/dev/null 2>&1; then
    echo "  ❌ ERROR: Cannot SSH to Worker Node (cluster@$WORKER_IP). Ensure Node 02 is powered on and connected." | tee -a "$LOG_FILE"
    exit 1
fi

cleanup_cluster
echo "    ✓ Cluster ports, processes & 1 Hz telemetry active (PID: $MONITOR_PID)." | tee -a "$LOG_FILE"

echo "  [sync] Syncing code on Worker Node ($WORKER_IP)..." | tee -a "$LOG_FILE"
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no cluster@"$WORKER_IP" \
    "cd ~/Research/CPU\ Cluster 2>/dev/null || cd ~/Research/'CPU Cluster' 2>/dev/null; git pull" 2>&1 | tee -a "$LOG_FILE"

echo "  [data] Verifying SpeechCommands & Whisper-Tiny cache on Master (Node 01)..." | tee -a "$LOG_FILE"
if [ ! -d "data/SpeechCommands/speech_commands_v0.02" ] || ! python3 -c "from transformers import WhisperTokenizer; WhisperTokenizer.from_pretrained('openai/whisper-tiny', local_files_only=True)" >/dev/null 2>&1; then
    echo "  [data] Downloading Voice dataset/tokenizer on Master (Node 01)..." | tee -a "$LOG_FILE"
    bash prepare_data.sh 2>&1 | tee -a "$LOG_FILE" || { echo "  ❌ prepare_data.sh failed on Master"; exit 1; }
fi

echo "  [data] Verifying SpeechCommands & Whisper-Tiny cache on Worker (Node 02)..." | tee -a "$LOG_FILE"
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no cluster@"$WORKER_IP" "
    cd ~/Research/CPU\ Cluster 2>/dev/null || cd ~/Research/'CPU Cluster'
    source .venv/bin/activate
    if [ ! -d 'data/SpeechCommands/speech_commands_v0.02' ] || ! python3 -c \"from transformers import WhisperTokenizer; WhisperTokenizer.from_pretrained('openai/whisper-tiny', local_files_only=True)\" >/dev/null 2>&1; then
        echo '  [data] Downloading Voice dataset/tokenizer on Worker (Node 02)...'
        bash prepare_data.sh
    fi
" 2>&1 | tee -a "$LOG_FILE" || { echo "  ❌ prepare_data.sh failed on Worker ($WORKER_IP)"; exit 1; }

python3 src/00_hardware_probe.py 2>&1 | tee -a "$LOG_FILE" || true

# ── STAGE 1: Fail-Fast Smoke Verification (Distributed Benchmark + Audio) ──────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 1/6] Running Whisper-Tiny Fail-Fast Smoke Check (K=1, 2 Steps)..." | tee -a "$LOG_FILE"
STAGE1_START=$(date +%s)
if ! bash launch_master.sh 04_benchmark.py 2 --model whisper --mode micro --k_list 1 --trials 1 --steps 2 2>&1 | tee -a "$LOG_FILE"; then
    echo "  ❌ ERROR: Stage 1 Whisper-Tiny distributed smoke test failed! Check worker logs in ${SESSION_DIR}/worker_logs/." | tee -a "$LOG_FILE"
    exit 1
fi
if ! python3 -c "import sys; sys.path.insert(0, 'src'); import importlib.util; spec = importlib.util.spec_from_file_location('w', 'src/07_train_whisper.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); ds = m.SpeechCommandsWhisperDataset(dec_len=32, rank=0); x, y = ds[0]; print('    ✓ Verified SpeechCommands sample:', tuple(x.shape), tuple(y.shape))" 2>&1 | tee -a "$LOG_FILE"; then
    echo "  ❌ ERROR: SpeechCommands audio loader check failed on Master!" | tee -a "$LOG_FILE"
    exit 1
fi
STAGE1_END=$(date +%s)
echo "    ⏱ Stage 1 Duration: $(format_duration $((STAGE1_END - STAGE1_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

# ── STAGE 2: Physical Interconnect Verification ───────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 2/6] Verifying Physical Network AllReduce Baseline..." | tee -a "$LOG_FILE"
STAGE2_START=$(date +%s)
bash launch_master.sh 00_net_test.py 2 2>&1 | tee -a "$LOG_FILE" || echo "  ⚠ Stage 2 net_test exited non-zero (continuing)" | tee -a "$LOG_FILE"
STAGE2_END=$(date +%s)
echo "    ⏱ Stage 2 Duration: $(format_duration $((STAGE2_END - STAGE2_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

# ── STAGE 3: Single-Node Whisper-Tiny Baseline ────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 3/6] Running Single-Node Whisper-Tiny Baseline (K=1,4)..." | tee -a "$LOG_FILE"
STAGE3_START=$(date +%s)
bash launch_single.sh 04_benchmark.py --model whisper --mode micro --k_list 1,4 --trials 2 --steps 4 2>&1 | tee -a "$LOG_FILE" || echo "  ⚠ Stage 3 single-node exited non-zero (continuing)" | tee -a "$LOG_FILE"
STAGE3_END=$(date +%s)
echo "    ⏱ Stage 3 Duration: $(format_duration $((STAGE3_END - STAGE3_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

# ── STAGE 4: Fixed-Microbatch K-Sweep (Whisper-Tiny) ──────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 4/6] Running Whisper-Tiny Fixed-Microbatch K-Sweep..." | tee -a "$LOG_FILE"
STAGE4_START=$(date +%s)
bash launch_master.sh 04_benchmark.py 2 --model whisper --mode micro --trials 3 --steps 6 --k_list "$K_LIST" 2>&1 | tee -a "$LOG_FILE" || echo "  ⚠ Stage 4 micro sweep exited non-zero (continuing)" | tee -a "$LOG_FILE"
STAGE4_END=$(date +%s)
echo "    ⏱ Stage 4 Duration: $(format_duration $((STAGE4_END - STAGE4_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

# ── STAGE 5: Fixed-Global-Batch K-Sweep (Whisper-Tiny) ────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 5/6] Running Whisper-Tiny Fixed-Global-Batch K-Sweep..." | tee -a "$LOG_FILE"
STAGE5_START=$(date +%s)
bash launch_master.sh 04_benchmark.py 2 --model whisper --mode global --trials 3 --steps 6 --k_list "$K_LIST" 2>&1 | tee -a "$LOG_FILE" || echo "  ⚠ Stage 5 global sweep exited non-zero (continuing)" | tee -a "$LOG_FILE"
STAGE5_END=$(date +%s)
echo "    ⏱ Stage 5 Duration: $(format_duration $((STAGE5_END - STAGE5_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

# ── STAGE 5b: Speech Commands Convergence Verification ──────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 5b/6] Running Whisper-Tiny Speech Commands Convergence (15 steps)..." | tee -a "$LOG_FILE"
STAGE5B_START=$(date +%s)
bash launch_master.sh 07_train_whisper.py 2 --steps 15 2>&1 | tee -a "$LOG_FILE" || echo "  ⚠ Stage 5b convergence exited non-zero (continuing)" | tee -a "$LOG_FILE"
STAGE5B_END=$(date +%s)
echo "    ⏱ Stage 5b Duration: $(format_duration $((STAGE5B_END - STAGE5B_START)))" | tee -a "$LOG_FILE"
cleanup_cluster

stop_telemetry

# ── STAGE 6: Auto-Generate Publication Figures ────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [Stage 6/6] Auto-generating Whisper-Tiny publication figures..." | tee -a "$LOG_FILE"
STAGE6_START=$(date +%s)
if python3 src/05_plot_results.py --results_dir "$SESSION_DIR" 2>&1 | tee -a "$LOG_FILE"; then
    echo "    ✓ Figures auto-generated in: ${SESSION_DIR}/figures/" | tee -a "$LOG_FILE"
else
    echo "    ⚠ Figure generation encountered an issue (CSVs in $SESSION_DIR are intact)." | tee -a "$LOG_FILE"
fi
STAGE6_END=$(date +%s)
echo "    ⏱ Stage 6 Duration: $(format_duration $((STAGE6_END - STAGE6_START)))" | tee -a "$LOG_FILE"

PIPELINE_END=$(date +%s)
TOTAL_SECS=$((PIPELINE_END - PIPELINE_START))

echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "  VOICE PIPELINE FINISHED!" | tee -a "$LOG_FILE"
echo "  End Time       : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "  Total Duration : $(format_duration $TOTAL_SECS)" | tee -a "$LOG_FILE"
echo "  Session Folder : $SESSION_DIR" | tee -a "$LOG_FILE"
echo "  Figures Folder : ${SESSION_DIR}/figures/" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
