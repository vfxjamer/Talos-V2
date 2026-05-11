#!/bin/bash
# ============================================================================
# Talon V2 - Auto Resume Script for Kaggle (P100 GPU, 12hr Session Limit)
# ============================================================================
# Handles Kaggle's 12-hour session limit by:
# 1. Starting training with GPU_CUDA (P100)
# 2. Running until ~11h40m
# 3. Gracefully stopping and saving checkpoint
# 4. Auto-restarting from latest checkpoint
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MAX_RUNTIME=43200          # 12 hours (seconds)
SHUTDOWN_BUFFER=600        # Stop 10 min before limit
BOT_BINARY="./build/TalonV2Bot"
CHECKPOINT_DIR="checkpoints"

echo ""
echo "============================================================"
echo "  TALON V2 - Auto Resume for Kaggle (P100)"
echo "============================================================"
echo "  Max Session: $((MAX_RUNTIME/3600)) hours"
echo "  Device: GPU_CUDA (P100)"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "============================================================"
echo ""

# Create directories
mkdir -p "$CHECKPOINT_DIR"

# Get latest checkpoint step
get_latest_step() {
    if [ -d "$CHECKPOINT_DIR" ]; then
        ls -1d "$CHECKPOINT_DIR"/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1
    else
        echo "0"
    fi
}

# Check if bot is running
is_running() {
    pgrep -f "TalonV2Bot" >/dev/null 2>&1
}

# Graceful stop
stop_bot() {
    echo "[$(date '+%H:%M:%S')] Saving and stopping..."
    pkill -TERM -f "TalonV2Bot" 2>/dev/null || true

    # Wait for graceful exit (max 30s)
    for i in {1..30}; do
        if ! is_running; then
            echo "[$(date '+%H:%M:%S')] Stopped gracefully"
            return 0
        fi
        sleep 1
    done

    # Force kill if needed
    pkill -KILL -f "TalonV2Bot" 2>/dev/null || true
    sleep 2
    return 1
}

# Main training loop
SESSION=1
TOTAL_STEPS=$(get_latest_step)

echo "[INFO] Resuming from: $TOTAL_STEPS steps"
echo ""

# Ensure build exists
if [ ! -x "$BOT_BINARY" ]; then
    echo "[ERROR] Bot not found: $BOT_BINARY"
    echo "[INFO] Run kaggle_train.sh first to build"
    exit 1
fi

# Start tracking
START_TIME=$(date +%s)

# Start training
echo "[$(date '+%H:%M:%S')] Starting training..."
$BOT_BINARY > training.log 2>&1 &
BOT_PID=$!

echo "[$(date '+%H:%M:%S')] PID: $BOT_PID"

# Monitor loop
while is_running; do
    ELAPSED=$(($(date +%s) - START_TIME))
    REMAINING=$((MAX_RUNTIME - ELAPSED))

    # Progress every 5 min
    if [ $((ELAPSED % 300)) -eq 0 ]; then
        STEPS=$(get_latest_step)
        echo "[$(date '+%H:%M:%S')] Time: ${ELAPSED}s | Steps: $STEPS | Left: ${REMAINING}s"
    fi

    # Stop before session limit
    if [ $REMAINING -le $SHUTDOWN_BUFFER ]; then
        echo "[$(date '+%H:%M:%S')] Approaching limit, stopping..."
        stop_bot
        break
    fi

    sleep 30
done

# Get final stats
wait $BOT_PID 2>/dev/null || true
FINAL_STEPS=$(get_latest_step)
GAINED=$((FINAL_STEPS - TOTAL_STEPS))

echo ""
echo "[INFO] Session $SESSION: +$GAINED steps (total: $FINAL_STEPS)"

# Check if done (48B = 48000000000)
if [ $FINAL_STEPS -ge 48000000000 ]; then
    echo "[INFO] Target reached: 48B steps!"
    cat training.log | tail -20
    exit 0
fi

echo "[INFO] Auto-resuming for next session..."

# Wait before restart
sleep 10
SESSION=$((SESSION + 1))

echo ""
echo "============================================================"
echo "  Session $SESSION - Continuing from $FINAL_STEPS steps"
echo "============================================================"

# Restart
exec "$0" "$@"