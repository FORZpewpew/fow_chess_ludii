#!/usr/bin/env bash
# Monitors training completion and auto-launches Tournament v4

LUDII_DIR="/Users/forzpewpew/Downloads/ludii"
LOG1="$LUDII_DIR/ppo/logs/train_ppo_lstm_v4.log"
LOG2="$LUDII_DIR/ppo/logs/pretrain_v4.log"
POLL_INTERVAL=600  # 10 minutes

check_done_v4() {
    grep -qE "Update\s+400/400" "$LOG1" 2>/dev/null
}

check_done_pretrained() {
    grep -qE "Update\s+300/300" "$LOG2" 2>/dev/null
}

echo "[monitor] Starting polling every ${POLL_INTERVAL}s ..."

while true; do
    V4_DONE=false
    PRE_DONE=false
    
    check_done_v4 && V4_DONE=true
    check_done_pretrained && PRE_DONE=true
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ppo_lstm_v4: $V4_DONE | ppo_lstm_pretrained_v4: $PRE_DONE"
    
    if $V4_DONE && $PRE_DONE; then
        echo "[monitor] Both trainings complete! Preparing tournament v4..."
        break
    fi
    
    sleep $POLL_INTERVAL
done

# Verify and copy checkpoints
echo "[monitor] Verifying checkpoints..."
ls -lh "$LUDII_DIR/checkpoints/ppo_lstm_v4_policy.pt" || { echo "ERROR: ppo_lstm_v4_policy.pt missing!"; exit 1; }
ls -lh "$LUDII_DIR/ppo/checkpoints/ppo_lstm_pretrained_v4.pt" || { echo "ERROR: ppo_lstm_pretrained_v4.pt missing!"; exit 1; }

# Copy pretrained checkpoint to main checkpoints dir (EvalRunner expects it at checkpoints/ OR ppo/checkpoints/)
cp "$LUDII_DIR/ppo/checkpoints/ppo_lstm_pretrained_v4.pt" "$LUDII_DIR/checkpoints/ppo_lstm_pretrained_v4.pt"
echo "[monitor] Copied ppo_lstm_pretrained_v4.pt to checkpoints/"

# Launch tournament v4
echo "[monitor] Launching tournament v4..."
cd "$LUDII_DIR"
mkdir -p evaluation/results_v4
bash evaluation/scripts/run_tournament_v4.sh 2>&1 | tee evaluation/logs/tournament_v4.log

echo "[monitor] Tournament v4 complete! Running Elo analysis..."
bash evaluation/scripts/merge_and_elo_v4.sh 2>&1 | tee evaluation/logs/elo_v4.log

echo "[monitor] ALL DONE. Results in evaluation/results_v4/"
