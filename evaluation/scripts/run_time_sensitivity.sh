#!/usr/bin/env bash
# run_time_sensitivity.sh — Think-time sensitivity experiment
# Tests how search agents perform at 2s and 5s vs the baseline 1s tournament
# Runs in parallel with the main GRAVE tournament (different output dir)

set -euo pipefail

LUDII_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
JAR="$LUDII_DIR/agents/agents.jar"
LUDII_JAR="$LUDII_DIR/Ludii-1.3.14.jar"
GAME="FoW_Chess.lud"
GAMES_PER_MATCHUP=50          # Fewer games since we run at multiple time controls
MAX_MOVES=300
RESULTS_DIR="/Users/forzpewpew/Downloads/ludii/evaluation/results_time_sensitivity"
LOG_DIR="$LUDII_DIR/evaluation/logs"

mkdir -p "$RESULTS_DIR"
echo "Results will be written to: $RESULTS_DIR"
mkdir -p "$LOG_DIR"

cd "$LUDII_DIR"

# Agents included: only search-based (time-sensitive) + PPO-LSTM pair for comparison
# PPO-LSTM pair included to show time-invariance in the thesis
SEARCH_AGENTS=(uct ismcts ismcts_v4 grave grave_mast)

# Key matchups to run at each time control:
# All ordered pairs among search agents + one PPO-LSTM pair to show time-invariance
declare -a MATCHUPS=(
  "uct ismcts"
  "ismcts uct"
  "uct ismcts_v4"
  "ismcts_v4 uct"
  "uct grave"
  "grave uct"
  "uct grave_mast"
  "grave_mast uct"
  "ismcts grave"
  "grave ismcts"
  "ismcts_v4 grave_mast"
  "grave_mast ismcts_v4"
  "grave grave_mast"
  "grave_mast grave"
  "ppo_lstm_pretrained_v4 uct"
  "uct ppo_lstm_pretrained_v4"
)

TIME_CONTROLS=(2.0 5.0)

run_matchup() {
  local p1="$1"
  local p2="$2"
  local think_time="$3"
  local time_tag="${think_time%.*}s"   # e.g. "2.0" -> "2s"
  local out_csv="$RESULTS_DIR/${time_tag}_${p1}_vs_${p2}.csv"

  if [[ -f "$out_csv" ]] && [[ $(wc -l < "$out_csv") -gt 1 ]]; then
    echo "  SKIPPED (already complete): $p1 vs $p2 at ${think_time}s"
    return
  fi

  echo "  Running [$p1 vs $p2 @ ${think_time}s/move]..."
  java -cp "$JAR:$LUDII_JAR" agents.EvalRunner \
    --game "$LUDII_DIR/$GAME" \
    --agent1 "$p1" \
    --agent2 "$p2" \
    --num-games "$GAMES_PER_MATCHUP" \
    --time-per-move "$think_time" \
    --max-moves "$MAX_MOVES" \
    --output "$out_csv"
  if [[ -f "$out_csv" ]]; then
    echo "  DONE: $p1 vs $p2 at ${think_time}s -> $(wc -l < "$out_csv") lines written to $out_csv"
  else
    echo "  ERROR: Output file not created: $out_csv"
  fi
}

TOTAL_MATCHUPS=$(( ${#MATCHUPS[@]} * ${#TIME_CONTROLS[@]} ))
echo "Think-time sensitivity experiment"
echo "Total matchups to run: $TOTAL_MATCHUPS (${#MATCHUPS[@]} pairs × ${#TIME_CONTROLS[@]} time controls)"
echo ""

idx=0
for think_time in "${TIME_CONTROLS[@]}"; do
  echo "=== Time control: ${think_time}s/move ==="
  for matchup in "${MATCHUPS[@]}"; do
    idx=$((idx + 1))
    p1=$(echo "$matchup" | cut -d' ' -f1)
    p2=$(echo "$matchup" | cut -d' ' -f2)
    echo "[${idx}/${TOTAL_MATCHUPS}] ${p1} vs ${p2} @ ${think_time}s"
    run_matchup "$p1" "$p2" "$think_time"
  done
done

echo ""
echo "All $TOTAL_MATCHUPS matchups complete."
echo "Results saved to: $RESULTS_DIR"
