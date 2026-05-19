#!/opt/homebrew/bin/bash
set -euo pipefail

# Tournament v4 Runner for Fog of War Chess (Dark Chess)
# Evaluates 3 new v4 agents against 8 v3 agents and each other
# Total: 54 matchups (all ordered pairs, both directions)

# ============================================================================
# SETUP
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LUDII_DIR="$PROJECT_DIR/ludii"

# Create results directory
mkdir -p "$LUDII_DIR/evaluation/results_v4"

# ============================================================================
# CONFIGURATION
# ============================================================================

# v3 agents (8 existing agents)
V3_AGENTS=(
  "ab_heuristic"
  "ab_learned"
  "uct"
  "ismcts"
  "ppo"
  "ppo_lstm"
  "ppo_lstm_pretrained"
  "random"
)

# v4 agents (3 new agents)
V4_AGENTS=(
  "ppo_lstm_v4"
  "ppo_lstm_pretrained_v4"
  "ismcts_v4"
)

# Games per matchup
GAMES_PER_MATCHUP=100

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Check if a CSV file already has results (more than 1 line = header + data)
has_results() {
  local csv_file="$1"
  if [[ ! -f "$csv_file" ]]; then
    return 1  # File doesn't exist
  fi
  local line_count
  line_count=$(wc -l < "$csv_file")
  [[ $line_count -gt 1 ]]  # Return true if more than 1 line
}

# Run a single matchup
run_matchup() {
  local agent1="$1"
  local agent2="$2"
  local counter="$3"
  local total="$4"
  
  local output_file="$LUDII_DIR/evaluation/results_v4/${agent1}_vs_${agent2}.csv"
  
  # Skip if already completed
  if has_results "$output_file"; then
    echo "[${counter}/${total}] SKIPPED (already completed): ${agent1} vs ${agent2}"
    return 0
  fi
  
  echo "[${counter}/${total}] Running ${agent1} vs ${agent2}..."
  
  # Run the evaluation from ludii/ directory so classpath and game paths resolve correctly
  (
    cd "$LUDII_DIR"
    if java -Xmx3g -cp "Ludii-1.3.14.jar:agents/agents.jar" agents.EvalRunner \
      --game "FoW_Chess.lud" \
      --agent1 "$agent1" \
      --agent2 "$agent2" \
      --games "$GAMES_PER_MATCHUP" \
      --output "$output_file"; then
      echo "  ✓ Completed: ${agent1} vs ${agent2}"
    else
      echo "  ✗ WARNING: java exited with non-zero status for ${agent1} vs ${agent2}. Continuing..."
      return 0  # Continue to next matchup despite error
    fi
  )
}

# ============================================================================
# MAIN TOURNAMENT EXECUTION
# ============================================================================

echo "=========================================="
echo "Tournament v4: Fog of War Chess Evaluation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Games per matchup: ${GAMES_PER_MATCHUP}"
echo "  v3 agents: ${#V3_AGENTS[@]}"
echo "  v4 agents: ${#V4_AGENTS[@]}"
echo "  Total matchups: 54 (24 v4 vs v3 + 6 v4 vs v4, both directions)"
echo "  Results dir: $LUDII_DIR/evaluation/results_v4"
echo ""

# Build list of all matchups
declare -a MATCHUPS

# v4 agents vs v3 agents (both directions)
for v4_agent in "${V4_AGENTS[@]}"; do
  for v3_agent in "${V3_AGENTS[@]}"; do
    MATCHUPS+=("${v4_agent}|${v3_agent}")
    MATCHUPS+=("${v3_agent}|${v4_agent}")
  done
done

# v4 agents vs each other (both directions)
for ((i=0; i<${#V4_AGENTS[@]}; i++)); do
  for ((j=i+1; j<${#V4_AGENTS[@]}; j++)); do
    MATCHUPS+=("${V4_AGENTS[$i]}|${V4_AGENTS[$j]}")
    MATCHUPS+=("${V4_AGENTS[$j]}|${V4_AGENTS[$i]}")
  done
done

total_matchups=${#MATCHUPS[@]}
echo "Total matchups to run: ${total_matchups}"
echo ""

# Execute all matchups sequentially
counter=1
for matchup in "${MATCHUPS[@]}"; do
  IFS='|' read -r agent1 agent2 <<< "$matchup"
  run_matchup "$agent1" "$agent2" "$counter" "$total_matchups"
  ((counter++))
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=========================================="
echo "Tournament v4 Complete"
echo "=========================================="

# Count completed CSV files
completed_count=$(find "$LUDII_DIR/evaluation/results_v4" -name "*_vs_*.csv" -type f | wc -l)
echo "Completed CSV files in $LUDII_DIR/evaluation/results_v4/: ${completed_count}"

if [[ $completed_count -eq $total_matchups ]]; then
  echo "✓ All ${total_matchups} matchups completed successfully!"
else
  echo "⚠ Only ${completed_count}/${total_matchups} matchups completed."
fi

echo ""
echo "Next steps:"
echo "  1. Run: ./evaluation/scripts/merge_and_elo_v4.sh"
echo "  2. Check results in: evaluation/results_v4/"
