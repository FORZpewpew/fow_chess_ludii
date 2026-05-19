#!/opt/homebrew/bin/bash
set -euo pipefail

# Tournament GRAVE Runner for Fog of War Chess
# Evaluates 2 new GRAVE agents against 6 key agents and each other
# Total: 26 matchups (all ordered pairs, both directions)
# Note: ab_heuristic and ab_learned excluded from thesis

# ============================================================================
# SETUP
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LUDII_DIR="$PROJECT_DIR/ludii"

# Create results directory
mkdir -p "$LUDII_DIR/evaluation/results_grave"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Key existing agents (6 agents, ab_heuristic and ab_learned excluded)
KEY_AGENTS=(
  "random"
  "uct"
  "ismcts"
  "ismcts_v4"
  "ppo_lstm_v4"
  "ppo_lstm_pretrained_v4"
)

# GRAVE agents (2 new agents)
GRAVE_AGENTS=(
  "grave"
  "grave_mast"
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
  
  local output_file="$LUDII_DIR/evaluation/results_grave/${agent1}_vs_${agent2}.csv"
  
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
      --num-games "$GAMES_PER_MATCHUP" \
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
echo "Tournament GRAVE: Fog of War Chess Evaluation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Games per matchup: ${GAMES_PER_MATCHUP}"
echo "  Key agents: ${#KEY_AGENTS[@]}"
echo "  GRAVE agents: ${#GRAVE_AGENTS[@]}"
echo "  Total matchups: 26 (12 GRAVE vs key + 2 GRAVE vs GRAVE, both directions)"
echo "  Results dir: $LUDII_DIR/evaluation/results_grave"
echo ""

# Build list of all matchups
declare -a MATCHUPS

# GRAVE agents vs key agents (both directions)
for grave_agent in "${GRAVE_AGENTS[@]}"; do
  for key_agent in "${KEY_AGENTS[@]}"; do
    MATCHUPS+=("${grave_agent}|${key_agent}")
    MATCHUPS+=("${key_agent}|${grave_agent}")
  done
done

# GRAVE agents vs each other (both directions)
for ((i=0; i<${#GRAVE_AGENTS[@]}; i++)); do
  for ((j=i+1; j<${#GRAVE_AGENTS[@]}; j++)); do
    MATCHUPS+=("${GRAVE_AGENTS[$i]}|${GRAVE_AGENTS[$j]}")
    MATCHUPS+=("${GRAVE_AGENTS[$j]}|${GRAVE_AGENTS[$i]}")
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
echo "Tournament GRAVE Complete"
echo "=========================================="

# Count completed CSV files
completed_count=$(find "$LUDII_DIR/evaluation/results_grave" -name "*_vs_*.csv" -type f | wc -l)
echo "Completed CSV files in $LUDII_DIR/evaluation/results_grave/: ${completed_count}"

if [[ $completed_count -eq $total_matchups ]]; then
  echo "✓ All ${total_matchups} matchups completed successfully!"
else
  echo "⚠ Only ${completed_count}/${total_matchups} matchups completed."
fi

echo ""
echo "Next steps:"
echo "  1. Check results in: evaluation/results_grave/"
echo "  2. Analyze tournament results"
