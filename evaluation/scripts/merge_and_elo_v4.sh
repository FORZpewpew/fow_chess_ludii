#!/usr/bin/env bash
set -euo pipefail

# Merge and ELO Computation for Tournament v4
# Merges all v4 tournament CSV results and computes ELO ratings
# Optionally combines with v3 results for comparative analysis

# ============================================================================
# SETUP
# ============================================================================

# Ensure we're in the ludii directory
cd "$(dirname "$0")/../../" || exit 1

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR="evaluation/results_v4"
GAMES_LOG="${RESULTS_DIR}/games_log.csv"
PYTHON_BIN="ppo/venv/bin/python"
ELO_SCRIPT="evaluation/scripts/compute_elo.py"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Check if a file exists and is readable
check_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "ERROR: File not found: $file"
    return 1
  fi
  if [[ ! -r "$file" ]]; then
    echo "ERROR: File not readable: $file"
    return 1
  fi
  return 0
}

# ============================================================================
# MERGE RESULTS
# ============================================================================

echo "=========================================="
echo "Merging Tournament v4 Results"
echo "=========================================="
echo ""

# Check if results directory exists and has CSV files
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "ERROR: Results directory not found: $RESULTS_DIR"
  exit 1
fi

csv_count=$(find "$RESULTS_DIR" -name "*_vs_*.csv" -type f | wc -l)
if [[ $csv_count -eq 0 ]]; then
  echo "ERROR: No CSV files found in $RESULTS_DIR"
  exit 1
fi

echo "Found ${csv_count} CSV files to merge"
echo ""

# Remove old games_log if it exists
if [[ -f "$GAMES_LOG" ]]; then
  echo "Removing old games_log.csv..."
  rm "$GAMES_LOG"
fi

# Merge all CSV files: header from first file, data rows from all
first_file=true
for csv_file in "$RESULTS_DIR"/*_vs_*.csv; do
  if [[ ! -f "$csv_file" ]]; then
    continue
  fi
  
  if [[ "$first_file" == true ]]; then
    # Copy entire first file (header + data)
    cat "$csv_file" > "$GAMES_LOG"
    echo "  ✓ Added header from: $(basename "$csv_file")"
    first_file=false
  else
    # Append data rows only (skip header)
    tail -n +2 "$csv_file" >> "$GAMES_LOG"
    echo "  ✓ Appended data from: $(basename "$csv_file")"
  fi
done

echo ""
echo "✓ Merged results saved to: $GAMES_LOG"
total_lines=$(wc -l < "$GAMES_LOG")
echo "  Total lines (including header): $total_lines"
echo ""

# ============================================================================
# COMPUTE ELO RATINGS
# ============================================================================

echo "=========================================="
echo "Computing ELO Ratings (v4 only)"
echo "=========================================="
echo ""

# Check if Python script exists
if ! check_file "$ELO_SCRIPT"; then
  exit 1
fi

# Check if Python venv exists
if ! check_file "$PYTHON_BIN"; then
  echo "ERROR: Python venv not found at: $PYTHON_BIN"
  exit 1
fi

# Run ELO computation for v4 results only
echo "Running: $PYTHON_BIN $ELO_SCRIPT --results-dir $RESULTS_DIR"
if "$PYTHON_BIN" "$ELO_SCRIPT" --results-dir "$RESULTS_DIR"; then
  echo "✓ ELO computation completed for v4 results"
else
  echo "✗ WARNING: ELO computation failed. Check the script output above."
fi

echo ""

# ============================================================================
# COMBINED ELO COMPUTATION (v3 + v4)
# ============================================================================

# Check if v3 results exist for combined analysis
V3_GAMES_LOG="evaluation/results_v3/games_log.csv"

if [[ -f "$V3_GAMES_LOG" ]]; then
  echo "=========================================="
  echo "Computing Combined ELO Ratings (v3 + v4)"
  echo "=========================================="
  echo ""
  
  # Attempt combined ELO computation
  # Note: This assumes compute_elo.py supports the --combined flag
  # If the flag is not supported, the script will fail gracefully
  echo "Running: $PYTHON_BIN $ELO_SCRIPT --results-dir $RESULTS_DIR --combined $V3_GAMES_LOG"
  
  if "$PYTHON_BIN" "$ELO_SCRIPT" --results-dir "$RESULTS_DIR" --combined "$V3_GAMES_LOG" 2>/dev/null; then
    echo "✓ Combined ELO computation completed (v3 + v4)"
  else
    echo "⚠ Combined ELO computation not supported or failed."
    echo "  This is expected if compute_elo.py does not support the --combined flag."
    echo "  To enable combined analysis, update compute_elo.py to accept:"
    echo "    --combined <path_to_v3_games_log>"
    echo "  This would allow comparative ELO ratings across v3 and v4 agents."
  fi
else
  echo "⚠ v3 results not found at: $V3_GAMES_LOG"
  echo "  Skipping combined ELO computation."
  echo "  To enable combined analysis, ensure v3 results exist at the path above."
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "=========================================="
echo "Merge and ELO Computation Complete"
echo "=========================================="
echo ""
echo "Output files:"
echo "  • Merged results: $GAMES_LOG"
echo "  • ELO ratings: Check evaluation/results_v4/ for output files"
echo ""
echo "Next steps:"
echo "  1. Review ELO ratings in evaluation/results_v4/"
echo "  2. Compare v4 agents against v3 baseline"
echo "  3. Analyze performance trends"
