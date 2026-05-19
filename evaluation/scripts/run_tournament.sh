#!/opt/homebrew/bin/bash
# run_tournament.sh — Round-robin FoW Chess tournament with optimized time budgets
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LUDII_JAR="$PROJECT_DIR/ludii/Ludii-1.3.14.jar"
AGENTS_JAR="$PROJECT_DIR/ludii/agents/jars/agents.jar"
GAME_FILE="$PROJECT_DIR/ludii/FoW_Chess.lud"
OUTDIR="$PROJECT_DIR/ludii/evaluation/results"
LOG_FILE="$OUTDIR/games_log.csv"

NGAMES=100
MAX_PARALLEL=4
AGENTS_ARG=""
MAX_MOVES=400

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngames)       NGAMES="$2";        shift 2 ;;
        --parallel)     MAX_PARALLEL="$2";  shift 2 ;;
        --agents)       AGENTS_ARG="$2";    shift 2 ;;
        --max-moves)    MAX_MOVES="$2";     shift 2 ;;
        *)              echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Default full agent list; override with --agents "a,b,c"
if [[ -n "$AGENTS_ARG" ]]; then
    IFS=',' read -r -a AGENTS <<< "$AGENTS_ARG"
else
    AGENTS=("random" "uct" "pimc_uct" "ab_heuristic" "trained_uct" "ab_learned")
fi
JAVA_CP="${LUDII_JAR}:${AGENTS_JAR}"

# Per-matchup time budget function
get_time_budget() {
    local a="$1" b="$2"
    # Both random
    if [[ "$a" == "random" && "$b" == "random" ]]; then echo "0.001"; return; fi
    # One is random
    if [[ "$a" == "random" || "$b" == "random" ]]; then echo "0.05"; return; fi
    # Both UCT
    if [[ "$a" == "uct" && "$b" == "uct" ]]; then echo "0.1"; return; fi
    # trained_uct involved
    if [[ "$a" == "trained_uct" || "$b" == "trained_uct" ]]; then echo "1.0"; return; fi
    # pimc or ab_heuristic or ab_learned involved
    if [[ "$a" == "pimc_uct" || "$b" == "pimc_uct" \
       || "$a" == "ab_heuristic" || "$b" == "ab_heuristic" \
       || "$a" == "ab_learned"   || "$b" == "ab_learned" ]]; then
        echo "0.5"; return
    fi
    # Default (uct vs uct already handled, remaining: uct vs other)
    echo "0.2"
}
export -f get_time_budget

# Function to run a single matchup — exported for GNU parallel
run_one_matchup() {
    local PAIR="$1"
    local A="${PAIR%%:*}"
    local B="${PAIR##*:}"
    local NGAMES_LOCAL="$2"
    local JAVA_CP_LOCAL="$3"
    local GAME_FILE_LOCAL="$4"
    local OUTDIR_LOCAL="$5"
    local MAX_MOVES_LOCAL="$6"

    local PAIR_OUT="$OUTDIR_LOCAL/${A}_vs_${B}.csv"

    # Skip if already complete (>= NGAMES data rows)
    if [ -f "$PAIR_OUT" ] && [ "$(tail -n +2 "$PAIR_OUT" | wc -l | tr -d ' ')" -ge "$NGAMES_LOCAL" ]; then
        echo "  [SKIP] $A vs $B — already have $(tail -n +2 "$PAIR_OUT" | wc -l | tr -d ' ') games"
        return 0
    fi

    local TIME_BUDGET
    TIME_BUDGET="$(get_time_budget "$A" "$B")"

    echo "  [$A vs $B] time=${TIME_BUDGET}s, games=${NGAMES_LOCAL}, max-moves=${MAX_MOVES_LOCAL}"

    java -Xmx3g \
         -cp "$JAVA_CP_LOCAL" \
         agents.EvalRunner \
         --game          "$GAME_FILE_LOCAL" \
         --agent1        "$A" \
         --agent2        "$B" \
         --num-games     "$NGAMES_LOCAL" \
         --time-per-move "$TIME_BUDGET" \
         --max-moves     "$MAX_MOVES_LOCAL" \
         --output        "$PAIR_OUT" || echo "  [WARN] $A vs $B failed (continuing)"
}
export -f run_one_matchup

# Preflight
echo "============================================================"
echo " FoW Chess Tournament (optimized)"
echo " Agents: ${AGENTS[*]}"
echo " Games/pair/colour: $NGAMES"
echo " Max parallel: $MAX_PARALLEL"
echo "============================================================"

[[ ! -f "$LUDII_JAR"  ]] && { echo "ERROR: $LUDII_JAR not found";  exit 1; }
[[ ! -f "$AGENTS_JAR" ]] && { echo "ERROR: $AGENTS_JAR not found. Run build.sh first."; exit 1; }
[[ ! -f "$GAME_FILE"  ]] && { echo "ERROR: $GAME_FILE not found";  exit 1; }

mkdir -p "$OUTDIR/figures"

# Write CSV header
if [[ ! -f "$LOG_FILE" ]]; then
    echo "game_id,agent_p1,agent_p2,winner,num_moves,draw,timestamp_utc" > "$LOG_FILE"
fi

# Build matchup list
MATCHUP_LIST=()
for i in "${!AGENTS[@]}"; do
    for j in "${!AGENTS[@]}"; do
        [[ "$i" -eq "$j" ]] && continue
        MATCHUP_LIST+=("${AGENTS[$i]}:${AGENTS[$j]}")
    done
done

echo ""
echo "Total matchups: ${#MATCHUP_LIST[@]}  (${#MATCHUP_LIST[@]} × $NGAMES = $(( ${#MATCHUP_LIST[@]} * NGAMES )) games)"
echo ""

# Run matchups
if command -v parallel &>/dev/null; then
    echo "[parallel mode] Using GNU parallel with $MAX_PARALLEL jobs"
    printf '%s\n' "${MATCHUP_LIST[@]}" | \
        parallel --jobs "$MAX_PARALLEL" \
            run_one_matchup {} "$NGAMES" "$JAVA_CP" "$GAME_FILE" "$OUTDIR" "$MAX_MOVES"
else
    echo "[sequential mode] GNU parallel not found"
    for PAIR in "${MATCHUP_LIST[@]}"; do
        run_one_matchup "$PAIR" "$NGAMES" "$JAVA_CP" "$GAME_FILE" "$OUTDIR" "$MAX_MOVES"
    done
fi

# Merge results
echo ""
echo "Merging per-pair CSVs into master log..."
: > "$LOG_FILE"
echo "game_id,agent_p1,agent_p2,winner,num_moves,draw,timestamp_utc" > "$LOG_FILE"
for f in "$OUTDIR"/*_vs_*.csv; do
    [[ -f "$f" ]] && tail -n +2 "$f" >> "$LOG_FILE"
done
echo "Total games: $(( $(wc -l < "$LOG_FILE") - 1 ))"

# Analysis
echo ""
echo "[Analysis] Elo computation..."
python3 "$SCRIPT_DIR/compute_elo.py" \
    --input  "$LOG_FILE" \
    --output "$OUTDIR/elo_ratings.csv" \
    --matrix "$OUTDIR/win_rate_matrix.csv"

echo ""
echo "[Analysis] Significance tests..."
python3 "$SCRIPT_DIR/significance.py" \
    --input  "$LOG_FILE" \
    --output "$OUTDIR/results_summary.csv" \
    --elo    "$OUTDIR/elo_ratings.csv"

echo ""
echo "[Analysis] Strategic analysis..."
python3 "$SCRIPT_DIR/strategic_analysis.py" \
    --input  "$LOG_FILE" \
    --output "$OUTDIR/strategic_analysis.csv" \
    --outdir "$OUTDIR/figures"

echo ""
echo "[Analysis] Plots..."
python3 "$SCRIPT_DIR/plot_results.py" \
    --games  "$LOG_FILE" \
    --elo    "$OUTDIR/elo_ratings.csv" \
    --matrix "$OUTDIR/win_rate_matrix.csv" \
    --outdir "$OUTDIR/figures"

echo ""
echo "============================================================"
echo " Tournament complete! Results: $OUTDIR"
echo "============================================================"
