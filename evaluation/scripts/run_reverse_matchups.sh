#!/opt/homebrew/bin/bash
# run_reverse_matchups.sh — Run the 12 reverse-pair matchups for tournament v3
# These are the 6 base agents playing as P1 against the 2 LSTM agents
# Missing pairs where base agents are P1:
#   ab_heuristic vs ppo_lstm, ab_heuristic vs ppo_lstm_pretrained
#   ab_learned vs ppo_lstm, ab_learned vs ppo_lstm_pretrained
#   ismcts vs ppo_lstm, ismcts vs ppo_lstm_pretrained
#   ppo vs ppo_lstm, ppo vs ppo_lstm_pretrained
#   random vs ppo_lstm, random vs ppo_lstm_pretrained
#   uct vs ppo_lstm, uct vs ppo_lstm_pretrained

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LUDII_DIR="$PROJECT_DIR/ludii"

LUDII_JAR="$LUDII_DIR/Ludii-1.3.14.jar"
AGENTS_JAR="$LUDII_DIR/agents/agents.jar"
GAME_FILE="$LUDII_DIR/FoW_Chess.lud"
OUTDIR="$LUDII_DIR/evaluation/results_v3"
LOG_FILE="$LUDII_DIR/evaluation/logs/reverse_matchups.log"

NGAMES=100
MAX_MOVES=400

# Ensure output directories exist
mkdir -p "$OUTDIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Verify jars exist
[[ ! -f "$LUDII_JAR"  ]] && { echo "ERROR: $LUDII_JAR not found"; exit 1; }
[[ ! -f "$AGENTS_JAR" ]] && { echo "ERROR: $AGENTS_JAR not found"; exit 1; }
[[ ! -f "$GAME_FILE"  ]] && { echo "ERROR: $GAME_FILE not found"; exit 1; }

JAVA_CP="${LUDII_JAR}:${AGENTS_JAR}"

# The 12 reverse-pair matchups (base agents as P1 vs LSTM agents)
REVERSE_PAIRS=(
    "ab_heuristic:ppo_lstm"
    "ab_heuristic:ppo_lstm_pretrained"
    "ab_learned:ppo_lstm"
    "ab_learned:ppo_lstm_pretrained"
    "ismcts:ppo_lstm"
    "ismcts:ppo_lstm_pretrained"
    "ppo:ppo_lstm"
    "ppo:ppo_lstm_pretrained"
    "random:ppo_lstm"
    "random:ppo_lstm_pretrained"
    "uct:ppo_lstm"
    "uct:ppo_lstm_pretrained"
)

echo "============================================================" | tee -a "$LOG_FILE"
echo " Running 12 Reverse-Pair Matchups (base agents vs LSTM)"
echo " Results dir: $OUTDIR"
echo " Log file: $LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

COUNTER=0
for PAIR in "${REVERSE_PAIRS[@]}"; do
    A="${PAIR%%:*}"
    B="${PAIR##*:}"
    COUNTER=$(( COUNTER + 1 ))

    PAIR_OUT="$OUTDIR/${A}_vs_${B}.csv"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Matchup $COUNTER/12: $A vs $B" | tee -a "$LOG_FILE"

    # Skip if already complete (file exists and has >= NGAMES data rows)
    if [ -f "$PAIR_OUT" ] && [ "$(tail -n +2 "$PAIR_OUT" 2>/dev/null | wc -l | tr -d ' ')" -ge "$NGAMES" ]; then
        echo "  [SKIP] $A vs $B — already have $(tail -n +2 "$PAIR_OUT" | wc -l | tr -d ' ') games" | tee -a "$LOG_FILE"
        continue
    fi

    # Time budget: neural agents need 1.0s per move
    TIME_BUDGET="1.0"
    echo "  [$A vs $B] time=${TIME_BUDGET}s, games=$NGAMES, max-moves=$MAX_MOVES" | tee -a "$LOG_FILE"

    # Run from LUDII_DIR so PPOLSTMAgent resolves ppo/venv/bin/python correctly
    (
        cd "$LUDII_DIR"
        ~/.asdf/installs/java/openjdk-26/bin/java -Xmx3g \
             -cp "$JAVA_CP" \
             agents.EvalRunner \
             --game          "$GAME_FILE" \
             --agent1        "$A" \
             --agent2        "$B" \
             --num-games     "$NGAMES" \
             --time-per-move "$TIME_BUDGET" \
             --max-moves     "$MAX_MOVES" \
             --output        "$PAIR_OUT"
    ) >> "$LOG_FILE" 2>&1 || echo "  [WARN] $A vs $B failed (continuing)" | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
done

echo "============================================================" | tee -a "$LOG_FILE"
echo " All 12 reverse-pair matchups complete!"
echo " Results: $OUTDIR"
echo "============================================================" | tee -a "$LOG_FILE"
