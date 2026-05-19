#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_belief_eval.sh  —  Belief-accuracy evaluation for IS-MCTS (ismcts_v4)
#
# Runs IS-MCTS vs UCT for 20 games (10 as P1, 10 as P2) with belief logging
# enabled.  Requires that agents/agents.jar has been compiled with the
# belief-logging changes to ISMCTSAgent.java.
#
# Usage:
#   cd /Users/forzpewpew/Downloads/ludii
#   bash evaluation/scripts/run_belief_eval.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUDII_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$LUDII_DIR"

BELIEF_LOG="evaluation/results_grave/belief_accuracy.csv"
GAME="FoW_Chess.lud"
JAR_CP="Ludii-1.3.14.jar:agents/agents.jar"
GAMES_EACH=10

mkdir -p evaluation/results_grave evaluation/logs

echo "================================================================="
echo " IS-MCTS Belief Accuracy Evaluation"
echo " Log file : $BELIEF_LOG"
echo " Game file: $GAME"
echo "================================================================="

# -----------------------------------------------------------------------
# Pass 1: ismcts_v4 as Player 1, uct as Player 2
# -----------------------------------------------------------------------
echo ""
echo "[Pass 1/2] ismcts_v4 (P1) vs uct (P2)  — $GAMES_EACH games"
java \
    -Dfow.belief.log=true \
    -Dfow.belief.logfile="$BELIEF_LOG" \
    -cp "$JAR_CP" \
    agents.EvalRunner \
    --game "$GAME" \
    --agent1 ismcts_v4 \
    --agent2 uct \
    --num-games "$GAMES_EACH" \
    --output /tmp/belief_eval_p1.csv \
    --time-per-move 1.0

echo "[Pass 1/2] Done. Results in /tmp/belief_eval_p1.csv"

# -----------------------------------------------------------------------
# Pass 2: uct as Player 1, ismcts_v4 as Player 2
# -----------------------------------------------------------------------
echo ""
echo "[Pass 2/2] uct (P1) vs ismcts_v4 (P2)  — $GAMES_EACH games"
java \
    -Dfow.belief.log=true \
    -Dfow.belief.logfile="$BELIEF_LOG" \
    -cp "$JAR_CP" \
    agents.EvalRunner \
    --game "$GAME" \
    --agent1 uct \
    --agent2 ismcts_v4 \
    --num-games "$GAMES_EACH" \
    --output /tmp/belief_eval_p2.csv \
    --time-per-move 1.0

echo "[Pass 2/2] Done. Results in /tmp/belief_eval_p2.csv"

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "================================================================="
echo " Belief accuracy log: $BELIEF_LOG"
ROWS=$(tail -n +2 "$BELIEF_LOG" 2>/dev/null | wc -l | tr -d ' ')
echo " Total rows in log : $ROWS"
echo "================================================================="
echo ""
echo "Run analysis:"
echo "  python3 evaluation/scripts/analyze_belief_accuracy.py $BELIEF_LOG"
