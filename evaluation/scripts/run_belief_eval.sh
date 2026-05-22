#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_belief_eval.sh  —  Belief-accuracy evaluation for IS-MCTS (ismcts_v4)
#
# Runs IS-MCTS vs UCT for 50 games (25 as P1, 25 as P2) with belief logging
# enabled.  Requires that agents/agents.jar has been compiled with the
# belief-logging changes to ISMCTSAgent.java.
#
# Usage:
#   cd /path/to/fow_chess_ludii
#   bash evaluation/scripts/run_belief_eval.sh
#
# Output:
#   evaluation/results_grave/ismcts_belief_accuracy.csv  — per-move Jaccard log
#   /tmp/belief_eval_p1.csv  — game outcomes (ismcts_v4 as P1)
#   /tmp/belief_eval_p2.csv  — game outcomes (uct as P1)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUDII_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$LUDII_DIR"

BELIEF_LOG="evaluation/results_grave/ismcts_belief_accuracy.csv"
GAME="FoW_Chess.lud"
JAR_CP="Ludii-1.3.14.jar:agents/agents.jar"
GAMES_EACH=25   # 25 as P1 + 25 as P2 = 50 games total

mkdir -p evaluation/results_grave evaluation/logs

echo "================================================================="
echo " IS-MCTS Belief Accuracy Evaluation"
echo " Log file : $BELIEF_LOG"
echo " Game file: $GAME"
echo " Games    : $((GAMES_EACH * 2))  ($GAMES_EACH as P1 + $GAMES_EACH as P2)"
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

echo "[Pass 1/2] Done. Game outcomes in /tmp/belief_eval_p1.csv"

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

echo "[Pass 2/2] Done. Game outcomes in /tmp/belief_eval_p2.csv"

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
echo ""
echo "Or compare with PPO probe results:"
echo "  python3 evaluation/scripts/analyze_belief_accuracy.py \\"
echo "    $BELIEF_LOG \\"
echo "    --ppo-probe evaluation/results_grave/ppo_belief_probe.csv"
