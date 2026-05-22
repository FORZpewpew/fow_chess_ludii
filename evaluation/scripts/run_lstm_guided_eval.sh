#!/bin/bash
# Evaluation of LSTMGuidedISMCTSAgent
# Must be run from fow_chess_ludii/ directory

set -e

JAVA=~/.asdf/installs/java/openjdk-26/bin/java
PYTHON=/Users/forzpewpew/Downloads/.venv/bin/python3
JAR_CP="/Users/forzpewpew/Downloads/fow_chess_ludii/Ludii-1.3.14.jar:/Users/forzpewpew/Downloads/fow_chess_ludii/agents/jars/agents.jar"
GAME=/Users/forzpewpew/Downloads/fow_chess_ludii/FoW_Chess.lud
OUT_DIR=/Users/forzpewpew/Downloads/fow_chess_ludii/evaluation/results_lstm_guided
GAMES=50
THINK=1

mkdir -p "$OUT_DIR"

echo "[eval] Starting lstm_belief_server on port 9998..."
$PYTHON /Users/forzpewpew/Downloads/fow_chess_ludii/ppo/lstm_belief_server.py --port 9998 > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "[eval] Server PID: $SERVER_PID"
sleep 6

run_match() {
    local p1=$1 p2=$2 out=$3
    echo "[eval] Running: $p1 vs $p2 ($GAMES games)..."
    $JAVA -cp "$JAR_CP" agents.EvalRunner \
        --agent1 "$p1" --agent2 "$p2" --num-games "$GAMES" --time-per-move "$THINK" \
        --game "$GAME" --output "$OUT_DIR/$out"
    echo "[eval] Done: $out"
}

# Match 1: lstm_guided_ismcts (P1) vs UCT (P2)
run_match lstm_guided_ismcts uct lstm_guided_vs_uct.csv

# Match 2: UCT (P1) vs lstm_guided_ismcts (P2)  
run_match uct lstm_guided_ismcts uct_vs_lstm_guided.csv

# Match 3: lstm_guided_ismcts (P1) vs ismcts_v4 (P2)
run_match lstm_guided_ismcts ismcts_v4 lstm_guided_vs_ismcts_v4.csv

# Match 4: ismcts_v4 (P1) vs lstm_guided_ismcts (P2)
run_match ismcts_v4 lstm_guided_ismcts ismcts_v4_vs_lstm_guided.csv

# Match 5: lstm_guided_ismcts (P1) vs particle_ismcts (P2)  
run_match lstm_guided_ismcts particle_ismcts lstm_guided_vs_particle.csv

# Match 6: particle_ismcts (P1) vs lstm_guided_ismcts (P2)
run_match particle_ismcts lstm_guided_ismcts particle_vs_lstm_guided.csv

# Belief Jaccard logging match (10 games with logging enabled)
echo "[eval] Running belief Jaccard logging match..."
$JAVA -cp "$JAR_CP" \
    -Dfow.belief.log=true \
    agents.EvalRunner \
    --agent1 lstm_guided_ismcts --agent2 uct --num-games 10 --time-per-move 1 \
    --game "$GAME" --output "$OUT_DIR/lstm_guided_belief_games.csv"
echo "[eval] Jaccard logging done."

kill $SERVER_PID 2>/dev/null
echo "[eval] All matches complete."
