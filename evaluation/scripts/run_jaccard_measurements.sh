#!/usr/bin/env bash
# =============================================================================
# Fog of War Chess — Jaccard Belief Accuracy Measurement Script
# Compiles agents.jar, starts belief server if needed, runs 10-game
# Jaccard measurements for IS-MCTS, Particle IS-MCTS, LSTM-Guided IS-MCTS.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Locate JDK ──────────────────────────────────────────────────────────
JAVA_HOME=$(ls -d ~/.asdf/installs/java/*/ 2>/dev/null | sort | tail -1 | tr -d '\n' || true)
if [ -z "${JAVA_HOME:-}" ]; then
  JAVA_HOME=$(/usr/libexec/java_home 2>/dev/null || true)
fi
if [ -z "${JAVA_HOME:-}" ]; then
  echo "ERROR: Cannot locate JDK. Set JAVA_HOME manually." >&2
  exit 1
fi
echo "Using JAVA_HOME: $JAVA_HOME"
JAVAC="$JAVA_HOME/bin/javac"
JAVA="$JAVA_HOME/bin/java"

# ── 2. Compile ─────────────────────────────────────────────────────────────
echo ""
echo "=== Step 1: Compiling agents ==="
mkdir -p agents/out
"$JAVAC" -cp "Ludii-1.3.14.jar" -d agents/out agents/src/*.java 2>&1
COMPILE_EXIT=$?
if [ "$COMPILE_EXIT" -ne 0 ]; then
  echo "ERROR: Compilation failed with exit code $COMPILE_EXIT" >&2
  exit "$COMPILE_EXIT"
fi
echo "Compilation succeeded."

cd agents/out
jar cf ../agents.jar $(find . -name "*.class")
cd "$SCRIPT_DIR"
ls -la agents/agents.jar
echo "agents.jar built successfully."

# ── 3. Ensure output directories exist ─────────────────────────────────────
mkdir -p evaluation/results_grave
mkdir -p evaluation/results_particle
mkdir -p evaluation/results_lstm_guided

# ── 4. Check / start belief server (needed for LSTM-Guided) ────────────────
echo ""
echo "=== Step 2: Checking belief server on port 9998 ==="

SERVER_RUNNING=0
if lsof -i :9998 2>/dev/null | grep -q LISTEN; then
  echo "Belief server already running on port 9998."
  SERVER_RUNNING=1
fi

if [ "$SERVER_RUNNING" -eq 0 ]; then
  echo "Belief server NOT running. Starting..."
  /Users/forzpewpew/Downloads/.venv/bin/python3 ppo/lstm_belief_server.py --port 9998 \
    > /tmp/bs_jaccard_fix.log 2>&1 &
  SERVER_PID=$!
  echo "Server PID: $SERVER_PID"

  echo "Waiting for server warmup (up to 90s)..."
  for i in $(seq 1 45); do
    sleep 2
    if grep -q "warmup complete\|Listening on" /tmp/bs_jaccard_fix.log 2>/dev/null; then
      echo "Server ready after $((i*2))s"
      SERVER_RUNNING=1
      break
    fi
  done

  echo "--- Server log tail ---"
  tail -10 /tmp/bs_jaccard_fix.log
  echo "-----------------------"

  if [ "$SERVER_RUNNING" -eq 0 ]; then
    echo "WARNING: Server may not be ready yet. Proceeding anyway."
  fi
fi

# ── 5. Run IS-MCTS ─────────────────────────────────────────────────────────
echo ""
echo "=== Step 3: IS-MCTS 10-game Jaccard measurement ==="
# Remove stale belief CSV so each run starts fresh
rm -f belief_accuracy.csv evaluation/results_grave/ismcts_belief_accuracy_fixed.csv
"$JAVA" -Dfow.belief.log=true \
  -Dfow.belief.logfile=evaluation/results_grave/ismcts_belief_accuracy_fixed.csv \
  -cp "agents/agents.jar:Ludii-1.3.14.jar" \
  agents.EvalRunner \
  --game FoW_Chess.lud \
  --agent1 ismcts_v4 \
  --agent2 uct \
  --num-games 10 \
  --time-per-move 1 \
  --output evaluation/results_grave/ismcts_belief_fixed_games.csv \
  2>&1 | tee /tmp/ismcts_run.log | tail -30
echo "IS-MCTS run complete."

echo ""
echo "--- IS-MCTS belief CSV ---"
wc -l evaluation/results_grave/ismcts_belief_accuracy_fixed.csv 2>/dev/null
head -3 evaluation/results_grave/ismcts_belief_accuracy_fixed.csv 2>/dev/null
ls -lt "$SCRIPT_DIR/evaluation/results_grave/" | head -10

# ── 6. Run Particle IS-MCTS ────────────────────────────────────────────────
echo ""
echo "=== Step 4: Particle IS-MCTS 10-game Jaccard measurement ==="
rm -f evaluation/results_particle/particle_belief_accuracy_fixed.csv
"$JAVA" -Dfow.belief.log=true \
  -Dfow.belief.logfile=evaluation/results_particle/particle_belief_accuracy_fixed.csv \
  -cp "agents/agents.jar:Ludii-1.3.14.jar" \
  agents.EvalRunner \
  --game FoW_Chess.lud \
  --agent1 particle_ismcts \
  --agent2 uct \
  --num-games 10 \
  --time-per-move 1 \
  --output evaluation/results_particle/particle_belief_fixed_games.csv \
  2>&1 | tee /tmp/particle_run.log | tail -30
echo "Particle IS-MCTS run complete."

echo ""
echo "--- Particle belief CSV ---"
wc -l evaluation/results_particle/particle_belief_accuracy_fixed.csv 2>/dev/null
head -3 evaluation/results_particle/particle_belief_accuracy_fixed.csv 2>/dev/null
ls -lt "$SCRIPT_DIR/evaluation/results_particle/" | head -10

# ── 7. Run LSTM-Guided IS-MCTS ─────────────────────────────────────────────
echo ""
echo "=== Step 5: LSTM-Guided IS-MCTS 10-game Jaccard measurement ==="
if ! lsof -i :9998 2>/dev/null | grep -q LISTEN; then
  echo "WARNING: Belief server is NOT listening on port 9998!"
fi

rm -f evaluation/results_lstm_guided/lstm_guided_belief_accuracy_fixed.csv
"$JAVA" -Dfow.belief.log=true \
  -Dfow.belief.logfile=evaluation/results_lstm_guided/lstm_guided_belief_accuracy_fixed.csv \
  -cp "agents/agents.jar:Ludii-1.3.14.jar" \
  agents.EvalRunner \
  --game FoW_Chess.lud \
  --agent1 lstm_guided_ismcts \
  --agent2 uct \
  --num-games 10 \
  --time-per-move 1 \
  --output evaluation/results_lstm_guided/lstm_guided_belief_fixed_games.csv \
  2>&1 | tee /tmp/lstm_guided_run.log | tail -30
echo "LSTM-Guided run complete."

echo ""
echo "--- Locating LSTM-Guided belief CSV ---"
find "$SCRIPT_DIR" -name "*lstm*belief*.csv" -newer agents/agents.jar 2>/dev/null | head -5
ls -lt "$SCRIPT_DIR/evaluation/results_lstm_guided/" | head -10

# ── 8. Compute corrected mean Jaccard values ───────────────────────────────
echo ""
echo "=== Step 6: Computing corrected mean Jaccard for all three agents ==="

/Users/forzpewpew/Downloads/.venv/bin/python3 - <<'PYEOF'
import csv, statistics, os, glob

search_paths = {
    'IS-MCTS': [
        '/Users/forzpewpew/Downloads/fow_chess_ludii/evaluation/results_grave/ismcts_belief_accuracy_fixed.csv',
        '/Users/forzpewpew/Downloads/fow_chess_ludii/evaluation/results_grave/ismcts_belief_accuracy.csv',
        '/Users/forzpewpew/Downloads/fow_chess_ludii/ismcts_belief_accuracy.csv',
        '/Users/forzpewpew/Downloads/fow_chess_ludii/belief_accuracy.csv',
    ],
    'Particle': [
        '/Users/forzpewpew/Downloads/fow_chess_ludii/evaluation/results_particle/particle_belief_accuracy_fixed.csv',
        '/Users/forzpewpew/Downloads/fow_chess_ludii/evaluation/results_particle/particle_belief_accuracy.csv',
        '/Users/forzpewpew/Downloads/fow_chess_ludii/particle_belief_accuracy.csv',
    ],
    'LSTM-Guided': [
        '/Users/forzpewpew/Downloads/fow_chess_ludii/evaluation/results_lstm_guided/lstm_guided_belief_accuracy_fixed.csv',
        '/Users/forzpewpew/Downloads/fow_chess_ludii/evaluation/results_lstm_guided/lstm_guided_belief_accuracy.csv',
    ],
}

recent = sorted(
    glob.glob('/Users/forzpewpew/Downloads/fow_chess_ludii/**/*belief*.csv', recursive=True),
    key=os.path.getmtime, reverse=True
)[:10]
print('Recently modified belief CSVs:')
for r in recent:
    print(' ', r)

print()
results = {}
for agent, paths in search_paths.items():
    found = False
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    reader = csv.DictReader(f)
                    rows = [float(r['avg_jaccard']) for r in reader if r.get('avg_jaccard')]
                if rows:
                    mean_j = statistics.mean(rows)
                    results[agent] = mean_j
                    print(f'{agent}  path={p}  N={len(rows)}  mean_J={mean_j:.4f}')
                    found = True
                    break
            except Exception as e:
                print(f'{agent}  ERROR reading {p}: {e}')
    if not found:
        print(f'{agent}: NOT FOUND in expected paths')

print()
print('=== CORRECTED MEAN JACCARD SUMMARY ===')
for agent, j in results.items():
    print(f'  {agent:20s}  J = {j:.4f}')
PYEOF

echo ""
echo "=== All steps complete ==="
