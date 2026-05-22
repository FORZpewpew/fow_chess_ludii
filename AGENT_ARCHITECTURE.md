# Agent Architecture: AI Research for Fog of War Chess in Ludii 1.3.14

**Thesis project:** Ivan Beltsov  
**Game:** `FoW_Chess.lud` — Fog of War Chess (Dark Chess) on an 8×8 board  
**Runtime:** `Ludii-1.3.14.jar`  
**Last updated:** 2026-05-19

---

## Table of Contents

1. [Imperfect Information Constraints](#1-imperfect-information-constraints)
2. [Agent Roster (8 Agents)](#2-agent-roster-8-agents)
3. [PPO-LSTM Architecture](#3-ppo-lstm-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Ludii–Python Bridge (JPype)](#5-ludiipython-bridge-jpype)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Belief-State Analysis](#7-belief-state-analysis)
8. [Key Results](#8-key-results)
9. [Appendix: Ludii API Quick Reference](#9-appendix-ludii-api-quick-reference)

---

## 1. Imperfect Information Constraints

### How FoW Chess Hides Information

The game's `.lud` source implements hidden information via Ludii's native `set Hidden` / `is Hidden` primitives. After every move `UpdateFog` runs:

```
(define "UpdateFog"
    (and {
        ("HideRegionFrom" (sites Board) P1)   ; blank entire board for P1
        ("HideRegionFrom" (sites Board) P2)   ; blank entire board for P2
        ("RevealVisionForPlayer" P1)           ; re-reveal P1's own FOV
        ("RevealVisionForPlayer" P2)           ; re-reveal P2's own FOV
    })
)
```

Each player's visibility is computed per piece type:
- **Sliding pieces** (Rook/Bishop/Queen): `sites LineOfSight Empty` + `sites LineOfSight Piece`
- **Knight**: walk pattern `{{F F R F} {F F L F}}`
- **King**: `sites Around (site) All`
- **Pawn**: two diagonal attack squares only (NE/NW for P1; SE/SW for P2)

### What an Agent Observes

When Ludii calls `selectAction(game, context, maxSeconds, maxIterations, maxDepth)`, the `context` argument contains the **filtered** view for that player — `isHidden(playerId, site, 0, SiteType.Cell)` returns `true` for fog-covered squares. The agent:

- Sees its own pieces and their exact types
- Sees any enemy pieces currently within its line-of-sight
- Does **not** see enemy pieces outside its FOV; those sites appear as hidden
- Has access only to `context.game().moves(context)` — legal moves from the observable state

This is the fundamental asymmetry: **the agent's context is already the correct filtered view**. Agents must not access physically hidden fields as if they were known, otherwise they are "cheating". (Note: Ludii stores full physical state internally — the `isHidden` flag gates visibility, it does not remove data from memory.)

### Agent Design Taxonomy

| Approach | Handles Uncertainty | Information Used |
|---|---|---|
| Random | Implicitly (trivially valid) | None |
| UCT (naïve) | None — treats observable state as complete | Observable only |
| IS-MCTS (determinization) | Via random world sampling | Observable + sampled completions |
| Alpha-Beta (Heuristic / Learned) | None — minimax on observable state | Observable only |
| PPO-LSTM | Via LSTM memory over game history | Observable + temporal memory |

---

## 2. Agent Roster (8 Agents)

### Agent 1 — Random

| Property | Value |
|---|---|
| **Slug** | `random` |
| **Source** | Ludii built-in |
| **Strength rank** | 1 (weakest / floor) |

Selects one move uniformly at random from all legal moves. Instantiated via:

```java
AI agent = AIFactory.createAI("Random");
```

**Rationale:** Establishes the absolute performance floor. Any agent that cannot beat Random reliably has a fundamental implementation error.

---

### Agent 2 — UCT

| Property | Value |
|---|---|
| **Slug** | `uct` |
| **Source** | Ludii built-in |
| **Strength rank** | 2 |

Ludii's built-in MCTS with UCB1 selection. Treats the filtered observable context as if it were a perfect-information game state. Instantiated via:

```java
AI agent = AIFactory.createAI("UCT");
```

**Imperfect information handling:** Single implicit determinization at the root (the observable state). Counterintuitively, this outperforms IS-MCTS in experiments — possibly because UCT's single-world search is deeper and more coherent than IS-MCTS's shallow multi-world search.

---

### Agent 3 — IS-MCTS (10 determinizations)

| Property | Value |
|---|---|
| **Slug** | `ismcts` |
| **Source** | `agents/src/ISMCTSAgent.java` |
| **Strength rank** | 3 |
| **Determinizations** | 10 |

Information Set MCTS with plurality voting over determinizations. Per move:
1. Deep-copy the observable context
2. Randomly shuffle hidden opponent pieces among hidden squares
3. Run UCT for `SIMS_PER_DET = 200` iterations on the fully observable copy
4. Record UCT's recommendation as one vote
5. Return the move with the most votes across all 10 determinizations

```java
new ISMCTSAgent(10);   // or ISMCTSAgent() — default is 10
```

**Determinization algorithm** (from `ISMCTSAgent.determinize()`):
- Identify sites hidden from self that are occupied by the opponent (`hiddenOccupiedSites`)
- Identify sites hidden from self that are empty (`emptyCandidateSites`)
- Pool all hidden sites together, shuffle with `ThreadLocalRandom`
- Clear the hidden occupied sites; re-place the pieces at the first `|hiddenPieces|` shuffled positions

```java
// Each determinization reuses the same MCTS instance (avoids thread-pool leaks):
uct = MCTS.createUCT();
uct.initAI(game, playerID);
// ... per determinization:
Move bestForDet = uct.selectAction(game, det, timePerDet, itersPerDet, maxDepth);
```

**Belief logging:** When system property `fow.belief.log=true` is set, the agent logs per-move Jaccard-similarity statistics comparing each determinization's placement against the true hidden state:

```
CSV: game_id, move_num, player, num_hidden_pieces, avg_jaccard, min_jaccard, max_jaccard, num_determinizations
```

---

### Agent 4 — IS-MCTS v4 (25 determinizations)

| Property | Value |
|---|---|
| **Slug** | `ismcts_v4` |
| **Source** | `agents/src/ISMCTSAgent.java` |
| **Strength rank** | ~3 (similar to ismcts) |
| **Determinizations** | 25 |

Same `ISMCTSAgent` class constructed with 25 determinizations:

```java
new ISMCTSAgent(25);
```

More world samples per move at the cost of less time per UCT search. In practice, strength is similar to the 10-determinization variant — the budget is split across more but shallower searches.

---

### Agent 5 — GRAVE

| Property | Value |
|---|---|
| **Slug** | `grave` |
| **Source** | Ludii built-in (custom construction in `EvalRunner.java`) |
| **Strength rank** | ~4 |

MCTS with GRAVE (Generalised Rapid Action Value Estimation) selection policy. Constructed directly via Ludii's MCTS builder:

```java
new MCTS(
    new search.mcts.selection.McGRAVE(),
    new search.mcts.playout.RandomPlayout(200),
    new search.mcts.backpropagation.MonteCarloBackprop(),
    new search.mcts.finalmoveselection.RobustChild()
);
```

GRAVE amortises action value estimates across the tree, beneficial when the search budget is limited.

---

### Agent 6 — GRAVE+MAST

| Property | Value |
|---|---|
| **Slug** | `grave_mast` |
| **Source** | Ludii built-in (custom construction in `EvalRunner.java`) |
| **Strength rank** | ~4 |

GRAVE selection with MAST (Move-Average Sampling Technique) playout policy:

```java
new MCTS(
    new search.mcts.selection.McGRAVE(),
    new search.mcts.playout.MAST(),
    new search.mcts.backpropagation.MonteCarloBackprop(),
    new search.mcts.finalmoveselection.RobustChild()
);
```

MAST biases random playouts toward moves that have historically been good, improving playout quality.

---

### Agent 7 — PPO-LSTM v4

| Property | Value |
|---|---|
| **Slug** | `ppo_lstm_v4` |
| **Source** | `agents/src/PPOLSTMAgent.java` + `ppo/ppo_lstm_server.py` + `ppo/policy_lstm.py` |
| **Checkpoint** | `checkpoints/ppo_lstm_v4_policy.pt` |
| **Elo (reported)** | ~1267 |

Recurrent actor-critic trained entirely from scratch via PPO with truncated BPTT. The Java agent spawns a Python subprocess at `initAI` time and communicates via newline-delimited JSON over stdin/stdout. See [Section 3](#3-ppo-lstm-architecture) for full architecture details.

```java
new PPOLSTMAgent("checkpoints/ppo_lstm_v4_policy.pt");
```

---

### Agent 8 — PPO-LSTM Pretrained v4

| Property | Value |
|---|---|
| **Slug** | `ppo_lstm_pretrained_v4` |
| **Source** | Same as Agent 7 |
| **Checkpoint** | `checkpoints/ppo_lstm_pretrained_v4.pt` |
| **Elo (reported)** | ~1283 |

Same architecture as Agent 7 but initialized via **Behavioural Cloning** (BC) pretraining on self-play game histories before PPO fine-tuning. BC pretraining improves both convergence speed and final performance.

```java
new PPOLSTMAgent("checkpoints/ppo_lstm_pretrained_v4.pt");
```

---

### Additional Agents (in codebase, not in final tournament)

**AB-Heuristic** (`ab_heuristic`, `ABHeuristicAgent.java`): Alpha-Beta minimax on the observable state with a custom FoW-specific heuristic (material + mobility + fog penalty + king safety + center control). Dominates all search/RL agents in final evaluation (~100% win rate). Uses `search.minimax.AlphaBetaSearch` as base class.

**AB-Learned** (`ab_learned`, `ABLearnedAgent.java`): Alpha-Beta with a self-play learned linear value function (10 features, SGD-trained on 20 Random-vs-UCT games at `initAI` time). Similar strength to AB-Heuristic.

---

## 3. PPO-LSTM Architecture

### Neural Network (`ppo/policy_lstm.py`)

```
FoWPolicyLSTM
├── encoder   : Linear(128→512) → ReLU → Linear(512→512) → ReLU
├── lstm      : LSTM(input=512, hidden=512, num_layers=1, batch_first=True)
├── actor     : Linear(512→4096)          — policy head
├── critic    : Linear(512→1)             — value head (standard)
├── critic_encoder : Linear(576→512) → ReLU  — privileged critic (training only)
└── critic_head    : Linear(512→1)             — privileged value head
```

Key constants:

| Constant | Value | Meaning |
|---|---|---|
| `OBS_DIM` | 128 | 2 channels × 64 squares |
| `HIDDEN_DIM` | 512 | LSTM hidden state size |
| `ACT_DIM` | 4096 | 64 × 64 from–to action space |
| `VERSION` | `v4` | Current model version |

LSTM weights are initialised with orthogonal initialisation for training stability.

### Observation Space (128-dim float32)

```
obs[0..63]   — ownership channel (egocentric, 4 discrete values):
                 0.000  → square hidden under fog (unknown)
                 0.333  → confirmed empty visible square
                 0.667  → square occupied by this player's own piece
                 1.000  → square visibly occupied by the opponent

obs[64..127] — piece-type channel:
                 piece_type_index / 6.0   (0 if hidden or absent)
```

The hidden channel (formerly a 3rd channel) was removed because `owner == 0.0` is already a lossless indicator of fog. This reduces the observation dimension from 192 to 128.

The encoding is **egocentric** — observation values are relative to the observing player, not absolute player indices.

### Action Space (4096 discrete actions)

```
action_index = from_site * 64 + to_site
```

- `from_site`, `to_site` ∈ [0, 63] (8×8 board, row-major)
- Illegal actions are masked to `-inf` before sampling; if all actions are masked (edge case), uniform distribution is used to prevent NaN

### Forward Pass

The model handles both inference (single-step, shape `(B, obs_dim)`) and training (sequence, shape `(B, T, obs_dim)`) transparently by inserting/removing the time dimension:

```python
dist, values, h_new, c_new = policy(obs, legal_mask, h, c)
# dist   : Categorical over 4096 actions (legal actions only)
# values : scalar value estimate (B,) or (B, T)
# h_new, c_new : updated LSTM state, shape (1, B, 512)
```

### Privileged Critic (Asymmetric Training)

During training, the critic can optionally receive the true board state (64 binary values) appended to the LSTM output, enabling asymmetric actor-critic:

```python
# critic_input shape: (B, T, 512 + 64) = (B, T, 576)
critic_input = torch.cat([lstm_out, true_board], dim=-1)
values = critic_head(critic_encoder(critic_input))
```

This is disabled during inference — the actor uses only observable information.

### Java–Python Protocol (`PPOLSTMAgent.java` ↔ `ppo_lstm_server.py`)

The Java agent spawns a Python subprocess at `initAI`:

```
ProcessBuilder: python ppo/ppo_lstm_server.py --checkpoint <path>
```

After the subprocess prints `READY\n`, communication is via newline-delimited JSON on stdin/stdout:

| Direction | Message | Meaning |
|---|---|---|
| Java → Python | `{"type":"new_game"}` | Reset LSTM hidden state to zeros |
| Python → Java | `{"status":"ok"}` | Acknowledgement |
| Java → Python | `{"type":"move","obs":[...128...],"legal":[...4096...]}` | Request action |
| Python → Java | `{"action": N}` | Selected action index |

**All Python informational output goes to stderr** to avoid contaminating the stdout JSON protocol.

---

## 4. Training Pipeline

### PPO with Truncated BPTT (`ppo/train_ppo_lstm.py`)

Training is performed entirely in Python using PyTorch. The key hyperparameters:

| Parameter | Value | Meaning |
|---|---|---|
| `N_EPISODES` | 16 | Games collected per update |
| `N_UPDATES` | 400 | Total PPO update steps |
| `PPO_EPOCHS` | 4 | Gradient passes per batch of segments |
| `CLIP_EPS` | 0.2 | PPO clipping threshold ε |
| `GAMMA` | 0.99 | Discount factor |
| `LAMBDA` | 0.95 | GAE λ |
| `LR` | 1e-4 | Adam learning rate |
| `TBPTT_LEN` | 64 | Truncated-BPTT segment length |
| `ENTROPY_COEFF` | 0.01 | Entropy bonus coefficient |
| `VALUE_COEFF` | 0.5 | Value loss coefficient |
| `MAX_GRAD_NORM` | 0.5 | Gradient clip norm |
| `RANDOM_ONLY_UPDATES` | 100 | Updates before self-play is introduced |

**Episode collection:** Full episodes are collected up to `MAX_PLIES = 400` plies per side. Each step stores `(obs, mask, action, log_prob, value, reward, done, h_t, c_t)`.

**Segmentation:** Episodes are chunked into segments of length `TBPTT_LEN = 64`. The stored `(h_start, c_start)` at the beginning of each segment is used as the initial LSTM state for that segment's forward pass. Gradients do **not** propagate across segment boundaries (`.detach()`).

**Opponent pool curriculum:**
- Updates 1–100: opponent is always Random
- Updates 101+: 50% Random / 50% a randomly chosen previous checkpoint from the pool
- Checkpoints saved every 25 updates are added to the pool

**Reward shaping** (on top of sparse ±1 terminal reward):
- `+0.02` — fog reveal bonus: when a previously hidden enemy piece becomes visible
- `+0.05 × piece_value` — capture bonus when an enemy piece disappears
- `−0.03` — king exposure penalty when own king is adjacent to a visible enemy piece

**Advantage estimation:** GAE(λ=0.95) computed per segment. Advantages are normalised within each segment (population std to avoid NaN on length-1 segments).

### Behavioural Cloning Pretraining (`ppo/pretrain.py`)

Two-phase pretraining for the Pretrained v4 agent:

**Phase 1 — Behavioural Cloning (BC):**

```
Dataset: training/results/selfplay_training_data.jsonl
Schema:  {game_id, ply, player, outcome, board{cells[64]}, legal_move_count, move_history[{from,to,mover}]}
```

- Samples are grouped by `game_id` and sorted by `ply` to enable sequential (game-level) forward passes
- 90% / 10% train / validation split at the game level
- Loss: `cross_entropy(logits / 2.0, action) + 0.5 * MSE(value, outcome)` — temperature scaling (÷2) prevents entropy collapse
- TBPTT with segment length 16 within each game (detach every 16 steps)
- 50 epochs with early stopping (patience = 10) on validation loss
- Best checkpoint by validation loss is saved as `checkpoints/ppo_lstm_pretrained_v4.pt`

**Phase 2 — PPO fine-tuning:**

- Loads the BC checkpoint as warm start
- Skips the `RANDOM_ONLY_UPDATES` phase (starts self-play immediately, since BC has already bootstrapped a reasonable policy)
- Runs 300 PPO updates using the standard `train_ppo_lstm.train()` loop

### Running Training

```bash
cd fow_chess_ludii

# From-scratch PPO (400 updates):
python3 ppo/train_ppo_lstm.py

# With warm start:
python3 ppo/train_ppo_lstm.py --checkpoint checkpoints/ppo_lstm_v4_policy.pt

# BC + PPO pretraining:
python3 ppo/pretrain.py --ppo_finetune --ppo_updates 300

# Smoke-test (2 updates × 2 episodes):
python3 ppo/train_ppo_lstm.py --updates 2 --episodes 2
```

---

## 5. Ludii–Python Bridge (JPype)

The Python training environment (`ppo/fow_env.py`) uses **JPype1** to start and call into the Ludii JVM directly from Python:

```python
import jpype
import jpype.imports

jpype.startJVM(
    jvm_path,                          # libjvm.dylib / .so from JAVA_HOME
    f"-Djava.class.path={LUDII_JAR}:{AGENTS_JAR}",
    "-Xmx2g",
    convertStrings=False,
)

from other import GameLoader
from other.context import Context
from other.trial import Trial
```

This eliminates the need for inter-process communication during training — the Python PPO loop calls Ludii methods directly as if they were Python functions.

**`FoWChessEnv`** wraps the JPype bridge in a Gym-style environment:

```python
env = FoWChessEnv(player=1, opponent_fn=None)  # opponent defaults to Random
obs, legal_mask = env.reset()
obs, reward, done, info = env.step(action_idx)
```

The `opponent_fn` parameter can be a callable that returns action indices for the opposing side, enabling self-play:

```python
def opponent_fn(obs: np.ndarray, legal_mask: np.ndarray) -> int:
    # returns action index for the opponent
    ...
env = FoWChessEnv(player=1, opponent_fn=opponent_fn)
```

**Key configuration constants** in `fow_env.py`:

```python
JAVA_HOME  = "/Users/forzpewpew/.asdf/installs/java/openjdk-26"
LUDII_JAR  = ".../Ludii-1.3.14.jar"
AGENTS_JAR = ".../agents/jars/agents.jar"
GAME_PATH  = ".../FoW_Chess.lud"
MAX_PLIES  = 400      # hard cap per game
OBS_DIM    = 128      # 2 × 64
ACTION_DIM = 4096     # 64 × 64
```

**Note:** The `PPOLSTMAgent.java` Java agent uses a different approach for inference — it spawns a Python subprocess (`ppo_lstm_server.py`) rather than using JPype. JPype is only used during Python-side training.

---

## 6. Evaluation Methodology

### Evaluation Harness (`agents/src/EvalRunner.java`)

Headless evaluation harness that runs agent matchups and writes per-game CSV logs.

**Usage:**

```bash
java -cp "Ludii-1.3.14.jar:agents.jar" agents.EvalRunner \
     --agent1 <slug> --agent2 <slug> \
     --num-games <N> \
     --output <path/results.csv> \
     [--time-per-move <seconds>] \
     [--max-moves <N>]
```

**Agent slugs:**
```
random, uct, pimc_uct, ab_heuristic, ab_learned,
ismcts, ismcts_v4, ppo_lstm, ppo_lstm_pretrained,
ppo_lstm_v4, ppo_lstm_pretrained_v4, grave, grave_mast
```

**Implementation note:** `trial.numMoves()` is not used to count plies because each `(set Hidden ...)` in `UpdateFog` fires ~164 times per chess ply, inflating the count by ~164×. Instead, a separate `plyCount` variable is incremented per `game.apply()` call.

**Max-moves limit:** Games exceeding `--max-moves` (default 300) are recorded as draws.

**CSV output schema:**
```csv
game_id,agent_p1,agent_p2,winner,num_moves,draw,timestamp_utc
```

### Build

```bash
bash fow_chess_ludii/evaluation/scripts/build.sh
```

Compiles all `.java` sources against `Ludii-1.3.14.jar` and packages them into `agents/jars/agents.jar`.

### Tournament Scripts

| Script | Purpose |
|---|---|
| `run_tournament.sh` | Initial round-robin (early agents) |
| `run_tournament_v4.sh` | V4 tournament (all 8 agents) |
| `run_grave_tournament.sh` | GRAVE/GRAVE+MAST matchups |
| `run_time_sensitivity.sh` | 2s vs 5s time control comparison |
| `run_belief_eval.sh` | Enable Jaccard belief logging |
| `merge_and_elo_v4.sh` | Merge CSVs + run Elo computation |

### Round-Robin Structure

All 8 agents participate in a complete round-robin. For every ordered pair `(A_i, A_j)` where `i ≠ j`:
- 100 games with `A_i` as P1 (White), `A_j` as P2 (Black)

Total: C(8,2) = 28 unordered pairs × 200 games = **5,600 games maximum**

### Elo Rating

Bayesian Elo via maximum-likelihood estimation (MLE):

```python
# evaluation/scripts/compute_elo.py
from scipy.optimize import minimize

def elo_likelihood(ratings, results_matrix):
    nll = 0.0
    for i in range(n):
        for j in range(n):
            if i == j: continue
            wins, games = results_matrix[i][j]
            if games == 0: continue
            p_ij = 1.0 / (1.0 + 10**((ratings[j] - ratings[i]) / 400.0))
            nll -= wins * np.log(p_ij + 1e-9)
            nll -= (games - wins) * np.log(1 - p_ij + 1e-9)
    return nll

result = minimize(elo_likelihood, x0=[1500.0] * n, method='L-BFGS-B')
```

### Draw Handling

FoW Chess terminates in a draw when `counter == 100` (50-move rule: 100 half-moves without pawn move or capture):

```
(if (= (counter) 100) (result Mover Draw))
```

Draws count as 0.5 for both agents in Elo.

### Statistical Significance (`evaluation/scripts/significance.py`)

Two-sided binomial test per pairwise comparison, α = 0.05, with Wilson 95% CI.

### Think-Time Sensitivity (`evaluation/scripts/analyze_time_sensitivity.py`)

Time controls tested: **2s/move** and **5s/move**. Results stored in `evaluation/results_time_sensitivity/`.

---

## 7. Belief-State Analysis

### Jaccard Similarity Tracking (ISMCTSAgent)

When `fow.belief.log=true`, `ISMCTSAgent` logs the quality of each determinization relative to the true hidden state:

```
Jaccard(ground_truth, belief) = |intersection| / |union|
```

Where each set element is a `(site, piece_type)` pair.

**Findings:**
- Using a position-only metric (set elements are site indices only), Jaccard degrades from ~0.80 in the early game to ~0.11 in the late game
- Note: the thesis uses a stricter joint position-and-piece-type metric (set elements are `(site, piece_type)` pairs), which yields substantially lower values (~0.022 in the opening, ~0.012 in the midgame), because a determinization must place the correct piece type on the correct square to count as a match
- As pieces are captured and positions become more constrained, random placement quality worsens
- This explains IS-MCTS's underperformance: late-game determinizations are poor approximations of reality

### Belief Probe (`ppo/belief_probe.py`)

A linear classifier (MLP probe) trained on the PPO-LSTM hidden state to predict hidden piece positions:

```
Architecture: h (512) → Linear(512→256) → ReLU → Linear(256→64×7) → view(64, 7)
Output: logits over 7 piece types (0=empty, 1=pawn, 2=rook, 3=bishop, 4=knight, 5=queen, 6=king) per square
```

Training: `ppo/train_belief_probe.py` using data generated by `ppo/generate_probe_data.py`.  
Evaluation: `ppo/eval_belief_probe.py`.  
Checkpoint: `checkpoints/belief_probe_v4.pt`.  
History: `checkpoints/belief_probe_v4_history.json`.

**Finding:** The belief probe achieves **89.68% accuracy** on hidden square classification, showing that the LSTM hidden state implicitly encodes a belief state over the opponent's pieces even though the training signal is only final game outcomes (sparse reward). This is evidence of emergent belief-state representation.

---

## 8. Key Results

### Final Elo Rankings (excluding AB-Heuristic and AB-Learned)

| Rank | Agent | Elo (approx) |
|---|---|---|
| 1 | PPO-LSTM Pretrained v4 | ~1756 |
| 2 | PPO-LSTM v4 | ~1745 |
| 3 | UCT | ~1643 |
| 4 | IS-MCTS (10 det.) | ~1480 |
| 5 | Enhanced IS-MCTS (25 det.) | ~1438 |
| 6 | GRAVE | ~1372 |
| 7 | GRAVE+MAST | ~1354 |
| 8 | Random | ~1211 |

### Key Findings

1. **UCT > IS-MCTS (counterintuitive):** Vanilla UCT outperforms IS-MCTS despite IS-MCTS being theoretically superior. Explanation: the total computation budget is split across many shallow UCT searches, whereas UCT's single deep search produces more coherent plans.

2. **PPO-LSTM > UCT:** Both PPO-LSTM v4 (Elo ~1745) and PPO-LSTM Pretrained v4 (Elo ~1756) outperform UCT (Elo ~1643). The LSTM's ability to maintain a history of observations provides a genuine advantage over stateless search.

4. **BC pretraining helps:** The pretrained variant outperforms the from-scratch variant in both Elo and training convergence speed.

5. **Rankings stable across time controls:** Agent rankings are consistent between 2s and 5s time controls.

6. **Emergent belief state:** The LSTM encodes implicit belief about hidden pieces (89.68% probe accuracy) despite being trained only on game outcomes.

---

## 9. Appendix: Ludii API Quick Reference

### Core Interfaces and Classes

| Class | Package | Key Methods |
|---|---|---|
| `AI` | `other` | `selectAction(Game, Context, double, int, int)`, `initAI(Game, int)`, `closeAI()`, `supportsGame(Game)` |
| `Game` | `game` | `moves(Context)`, `apply(Context, Move)`, `start(Context)`, `board()` |
| `Context` | `other.context` | `state()`, `trial()`, `game()` |
| `State` | `other.state` | `mover()`, `containerStates()` |
| `ContainerState` | `other.state` | `who(site, SiteType)`, `what(site, SiteType)`, `isHidden(player, site, level, SiteType)`, `setSite(...)` |
| `Trial` | `other.trial` | `over()`, `ranking()`, `numMoves()` |
| `Move` | `other.move` | `fromNonDecision()`, `toNonDecision()`, `from()`, `to()`, `mover()` |
| `MCTS` | `search.mcts` | `createUCT()`, `initAI(Game, int)`, `closeAI()` |
| `AlphaBetaSearch` | `search.minimax` | `selectAction(...)`, `setHeuristics(Heuristics)` |
| `Heuristics` | `metadata.ai.heuristics` | `Heuristics(HeuristicTerm[])`, `init(Game)` |
| `HeuristicTerm` | `metadata.ai.heuristics.terms` | `computeValue(Context, int, float)` |
| `Material` | `metadata.ai.heuristics.terms` | Standard material balance |
| `MobilityAdvanced` | `metadata.ai.heuristics.terms` | Weighted move count |
| `GameLoader` | `other` | `loadGameFromFile(File)` |
| `AIFactory` | `utils` | `createAI(String name)` |

### Ludii Agent Name Strings (for `AIFactory.createAI()`)

| String | Agent |
|---|---|
| `"Random"` | Uniform random |
| `"UCT"` | UCT / MCTS with UCB1 |
| `"Alpha-Beta"` | Alpha-Beta with default heuristic |

### `isHidden` Index Convention

**Important:** Different classes use different index conventions for `isHidden`:

- [`PPOLSTMAgent.java`](agents/src/PPOLSTMAgent.java) and [`ISMCTSAgent.java`](agents/src/ISMCTSAgent.java): use **1-based** player indices (1 = White/P1, 2 = Black/P2), matching `fow_env.py`.
- [`FoWHeuristicTerm.java`](agents/src/FoWHeuristicTerm.java): uses **0-based** indices (`player - 1`).

```java
// PPOLSTMAgent / ISMCTSAgent — 1-based:
boolean hidden = state.containerStates()[0]
        .isHidden(playerId, site, 0, SiteType.Cell);   // playerId ∈ {1, 2}

// FoWHeuristicTerm — 0-based:
boolean hidden = state.containerStates()[0]
        .isHidden(player - 1, site, 0, SiteType.Cell); // player-1 ∈ {0, 1}
```

```python
# fow_env.py — 1-based (matches PPOLSTMAgent / ISMCTSAgent):
hidden = bool(cs.isHidden(player, site, 0, SiteType.Cell))  # player ∈ {1, 2}
```

When porting to a different Ludii game, verify the `isHidden` index convention against the game's `set Hidden` implementation.

### Java Compilation Template

```bash
# Compile all agents against Ludii JAR
javac -cp fow_chess_ludii/Ludii-1.3.14.jar \
      fow_chess_ludii/agents/src/*.java \
      -d fow_chess_ludii/agents/compiled/

# Package into a single JAR
jar cf fow_chess_ludii/agents/jars/agents.jar \
    -C fow_chess_ludii/agents/compiled .

# Or use the provided build script:
bash fow_chess_ludii/evaluation/scripts/build.sh
```

### FoW Chess Game-Specific Constants

| Constant | Value | Source |
|---|---|---|
| Board size | 8×8 = 64 sites | `(board (square 8))` |
| Players | P1 (White/South), P2 (Black/North) | `("TwoPlayersNorthSouth")` |
| Draw condition | counter = 100 half-moves | `(if (= (counter) 100) (result Mover Draw))` |
| Win condition | Capture opponent's king | `(if (no Pieces Next "King") (result Mover Win))` |
| Piece types (Ludii indices) | 1=Pawn, 2=Rook, 3=Bishop, 4=Knight, 5=Queen, 6=King | `(equipment {...})` |
| Starting position | Standard chess setup | Rows 1/6 (pawns), Rows 0/7 (pieces) |
| Hard move cap (EvalRunner) | 300 plies | `--max-moves` flag default |
| Max plies (training env) | 400 | `MAX_PLIES` in `fow_env.py` |
| Observation dimension | 128 | 2 channels × 64 squares |
| Action space | 4096 | 64 × 64 from–to |

---
