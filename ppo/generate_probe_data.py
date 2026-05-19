"""
generate_probe_data.py

Runs N self-play games using FoWChessEnv + the trained PPO-LSTM policy,
recording for each step BOTH the fog observation AND the true board state.

Key insight: The Java context has the full board regardless of fog. We can
call cs.what(site, SiteType.Cell) WITHOUT the hidden check to get the true
piece type for every square.

Ludii piece_type → canonical probe class (7 classes):
  0           → 0 (empty)
  1, 2        → 1 (pawn)
  3, 4        → 2 (rook)
  5, 6        → 6 (king)
  7, 8        → 3 (bishop)
  9, 10       → 4 (knight)
  11, 12      → 5 (queen)

Output: training/results/probe_training_data.jsonl
  game_id, ply, player, fog_obs[128], true_board[64], hidden_mask[64], outcome

Run:
  cd /Users/forzpewpew/Downloads/ludii
  ppo/venv/bin/python ppo/generate_probe_data.py [--num-games 50]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import json
import argparse
import numpy as np
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
GAME_PATH  = str(BASE / "FoW_Chess.lud")
LUDII_JAR  = str(BASE / "Ludii-1.3.14.jar")
AGENTS_JAR = str(BASE / "agents" / "jars" / "agents.jar")
JAVA_HOME  = "/Users/forzpewpew/.asdf/installs/java/openjdk-26"
CKPT_PATH  = BASE / "checkpoints" / "ppo_lstm_v4_policy.pt"
DST        = BASE / "training" / "results" / "probe_training_data.jsonl"
MAX_PLIES  = 400

# Ludii component index → canonical 7-class piece type
# 0=empty, 1=pawn, 2=rook, 3=bishop, 4=knight, 5=queen, 6=king
LUDII_TO_CLASS = {
    0: 0,    # empty
    1: 1, 2: 1,    # pawn
    3: 2, 4: 2,    # rook
    5: 6, 6: 6,    # king
    7: 3, 8: 3,    # bishop
    9: 4, 10: 4,   # knight
    11: 5, 12: 5,  # queen
}


# ── JVM + Ludii bootstrap ─────────────────────────────────────────────────────
def start_jvm():
    import jpype
    import jpype.imports
    if jpype.isJVMStarted():
        return
    jvm_path = os.path.join(JAVA_HOME, "lib", "server", "libjvm.dylib")
    jpype.startJVM(
        jvm_path,
        f"-Djava.class.path={LUDII_JAR}:{AGENTS_JAR}",
        f"-Djava.home={JAVA_HOME}",
        "-Xmx2g",
        convertStrings=False,
    )


# ── Policy wrapper ────────────────────────────────────────────────────────────
def load_policy():
    import torch
    from policy_lstm import FoWPolicyLSTM
    policy = FoWPolicyLSTM()
    if CKPT_PATH.exists():
        state = torch.load(CKPT_PATH, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "policy_state_dict" in state:
            state = state["policy_state_dict"]
        policy.load_state_dict(state, strict=False)
        print(f"  Loaded policy from {CKPT_PATH}")
    else:
        alt = CKPT_PATH.parent / "ppo_lstm_pretrained_v4.pt"
        if alt.exists():
            state = torch.load(alt, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "policy_state_dict" in state:
                state = state["policy_state_dict"]
            policy.load_state_dict(state, strict=False)
            print(f"  Loaded policy from {alt}")
        else:
            print("  WARNING: No checkpoint found, using random policy!")
    policy.eval()
    return policy


def make_action_fn(policy):
    """Return a callable(obs, legal_mask) -> action_idx using the LSTM policy."""
    import torch
    # Each call is independent (no LSTM state persistence) to keep the wrapper simple.
    # The LSTM is properly stepped inside play_game() below.
    def fn(obs, legal_mask):
        obs_t  = torch.tensor(obs,        dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(legal_mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            dist, _, _, _ = policy(obs_t, mask_t)
        return int(dist.sample().item())
    return fn


# ── True board extraction (no fog) ───────────────────────────────────────────
def get_true_board(context, SiteType):
    """Read true piece_type for all 64 squares, ignoring fog."""
    state = context.state()
    cs    = state.containerStates()[0]
    board = []
    for site in range(64):
        raw_type = int(cs.what(site, SiteType.Cell))
        board.append(LUDII_TO_CLASS.get(raw_type, 0))
    return board


def get_fog_obs_and_mask(context, player, SiteType):
    """Return (fog_obs[128], hidden_mask[64]) for the given player.

    Layout: [owner_ch×64, piece_type_ch×64] — 2 channels × 64 squares = 128 values.
    Owner encoding (egocentric 4-value):
      0.000 = hidden, 1/3 = empty, 2/3 = own piece, 1.0 = opponent piece.
    """
    state    = context.state()
    cs       = state.containerStates()[0]
    owner_ch = np.zeros(64, dtype=np.float32)
    piece_ch = np.zeros(64, dtype=np.float32)
    mask     = np.zeros(64, dtype=np.int8)
    for site in range(64):
        owner      = int(cs.who(site, SiteType.Cell))
        piece_type = int(cs.what(site, SiteType.Cell))
        hidden     = bool(cs.isHidden(player, site, 0, SiteType.Cell))
        if hidden:
            # owner_ch[site] stays 0.0 (hidden sentinel)
            mask[site] = 1
        else:
            if owner == 0:
                owner_ch[site] = 1.0 / 3.0   # confirmed empty
            elif owner == player:
                owner_ch[site] = 2.0 / 3.0   # own piece
            else:
                owner_ch[site] = 1.0          # opponent piece
            piece_ch[site] = float(piece_type) / 6.0
    return np.concatenate([owner_ch, piece_ch]).tolist(), mask.tolist()


def legal_mask_for(context, game, player, ACTION_DIM=4096):
    state = context.state()
    if int(state.mover()) != player:
        return np.zeros(ACTION_DIM, dtype=bool)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    for m in game.moves(context).moves():
        f = int(m.fromNonDecision())
        t = int(m.toNonDecision())
        if 0 <= f < 64 and 0 <= t < 64:
            mask[f * 64 + t] = True
    return mask


# ── Self-play game runner ─────────────────────────────────────────────────────
def play_game(game_id, game, Context, Trial, SiteType, policy, rng):
    """Run one full self-play game, returning a list of step records."""
    import torch
    trial   = Trial(game)
    context = Context(game, trial)
    game.start(context)
    ply   = 0
    records = []

    # Separate LSTM states for each player
    h = {1: torch.zeros(1, 1, 512), 2: torch.zeros(1, 1, 512)}
    c = {1: torch.zeros(1, 1, 512), 2: torch.zeros(1, 1, 512)}

    while not trial.over() and ply < MAX_PLIES:
        mover = int(context.state().mover())

        # Get fog obs and legal mask for the current mover
        fog_obs, hidden_mask = get_fog_obs_and_mask(context, mover, SiteType)
        true_board           = get_true_board(context, SiteType)
        legal                = legal_mask_for(context, game, mover)

        # Record this position
        records.append({
            "game_id":     game_id,
            "ply":         ply,
            "player":      mover,
            "fog_obs":     fog_obs,
            "true_board":  true_board,
            "hidden_mask": hidden_mask,
            "outcome":     None,   # filled in after game ends
        })

        # Select action using policy (with proper LSTM state)
        obs_t  = torch.tensor(fog_obs,  dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(legal,    dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            dist, _, h[mover], c[mover] = policy(obs_t, mask_t,
                                                   h[mover], c[mover])
        action_idx = int(dist.sample().item())

        from_site = action_idx // 64
        to_site   = action_idx  % 64

        # Find matching legal move
        moves = list(game.moves(context).moves())
        chosen = None
        for m in moves:
            if m.fromNonDecision() == from_site and m.toNonDecision() == to_site:
                chosen = m
                break
        if chosen is None and moves:
            chosen = moves[rng.integers(len(moves))]
        if chosen is None:
            break

        game.apply(context, chosen)
        ply += 1

    # Determine outcome
    ranking = trial.ranking()
    outcomes = {1: 0.0, 2: 0.0}
    if ranking is not None and len(ranking) > 2:
        r1, r2 = float(ranking[1]), float(ranking[2])
        if r1 < r2:
            outcomes = {1: 1.0, 2: -1.0}
        elif r2 < r1:
            outcomes = {1: -1.0, 2: 1.0}

    for rec in records:
        rec["outcome"] = outcomes[rec["player"]]

    return records


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=50, help="Number of self-play games")
    args = parser.parse_args()

    print("Starting JVM …")
    start_jvm()
    print("JVM started.")

    import jpype
    from other import GameLoader
    from other.context import Context
    from other.trial import Trial
    from java.io import File
    from game.types.board import SiteType

    print("Loading game …")
    game = GameLoader.loadGameFromFile(File(GAME_PATH))
    print(f"  Game: {game.name()}")

    print("Loading policy …")
    policy = load_policy()

    rng = np.random.default_rng(42)
    DST.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    print(f"\nRunning {args.num_games} self-play games …")
    with open(DST, "w") as out:
        for game_id in range(1, args.num_games + 1):
            if game_id % 10 == 0 or game_id == 1:
                print(f"  Game {game_id}/{args.num_games} …")
            records = play_game(game_id, game, Context, Trial, SiteType, policy, rng)
            for rec in records:
                out.write(json.dumps(rec) + "\n")
            total_records += len(records)

    print(f"\nDone. Wrote {total_records} records to {DST}")

    # Sanity check
    with open(DST) as f:
        sample = json.loads(f.readline())
    print(f"\nFirst record: game={sample['game_id']} ply={sample['ply']} player={sample['player']}")
    print(f"  fog_obs len={len(sample['fog_obs'])}")
    print(f"  true_board len={len(sample['true_board'])} non-zero={sum(x>0 for x in sample['true_board'])}")
    print(f"  hidden_squares={sum(sample['hidden_mask'])}")


if __name__ == "__main__":
    main()
