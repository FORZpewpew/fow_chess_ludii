"""
eval_belief_probe.py

Evaluate a trained BeliefProbeHead on held-out game sequences.

For each position:
  1. Feed fog_obs through frozen backbone LSTM (maintaining h,c state per game)
  2. Get probe logits → argmax → predicted piece_type per square
  3. Compute Jaccard similarity on hidden squares:
       intersection = squares where predicted == true AND true != 0 (non-empty hidden pieces)
       union = squares that are hidden AND (true!=0 OR predicted!=0)
       Jaccard = |intersection| / |union|  (0.0 if union=0)

Output CSV: evaluation/results_grave/ppo_belief_accuracy.csv
  game_id, ply, num_hidden_squares, jaccard, phase

Phase:
  early   : ply  < 20
  mid     : 20 ≤ ply < 60
  late    : ply ≥ 60

Run:
  cd /Users/forzpewpew/Downloads/ludii
  python ppo/eval_belief_probe.py [--split val|all] [--max-games N]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from policy_lstm import FoWPolicyLSTM
from belief_probe import BeliefProbeHead

BASE       = Path(__file__).parent.parent
DATA_PATH  = BASE / "training" / "results" / "probe_training_data.jsonl"
CKPT_LSTM  = BASE / "checkpoints" / "ppo_lstm_v4_policy.pt"
CKPT_PROBE = BASE / "checkpoints" / "belief_probe_v4.pt"
OUT_DIR    = BASE / "evaluation" / "results_grave"
OUT_CSV    = OUT_DIR / "ppo_belief_accuracy.csv"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_FRAC = 0.9
SEED       = 42


def phase_label(ply: int) -> str:
    if ply < 20:
        return "early"
    elif ply < 60:
        return "mid"
    return "late"


def jaccard(pred: np.ndarray, true: np.ndarray, hidden: np.ndarray) -> float:
    """
    Compute Jaccard similarity over hidden squares only.

    pred, true, hidden: (64,) int arrays
    Considers piece presence accuracy: a prediction is correct when both
    the predicted and true piece types match and are non-empty.
    """
    hidden_idx = np.where(hidden == 1)[0]
    if len(hidden_idx) == 0:
        return float("nan")
    p = pred[hidden_idx]
    t = true[hidden_idx]
    intersection = np.sum((p == t) & (t != 0))
    union = np.sum((t != 0) | (p != 0))
    if union == 0:
        return float("nan")
    return float(intersection) / float(union)


def load_backbone(ckpt_path: Path) -> FoWPolicyLSTM:
    policy = FoWPolicyLSTM()
    state  = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "policy_state_dict" in state:
        state = state["policy_state_dict"]
    policy.load_state_dict(state, strict=False)
    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()
    return policy.to(DEVICE)


def load_probe(ckpt_path: Path) -> BeliefProbeHead:
    probe = BeliefProbeHead()
    probe.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    probe.eval()
    return probe.to(DEVICE)


def get_eval_games(data_path: Path, split: str = "val", max_games=None) -> dict:
    """Load records grouped by game_id. Returns {game_id: [records sorted by ply]}."""
    by_game = defaultdict(list)
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            by_game[r["game_id"]].append(r)

    game_ids = sorted(by_game.keys())

    if split == "val":
        rng = random.Random(SEED)
        rng.shuffle(game_ids)
        game_ids = game_ids[int(len(game_ids) * TRAIN_FRAC):]

    if max_games is not None:
        game_ids = game_ids[:max_games]

    return {gid: sorted(by_game[gid], key=lambda r: r["ply"]) for gid in game_ids}


def evaluate(backbone: FoWPolicyLSTM, probe: BeliefProbeHead, games: dict) -> list:
    """Evaluate probe on game dict, return list of row dicts."""
    rows    = []
    n_games = len(games)

    for i, (gid, seq) in enumerate(sorted(games.items())):
        if (i + 1) % 50 == 0:
            print(f"  Game {i+1}/{n_games} …")

        # Evaluate each player's perspective separately
        by_player = defaultdict(list)
        for r in seq:
            by_player[r["player"]].append(r)

        for player, precs in by_player.items():
            precs_sorted = sorted(precs, key=lambda r: r["ply"])
            h, c = backbone.init_hidden(batch_size=1, device=DEVICE)

            for r in precs_sorted:
                fog = torch.tensor(r["fog_obs"], dtype=torch.float32,
                                   device=DEVICE).unsqueeze(0)  # (1, 128)

                with torch.no_grad():
                    enc     = backbone.encoder(fog)
                    out, (h, c) = backbone.lstm(enc.unsqueeze(1), (h, c))
                    h_t     = out.squeeze(1)       # (1, 512)
                    logits  = probe(h_t)            # (1, 64, 7)
                    pred    = logits.argmax(-1).squeeze(0).cpu().numpy()  # (64,)

                true  = np.array(r["true_board"],  dtype=np.int32)
                hid   = np.array(r["hidden_mask"], dtype=np.int32)
                ply   = r["ply"]

                rows.append({
                    "game_id":            gid,
                    "player":             player,
                    "ply":                ply,
                    "num_hidden_squares": int(hid.sum()),
                    "jaccard":            jaccard(pred, true, hid),
                    "phase":              phase_label(ply),
                })

    return rows


def summarise(rows: list) -> pd.DataFrame:
    df       = pd.DataFrame(rows)
    df_valid = df.dropna(subset=["jaccard"])
    print(f"\nTotal positions evaluated: {len(df)}")
    print(f"Positions with hidden pieces (non-NaN Jaccard): {len(df_valid)}")

    print("\nMean Jaccard by phase:")
    for phase in ["early", "mid", "late"]:
        sub = df_valid[df_valid["phase"] == phase]
        if len(sub):
            print(f"  {phase:6s}: {sub['jaccard'].mean():.4f} ± {sub['jaccard'].std():.4f}  (n={len(sub)})")
        else:
            print(f"  {phase:6s}: no data")

    print(f"\nOverall mean Jaccard: {df_valid['jaccard'].mean():.4f}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",     choices=["val", "all"], default="val")
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    ckpt_path = CKPT_LSTM
    if not ckpt_path.exists():
        alt = ckpt_path.parent / "ppo_lstm_pretrained_v4.pt"
        ckpt_path = alt if alt.exists() else ckpt_path

    print(f"Loading backbone from {ckpt_path} …")
    backbone = load_backbone(ckpt_path)

    print(f"Loading probe from {CKPT_PROBE} …")
    probe = load_probe(CKPT_PROBE)

    print(f"Loading eval games ({args.split} split) …")
    games = get_eval_games(DATA_PATH, split=args.split, max_games=args.max_games)
    print(f"  {len(games)} games loaded")

    print("Evaluating …")
    rows = evaluate(backbone, probe, games)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = summarise(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
