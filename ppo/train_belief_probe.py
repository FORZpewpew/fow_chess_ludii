"""
train_belief_probe.py

Train a BeliefProbeHead on top of a frozen FoWPolicyLSTM backbone.

Pipeline:
  1. Load frozen LSTM backbone from checkpoints/ppo_lstm_v4_policy.pt
  2. For each game sequence in probe_training_data.jsonl:
       - Reset (h, c) = zeros
       - Feed fog_obs through encoder + LSTM (no grad)
       - At every step collect h_t for ALL squares (probe runs on full board)
       - Target: true_board[sq] for every square (hidden or not)
         → Loss computed only on HIDDEN squares (is_hidden=1)
  3. Loss: cross-entropy(predicted_piece_type, true_piece_type) over hidden squares
  4. Sequence-aware training: propagates LSTM state across full game sequences
  5. Save best checkpoint to checkpoints/belief_probe_v4.pt

Run:
  cd /Users/forzpewpew/Downloads/ludii
  python ppo/train_belief_probe.py [--epochs 20] [--lr 1e-3] [--batch-size 256]
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from policy_lstm import FoWPolicyLSTM
from belief_probe import BeliefProbeHead

BASE       = Path(__file__).parent.parent
DATA_PATH  = BASE / "training" / "results" / "probe_training_data.jsonl"
CKPT_LSTM  = BASE / "checkpoints" / "ppo_lstm_v4_policy.pt"
CKPT_PROBE = BASE / "checkpoints" / "belief_probe_v4.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ProbeDataset(Dataset):
    """
    Each item is a single (fog_obs, true_board, hidden_mask) tuple.

    The LSTM sequencing is handled separately during training via game grouping
    (see train_epoch_sequence). This dataset exposes flat items for mini-batch
    validation (no LSTM state continuity needed there).
    """
    def __init__(self, records):
        self.fog_obs     = torch.tensor(np.array([r["fog_obs"]     for r in records]), dtype=torch.float32)
        self.true_board  = torch.tensor(np.array([r["true_board"]  for r in records]), dtype=torch.long)
        self.hidden_mask = torch.tensor(np.array([r["hidden_mask"] for r in records]), dtype=torch.bool)
        self.game_ids    = [r["game_id"] for r in records]
        self.plies       = [r["ply"]     for r in records]

    def __len__(self):
        return len(self.game_ids)

    def __getitem__(self, idx):
        return self.fog_obs[idx], self.true_board[idx], self.hidden_mask[idx]


def load_data(path: Path, train_frac=0.9, seed=42):
    print(f"Loading {path} …")
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Total records: {len(records)}")

    by_game  = defaultdict(list)
    for r in records:
        by_game[r["game_id"]].append(r)
    game_ids = sorted(by_game.keys())

    rng = random.Random(seed)
    rng.shuffle(game_ids)
    split     = int(len(game_ids) * train_frac)
    train_ids = set(game_ids[:split])
    val_ids   = set(game_ids[split:])

    train_records = [r for r in records if r["game_id"] in train_ids]
    val_records   = [r for r in records if r["game_id"] in val_ids]
    print(f"  Train: {len(train_records)} records from {len(train_ids)} games")
    print(f"  Val  : {len(val_records)} records from {len(val_ids)} games")

    # Sequence groups preserve temporal order within each game for LSTM state propagation
    train_seqs = {gid: sorted(by_game[gid], key=lambda r: r["ply"]) for gid in train_ids}
    val_seqs   = {gid: sorted(by_game[gid], key=lambda r: r["ply"]) for gid in val_ids}

    return ProbeDataset(train_records), ProbeDataset(val_records), train_seqs, val_seqs


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


def compute_probe_loss_sequence(backbone, probe, seq_records, criterion):
    """
    Feed a single game sequence through backbone LSTM (maintaining h, c state)
    and compute cross-entropy loss on hidden squares only.

    Returns (total_loss_tensor, num_hidden_positions).
    """
    h, c       = backbone.init_hidden(batch_size=1, device=DEVICE)
    total_loss = torch.tensor(0.0, device=DEVICE)
    n_hidden   = 0

    for r in seq_records:
        fog = torch.tensor(r["fog_obs"], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            enc = backbone.encoder(fog)
            out, (h, c) = backbone.lstm(enc.unsqueeze(1), (h, c))
            h_t = out.squeeze(1)  # (1, 512)

        logits = probe(h_t)  # (1, 64, 7)

        hidden_mask = torch.tensor(r["hidden_mask"], dtype=torch.bool, device=DEVICE)
        true_board  = torch.tensor(r["true_board"],  dtype=torch.long, device=DEVICE)

        hidden_sites = hidden_mask.nonzero(as_tuple=True)[0]
        if len(hidden_sites) == 0:
            continue

        loss       = criterion(logits[0][hidden_sites], true_board[hidden_sites])
        total_loss = total_loss + loss
        n_hidden  += len(hidden_sites)

    return total_loss, n_hidden


def train_epoch_sequence(backbone, probe, seq_dict, optimizer, criterion):
    """
    Sequence-aware training: propagates LSTM hidden state across each game.

    For each game:
      1. Reset (h, c) = zeros
      2. Feed positions sequentially through backbone encoder + LSTM
      3. Compute probe cross-entropy on hidden squares at each step
      4. Accumulate loss over the full game, then backprop once per game

    This matches the LSTM state distribution seen during evaluation.
    """
    probe.train()
    total_loss   = 0.0
    total_hidden = 0

    game_ids = list(seq_dict.keys())
    random.shuffle(game_ids)

    for gid in game_ids:
        seq_records = seq_dict[gid]
        if not seq_records:
            continue

        loss_game, n_hidden = compute_probe_loss_sequence(backbone, probe, seq_records, criterion)
        if n_hidden == 0:
            continue

        avg_loss = loss_game / n_hidden
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        total_loss   += avg_loss.item() * n_hidden
        total_hidden += n_hidden

    return total_loss / max(total_hidden, 1)


def train_epoch_flat(backbone, probe, loader, optimizer, criterion):
    """
    Fast flat-batch training: ignores LSTM state continuity (h=0 per sample).
    Useful for probe weight pretraining before sequence-aware fine-tuning.
    """
    probe.train()
    total_loss   = 0.0
    total_hidden = 0

    for fog_obs, true_board, hidden_mask in loader:
        fog_obs     = fog_obs.to(DEVICE)
        true_board  = true_board.to(DEVICE)
        hidden_mask = hidden_mask.to(DEVICE)

        B = fog_obs.size(0)

        with torch.no_grad():
            enc = backbone.encoder(fog_obs)
            h0, c0 = backbone.init_hidden(B, DEVICE)
            out, _ = backbone.lstm(enc.unsqueeze(1), (h0, c0))
            h_t    = out.squeeze(1)  # (B, 512)

        logits = probe(h_t)  # (B, 64, 7)

        logits_flat = logits.view(B * 64, 7)
        true_flat   = true_board.view(B * 64)
        hidden_flat = hidden_mask.view(B * 64)

        if hidden_flat.sum() == 0:
            continue

        loss = criterion(logits_flat[hidden_flat], true_flat[hidden_flat])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_h           = hidden_flat.sum().item()
        total_loss   += loss.item() * n_h
        total_hidden += n_h

    return total_loss / max(total_hidden, 1)


def eval_epoch_flat(backbone, probe, loader, criterion):
    probe.eval()
    total_loss   = 0.0
    total_hidden = 0
    correct      = 0

    with torch.no_grad():
        for fog_obs, true_board, hidden_mask in loader:
            fog_obs     = fog_obs.to(DEVICE)
            true_board  = true_board.to(DEVICE)
            hidden_mask = hidden_mask.to(DEVICE)

            B = fog_obs.size(0)
            enc = backbone.encoder(fog_obs)
            h0, c0 = backbone.init_hidden(B, DEVICE)
            out, _ = backbone.lstm(enc.unsqueeze(1), (h0, c0))
            h_t    = out.squeeze(1)

            logits = probe(h_t)

            logits_flat = logits.view(B * 64, 7)
            true_flat   = true_board.view(B * 64)
            hidden_flat = hidden_mask.view(B * 64)

            if hidden_flat.sum() == 0:
                continue

            pred_h = logits_flat[hidden_flat]
            true_h = true_flat[hidden_flat]

            n_h           = hidden_flat.sum().item()
            total_loss   += criterion(pred_h, true_h).item() * n_h
            total_hidden += n_h
            correct      += (pred_h.argmax(-1) == true_h).sum().item()

    return total_loss / max(total_hidden, 1), correct / max(total_hidden, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {DEVICE}")

    train_ds, val_ds, train_seqs, val_seqs = load_data(DATA_PATH)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt_path = CKPT_LSTM
    if not ckpt_path.exists():
        alt = ckpt_path.parent / "ppo_lstm_pretrained_v4.pt"
        if alt.exists():
            print(f"Primary checkpoint not found, using {alt}")
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"No checkpoint found: {CKPT_LSTM}")
    print(f"Loading backbone from {ckpt_path} …")
    backbone = load_backbone(ckpt_path)
    print("  Backbone loaded & frozen.")

    probe     = BeliefProbeHead().to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nProbe parameters: {sum(p.numel() for p in probe.parameters()):,}")
    print(f"Training for {args.epochs} epochs …\n")

    best_val_loss = float("inf")
    history       = []

    for epoch in range(1, args.epochs + 1):
        # Sequence-aware training matches the LSTM state distribution seen at eval time
        train_loss            = train_epoch_sequence(backbone, probe, train_seqs, optimizer, criterion)
        val_loss, val_acc     = eval_epoch_flat(backbone, probe, val_loader, criterion)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        })

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(probe.state_dict(), CKPT_PROBE)
            print(f"             ↑ saved best probe → {CKPT_PROBE}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Final epoch train_loss: {history[-1]['train_loss']:.4f}")

    hist_path = BASE / "checkpoints" / "belief_probe_v4_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved → {hist_path}")


if __name__ == "__main__":
    main()
