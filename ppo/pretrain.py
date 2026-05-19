"""
pretrain.py — Two-phase pretraining for FoWPolicyLSTM.

Phase 1 — Behavioural Cloning (BC) from selfplay_training_data.jsonl
──────────────────────────────────────────────────────────────────────
  Dataset: ludii/training/results/selfplay_training_data.jsonl
  Each line: {"obs": [128 floats], "action": int, "result": float}
    result: +1 win / -1 loss / 0 draw from the acting player's perspective.

  Loss: cross_entropy(action_logits, action) + 0.5 * MSE(critic_output, result)
  Hidden state is all-zeros for every sample (individual positions, not sequences).
  Trains for NUM_EPOCHS epochs (default 50), with validation loss tracking and
  early stopping (patience = EARLY_STOP_PATIENCE epochs).
  Saves best checkpoint to checkpoints/ppo_lstm_pretrained_v5.pt.
  Also saves a checkpoint every 5 epochs to checkpoints/pretrain_v5_ckpt_{epoch}.pt.

Phase 2 — PPO fine-tuning (always runs after Phase 1)
──────────────────────────────────────────────────────────────────
  Loads Phase 1 checkpoint as warm start.
  Runs --ppo_updates PPO updates using train_ppo_lstm.train().
  Overwrites checkpoints/ppo_lstm_pretrained_v5.pt with the final policy.

Usage:
  cd /Users/forzpewpew/Downloads/ludii
  # BC only (50 epochs):
  python3 ppo/pretrain.py

  # BC + PPO fine-tune:
  python3 ppo/pretrain.py --ppo_finetune --ppo_updates 50

  # Quick smoke-test:
  python3 ppo/pretrain.py --epochs 1 --batch_size 256
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ppo.policy_lstm import FoWPolicyLSTM

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH   = Path(__file__).parent.parent / "training" / "results" / "selfplay_training_data.jsonl"

# v4 checkpoints (original location, kept for Phase 2 reference)
CKPT_DIR    = Path(__file__).parent / "checkpoints"
CKPT_BC     = CKPT_DIR / "ppo_lstm_pretrained_v4.pt"

# v5 checkpoints — save to ludii/checkpoints/ (one level above ppo/)
CKPT_DIR_V5       = Path(__file__).parent.parent / "checkpoints"
CKPT_BC_V5        = CKPT_DIR_V5 / "ppo_lstm_pretrained_v5.pt"
CKPT_V5_EPOCH_FMT = str(CKPT_DIR_V5 / "pretrain_v5_ckpt_{epoch}.pt")

NUM_EPOCHS          = 50     # v5: 50 epochs (was 5 in v4)
BC_EPOCHS           = NUM_EPOCHS
BC_BATCH            = 256
BC_LR               = 1e-4
VAL_COEFF           = 0.5
VAL_SPLIT           = 0.10   # 10 % of games reserved for validation
EARLY_STOP_PATIENCE = 10     # stop if val_loss does not improve for this many epochs
CKPT_EVERY          = 5      # save a periodic checkpoint every N epochs

PPO_UPDATES = 300  # was 100 (v4 enhancement)


# ── Data loading ──────────────────────────────────────────────────────────────
def _board_to_obs(cells, player: int) -> list:
    """
    Convert board.cells list (64 dicts with owner/piece_type/hidden) to a
    128-dim float32 observation vector matching FoWChessEnv._observe_for():

    Layout: [owner_0, owner_1, …, owner_63, pt_0, pt_1, …, pt_63]  (2 × 64 = 128 values)

      owner channel (indices 0–63):
                   0.000  hidden square (fog)
                   1/3    confirmed empty visible square
                   2/3    square occupied by this player's own piece
                   1.000  square visibly occupied by the opponent
      piece_type channel (indices 64–127): piece_type / 6.0  (0 if hidden or absent)
    """
    owner_ch = []
    piece_ch = []
    for cell in cells:
        hidden = bool(cell["hidden"])
        if hidden:
            owner_ch.append(0.0)   # hidden sentinel
            piece_ch.append(0.0)   # unknown piece
        else:
            owner = int(cell["owner"])
            if owner == 0:
                ch0 = 1.0 / 3.0   # confirmed empty
            elif owner == player:
                ch0 = 2.0 / 3.0   # own piece
            else:
                ch0 = 1.0          # opponent piece
            owner_ch.append(ch0)
            piece_ch.append(float(cell["piece_type"]) / 6.0)
    return owner_ch + piece_ch  # length 128


def load_dataset(path: Path):
     """
     Load selfplay_training_data.jsonl into a list of samples (dicts).

     The dataset schema is:
       game_id, ply, player, outcome, board{cells[64]}, legal_move_count, move_history[{from,to,mover}]

     We reconstruct:
       obs    ← board.cells → 128-dim vector
       action ← last move in move_history where mover == player → from*64+to
       result ← outcome
       game_id, ply ← from sample (for sequential BC)

     Samples where the acting player has no move in move_history are skipped.

     Returns:
         all_data: list of dicts with keys: obs, action, result, game_id, ply
     """
     all_data = []
     skipped = 0
     has_game_id = False
     has_ply = False
     
     print(f"[Pretrain] Loading dataset from {path} ...", flush=True)
     with open(path, "r") as fh:
         for i, line in enumerate(fh):
             line = line.strip()
             if not line:
                 continue
             sample = json.loads(line)

             player = sample["player"]
             # Find the last move made by this player in move_history
             action = None
             for move in reversed(sample.get("move_history", [])):
                 if move["mover"] == player:
                     action = move["from"] * 64 + move["to"]
                     break
             if action is None:
                 skipped += 1
                 continue

             # Check for game_id and ply fields
             game_id = sample.get("game_id", str(i))
             ply = sample.get("ply", i)
             if "game_id" in sample:
                 has_game_id = True
             if "ply" in sample:
                 has_ply = True

             all_data.append({
                 "obs": _board_to_obs(sample["board"]["cells"], player=player),
                 "action": action,
                 "result": float(sample["outcome"]),
                 "game_id": game_id,
                 "ply": ply,
             })

     print(f"[Pretrain] Loaded {len(all_data):,} samples (skipped {skipped:,} without action).", flush=True)
     if not has_game_id or not has_ply:
         print(f"[Pretrain] WARNING: Dataset missing game_id/ply fields. Will treat all samples as single sequence.", flush=True)
     
     return all_data


# ── Validation loss computation ───────────────────────────────────────────────
def _compute_val_loss(policy: FoWPolicyLSTM,
                      val_games: dict,
                      device: torch.device) -> float:
    """
    Compute mean BC loss on the validation set (no gradients).
    Mirrors the sequential forward pass of run_bc() but uses torch.no_grad().
    """
    policy.eval()
    total_loss = 0.0
    game_count = 0

    with torch.no_grad():
        for gid, samples in val_games.items():
            h, c = policy.init_hidden(1, device)

            game_loss = 0.0
            for sample in samples:
                obs_np   = np.array(sample["obs"], dtype=np.float32)
                act_i    = sample["action"]
                result_f = sample["result"]

                obs_t    = torch.tensor(obs_np,   dtype=torch.float32, device=device).unsqueeze(0)
                act_t    = torch.tensor(act_i,    dtype=torch.long,    device=device).unsqueeze(0)
                result_t = torch.tensor(result_f, dtype=torch.float32, device=device).unsqueeze(0)

                legal_mask = torch.ones(1, 4096, dtype=torch.bool, device=device)

                dist, value, h, c = policy(obs_t, legal_mask, h, c)

                logits_softened = dist.logits / 2.0
                p_loss = F.cross_entropy(logits_softened, act_t)
                v_loss = F.mse_loss(value.squeeze(), result_t.squeeze())

                game_loss += (p_loss + VAL_COEFF * v_loss).item()

            total_loss += game_loss
            game_count += 1

    policy.train()
    return total_loss / max(game_count, 1)


# ── Phase 1: Behavioural Cloning ──────────────────────────────────────────────
def run_bc(epochs: int = NUM_EPOCHS,
           batch_size: int = BC_BATCH,
           device: torch.device = None) -> FoWPolicyLSTM:
     """
     Train FoWPolicyLSTM with sequential behavioural cloning on the self-play dataset.
     
     v5 enhancements over v4:
       - 50 epochs (up from 5)
       - 10% validation split for per-epoch val_loss tracking
       - Early stopping (patience = EARLY_STOP_PATIENCE)
       - Best checkpoint saved to CKPT_BC_V5
       - Periodic checkpoints every CKPT_EVERY epochs

     Returns the trained policy (also saved to CKPT_BC_V5).
     """
     if device is None:
         device = torch.device("cpu")

     CKPT_DIR.mkdir(parents=True, exist_ok=True)
     CKPT_DIR_V5.mkdir(parents=True, exist_ok=True)

     all_data = load_dataset(DATA_PATH)
     
     # Group samples by game_id, sort by ply within each game
     games = defaultdict(list)
     for i, sample in enumerate(all_data):
         games[sample.get("game_id", str(i))].append(sample)

     # Sort each game's samples by ply
     for gid in games:
         games[gid].sort(key=lambda x: x.get("ply", 0))

     game_ids = list(games.keys())

     # ── Train / validation split (done once, before training) ────────────────
     random.shuffle(game_ids)
     n_val = max(1, int(len(game_ids) * VAL_SPLIT))
     val_ids   = game_ids[:n_val]
     train_ids = game_ids[n_val:]
     val_games   = {gid: games[gid] for gid in val_ids}
     train_games = {gid: games[gid] for gid in train_ids}

     print(
         f"[Pretrain] Phase 1 — Sequential BC for {epochs} epoch(s), "
         f"{len(train_ids)} train games / {len(val_ids)} val games, lr={BC_LR}",
         flush=True,
     )
     print(
         f"[Pretrain] Early stopping patience={EARLY_STOP_PATIENCE}, "
         f"checkpoint every {CKPT_EVERY} epochs → {CKPT_BC_V5}",
         flush=True,
     )

     policy    = FoWPolicyLSTM().to(device)
     optimizer = torch.optim.Adam(policy.parameters(), lr=BC_LR)

     best_val_loss   = float("inf")
     best_epoch      = 0
     epochs_no_improve = 0

     for epoch in range(1, epochs + 1):
         policy.train()
         total_p_loss = total_v_loss = 0.0
         game_count = 0

         shuffled_train = list(train_ids)
         random.shuffle(shuffled_train)
         
         for gid in shuffled_train:
             samples = train_games[gid]
             h, c = policy.init_hidden(1, device)
             h, c = h.detach(), c.detach()

             game_loss = 0.0
             optimizer.zero_grad()

             for step_idx, sample in enumerate(samples):
                 # Parse sample
                 obs_np = np.array(sample["obs"], dtype=np.float32)
                 act_i = sample["action"]
                 result_f = sample["result"]

                 obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 128)
                 act_t = torch.tensor(act_i, dtype=torch.long, device=device).unsqueeze(0)      # (1,)
                 result_t = torch.tensor(result_f, dtype=torch.float32, device=device).unsqueeze(0)  # (1,)

                 # All-True legal mask (BC: trust dataset)
                 legal_mask = torch.ones(1, 4096, dtype=torch.bool, device=device)

                 # Forward pass with carried state
                 dist, value, h, c = policy(obs_t, legal_mask, h, c)
                 # dist.logits: (1, 4096); value: (1,)

                 # Soft-label BC loss (temperature=2.0 to prevent entropy collapse)
                 logits_softened = dist.logits / 2.0
                 p_loss = F.cross_entropy(logits_softened, act_t)

                 # Value loss toward game result
                 v_loss = F.mse_loss(value.squeeze(), result_t.squeeze())

                 step_loss = p_loss + VAL_COEFF * v_loss
                 step_loss.backward(retain_graph=True)
                 game_loss += step_loss.item()

                 # TBPTT: detach every 16 steps
                 if (step_idx + 1) % 16 == 0:
                     h, c = h.detach(), c.detach()

             # Gradient clipping and step
             torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
             optimizer.step()
             optimizer.zero_grad()
             h, c = h.detach(), c.detach()

             total_p_loss += game_loss
             game_count += 1

         train_loss = total_p_loss / max(game_count, 1)

         # ── Validation loss (no gradients) ────────────────────────────────────
         val_loss = _compute_val_loss(policy, val_games, device)

         print(
             f"[Pretrain] [Epoch {epoch}/{epochs}]"
             f"  train_loss={train_loss:.4f}"
             f"  val_loss={val_loss:.4f}",
             flush=True,
         )

         # ── Periodic checkpoint every CKPT_EVERY epochs ───────────────────────
         if epoch % CKPT_EVERY == 0:
             periodic_ckpt = Path(CKPT_V5_EPOCH_FMT.format(epoch=epoch))
             torch.save(policy.state_dict(), periodic_ckpt)
             print(f"[Pretrain] Periodic checkpoint saved: {periodic_ckpt}", flush=True)

         # ── Best-model tracking & early stopping ─────────────────────────────
         if val_loss < best_val_loss:
             best_val_loss = val_loss
             best_epoch = epoch
             epochs_no_improve = 0
             torch.save(policy.state_dict(), CKPT_BC_V5)
             print(
                 f"[Pretrain] ✓ New best val_loss={best_val_loss:.4f} at epoch {epoch}."
                 f"  Checkpoint: {CKPT_BC_V5}",
                 flush=True,
             )
         else:
             epochs_no_improve += 1
             print(
                 f"[Pretrain] No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epoch(s).",
                 flush=True,
             )
             if epochs_no_improve >= EARLY_STOP_PATIENCE:
                 print(
                     f"[Pretrain] Early stopping triggered at epoch {epoch}."
                     f"  Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}.",
                     flush=True,
                 )
                 break

     # Reload best checkpoint into policy before returning
     print(f"[Pretrain] Phase 1 complete. Best checkpoint (epoch {best_epoch}): {CKPT_BC_V5}", flush=True)
     policy.load_state_dict(torch.load(CKPT_BC_V5, map_location=device))
     return policy


# ── Phase 2: PPO fine-tuning ──────────────────────────────────────────────────
def run_ppo_finetune(ppo_updates: int = PPO_UPDATES,
                     device: torch.device = None):
     """
     Load the BC checkpoint and run PPO fine-tuning via train_ppo_lstm.train().
     
     v5 enhancements:
     - Warm-starts from CKPT_BC_V5 (best validation checkpoint from Phase 1)
     - Skip RANDOM_ONLY phase (start immediately with self-play)
     - LR warmup over first 20 updates
     - Elevated entropy for first 30 updates to recover from BC entropy collapse
     
     The final checkpoint overwrites CKPT_BC_V5 (ppo_lstm_pretrained_v5.pt).
     """
     if device is None:
         device = torch.device("cpu")

     # Defer import to avoid circular dependency if run standalone
     from ppo import train_ppo_lstm

     print(
         f"[Pretrain] Phase 2 — PPO fine-tune for {ppo_updates} updates"
         f"  warm-start: {CKPT_BC_V5}",
         flush=True,
     )

     # Temporarily override train_ppo_lstm settings for pretrained variant
     original_random_only = train_ppo_lstm.RANDOM_ONLY_UPDATES
     original_ckpt_final = train_ppo_lstm.CKPT_FINAL
     original_ckpt_pool_fmt = train_ppo_lstm.CKPT_POOL_FMT
     
     train_ppo_lstm.RANDOM_ONLY_UPDATES = 0  # Skip random-only phase for pretrained
     train_ppo_lstm.CKPT_FINAL = os.path.join(train_ppo_lstm.CKPT_DIR, "ppo_lstm_v5_policy.pt")
     train_ppo_lstm.CKPT_POOL_FMT = os.path.join(train_ppo_lstm.CKPT_DIR, "ppo_lstm_v5_ckpt_{update}.pt")

     # train() saves to CKPT_DIR/ppo_lstm_v5_policy.pt; we copy to ppo_lstm_pretrained_v5.pt afterwards
     train_ppo_lstm.train(
         n_updates=ppo_updates,
         n_episodes=train_ppo_lstm.N_EPISODES,
         checkpoint=str(CKPT_BC_V5),
     )

     # Restore original settings
     train_ppo_lstm.RANDOM_ONLY_UPDATES = original_random_only
     train_ppo_lstm.CKPT_FINAL = original_ckpt_final
     train_ppo_lstm.CKPT_POOL_FMT = original_ckpt_pool_fmt

     # Copy the final PPO checkpoint over the pretrained checkpoint
     ppo_final = Path(train_ppo_lstm.CKPT_FINAL)
     if ppo_final.exists():
         import shutil
         shutil.copy2(ppo_final, CKPT_BC_V5)
         print(f"[Pretrain] Phase 2 complete. Overwrote: {CKPT_BC_V5}", flush=True)
     else:
         print("[Pretrain] WARNING: PPO final checkpoint not found; pretrained_v5.pt unchanged.", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain FoWPolicyLSTM (BC + PPO fine-tuning)")
    parser.add_argument("--epochs",      type=int,  default=NUM_EPOCHS,
                        help=f"BC training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch_size",  type=int,  default=BC_BATCH,
                        help=f"BC batch size (default: {BC_BATCH})")
    parser.add_argument("--ppo_updates", type=int,  default=PPO_UPDATES,
                        help=f"PPO updates for Phase 2 (default: {PPO_UPDATES})")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"[Pretrain] Device: {device}", flush=True)

    # Phase 1
    run_bc(epochs=args.epochs, batch_size=args.batch_size, device=device)

    # Phase 2 (always runs after Phase 1)
    run_ppo_finetune(ppo_updates=args.ppo_updates, device=device)
