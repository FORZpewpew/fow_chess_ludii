"""
train_ppo_lstm.py — PPO with truncated BPTT for FoW Chess (LSTM policy).

Key design choices
──────────────────
• Episodes are collected in full (up to MAX_PLIES plies per side) and stored
  as flat lists of (obs, action, logprob, value, reward, done, h_t, c_t).
• After collection, each episode is chunked into segments of length T=64.
  The stored (h_start, c_start) at the beginning of each segment is used as
  the initial hidden state when running the forward pass through that segment.
  Gradients are NOT propagated beyond segment boundaries (.detach()) — this is
  the "truncated" part of truncated BPTT.
• GAE(λ=0.95) advantages are computed within each segment.
• PPO clip ε=0.2, entropy bonus 0.01, value loss coeff 0.5, grad clip 0.5.
• Opponent pool: random baseline for the first 100 updates; after that 50%
  random / 50% self-play previous checkpoint.

Usage:
  cd /Users/forzpewpew/Downloads/ludii
  python3 ppo/train_ppo_lstm.py
  python3 ppo/train_ppo_lstm.py --updates 2 --episodes 2   # smoke-test
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.distributions import Categorical

from ppo.fow_env     import FoWChessEnv
from ppo.policy_lstm import FoWPolicyLSTM

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_EPISODES    = 16
N_UPDATES     = 400
PPO_EPOCHS    = 4
CLIP_EPS      = 0.2
GAMMA         = 0.99
LAMBDA        = 0.95        # GAE lambda
LR            = 1e-4
TBPTT_LEN     = 64          # truncated-BPTT segment length
ENTROPY_COEFF = 0.01
VALUE_COEFF   = 0.5
MAX_GRAD_NORM = 0.5
SAVE_EVERY    = 25
HIDDEN_DIM    = 512

CKPT_DIR      = "checkpoints"
CKPT_FINAL    = os.path.join(CKPT_DIR, "ppo_lstm_v4_policy.pt")
CKPT_POOL_FMT = os.path.join(CKPT_DIR, "ppo_lstm_v4_ckpt_{update}.pt")

# Use only random opponent for this many updates before introducing self-play
RANDOM_ONLY_UPDATES = 100

OBS_DIM  = 128
ACT_DIM  = 4096

# Reward shaping constants (egocentric owner encoding from fow_env.py)
_OWN_VAL   = 2.0 / 3.0
_ENEMY_VAL = 1.0
_OWNER_TOL = 0.15
_PIECE_VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}  # pawn, knight, bishop, rook, queen

_policy_cache: dict = {}


def get_cached_policy(path: str, device: torch.device) -> FoWPolicyLSTM:
    """Load a FoWPolicyLSTM from *path* and cache it; reuse on subsequent calls."""
    if path not in _policy_cache:
        p = FoWPolicyLSTM().to(device)
        p.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        p.eval()
        _policy_cache[path] = p
    return _policy_cache[path]


def make_opponent_fn(opponent_entry, device: torch.device):
    """
    Return callable(obs: np.ndarray, legal_mask: np.ndarray) -> int.

    opponent_entry:
      None — random baseline (uniform over legal moves)
      str  — filesystem path to a saved FoWPolicyLSTM checkpoint
    """
    if opponent_entry is None:
        def _random_fn(obs: np.ndarray, legal_mask: np.ndarray) -> int:
            legal_indices = np.where(legal_mask)[0]
            if len(legal_indices) == 0:
                return 0
            return int(legal_indices[random.randrange(len(legal_indices))])
        return _random_fn

    # Stateful LSTM opponent — hidden state is carried across calls within
    # a single game via a closure-held mutable list.
    _state = [None, None]  # [h, c]

    def _policy_fn(obs: np.ndarray, legal_mask: np.ndarray) -> int:
        policy = get_cached_policy(opponent_entry, device)
        obs_t  = torch.tensor(obs,        dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(legal_mask, dtype=torch.bool,    device=device).unsqueeze(0)
        h, c   = _state
        with torch.no_grad():
            # Single forward pass: dist and h_new/c_new come from the same step.
            dist, _, h_new, c_new = policy(obs_t, mask_t, h, c)
            action = dist.sample().item()
        _state[0] = h_new
        _state[1] = c_new
        return int(action)

    # Expose reset so collect_episode can call it between games
    _policy_fn._reset = lambda: _state.__setitem__(slice(None), [None, None])
    return _policy_fn


def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """
    Compute GAE advantages for a segment.

    rewards, values, dones: lists or 1-D arrays of length T
    values must contain one extra entry (bootstrap): length T+1

    Returns advantages (list, length T) and returns (list, length T).
    """
    T        = len(rewards)
    advs     = [0.0] * T
    last_gae = 0.0

    for t in reversed(range(T)):
        non_terminal = 1.0 - float(dones[t])
        delta        = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
        last_gae     = delta + gamma * lam * non_terminal * last_gae
        advs[t]      = last_gae

    returns = [advs[t] + values[t] for t in range(T)]

    # Guard against NaN/Inf that can arise from degenerate episodes
    advs_t    = torch.nan_to_num(torch.tensor(advs,    dtype=torch.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    returns_t = torch.nan_to_num(torch.tensor(returns, dtype=torch.float32), nan=0.0)
    return advs_t.tolist(), returns_t.tolist()


def _shape_reward(reward: float, steps: list, next_obs: np.ndarray) -> float:
    """Apply FoW reward shaping on top of the sparse terminal reward.

    Fog reveal bonus: +0.02 when a previously hidden enemy piece becomes visible.
    Capture bonus: +0.05 × piece_value when an enemy piece disappears.
    King exposure penalty: -0.03 when own king is adjacent to a visible enemy piece.
    """
    if not steps:
        return reward

    prev_obs = steps[-1][0]

    # Fog reveal bonus
    for sq in range(64):
        if (prev_obs[sq] < _OWNER_TOL
                and abs(next_obs[sq] - _ENEMY_VAL) < _OWNER_TOL
                and next_obs[64 + sq] > 0.0):
            reward += 0.02

    # Piece capture bonus
    for sq in range(64):
        prev_piece = int(round(prev_obs[64 + sq] * 6))
        if (abs(prev_obs[sq] - _ENEMY_VAL) < _OWNER_TOL
                and prev_piece > 0
                and abs(next_obs[sq] - 1.0 / 3.0) < _OWNER_TOL):
            reward += 0.05 * _PIECE_VALUES.get(prev_piece, 0)

    # King exposure penalty
    for sq in range(64):
        piece = int(round(next_obs[64 + sq] * 6))
        if piece == 6 and abs(next_obs[sq] - _OWN_VAL) < _OWNER_TOL:
            row, col = sq // 8, sq % 8
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    ar, ac = row + dr, col + dc
                    if 0 <= ar < 8 and 0 <= ac < 8:
                        adj = ar * 8 + ac
                        if (abs(next_obs[adj] - _ENEMY_VAL) < _OWNER_TOL
                                and next_obs[64 + adj] > 0.0):
                            reward -= 0.03

    return reward


def collect_episode(policy: FoWPolicyLSTM,
                    opponent_pool: list,
                    device: torch.device):
    """
    Play one full game.

    Returns a list of step tuples:
      (obs, mask, action, log_prob, value, reward, done, h_t, c_t)
    where (h_t, c_t) is the hidden state BEFORE the step was taken.

    Also returns ppo_won (bool).
    """
    ppo_side       = random.choice([1, 2])
    opponent_entry = random.choice(opponent_pool)
    opponent_fn    = make_opponent_fn(opponent_entry, device)

    env  = FoWChessEnv(player=ppo_side, opponent_fn=opponent_fn)
    obs, mask = env.reset()

    h, c = policy.init_hidden(batch_size=1, device=device)

    steps = []
    done  = False

    while not done:
        obs_t  = torch.tensor(obs,  dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool,    device=device).unsqueeze(0)

        h_snap = h.detach().clone()
        c_snap = c.detach().clone()

        with torch.no_grad():
            action, log_prob, value, h, c = policy.select_action(obs_t, mask_t, h, c)

        act_i = int(action.item())
        lp_f  = float(log_prob.item())
        val_f = float(value.item())

        next_obs, reward, done, _ = env.step(act_i)
        next_mask = env._legal_mask() if not done else mask

        try:
            reward = _shape_reward(reward, steps, next_obs)
        except Exception:
            pass

        steps.append((obs, mask, act_i, lp_f, val_f, reward, done, h_snap, c_snap))
        obs, mask = next_obs, next_mask

    ppo_won = steps[-1][5] > 0
    return steps, ppo_won


def process_segments(episodes_steps: list,
                     policy: FoWPolicyLSTM,
                     optimizer: torch.optim.Optimizer,
                     device: torch.device,
                     n_epochs: int):
    """
    Chunk all collected episodes into segments of length TBPTT_LEN, run PPO
    updates over them for n_epochs passes.

    Returns (mean_policy_loss, mean_value_loss) over the final epoch.
    """
    all_segments = []
    for steps in episodes_steps:
        for start in range(0, len(steps), TBPTT_LEN):
            seg = steps[start: start + TBPTT_LEN]
            if seg:
                all_segments.append(seg)

    if not all_segments:
        return 0.0, 0.0

    final_p_loss = final_v_loss = 0.0

    for epoch in range(n_epochs):
        random.shuffle(all_segments)
        epoch_p_loss = epoch_v_loss = 0.0

        for seg in all_segments:
            T = len(seg)

            obs_list     = [s[0] for s in seg]
            mask_list    = [s[1] for s in seg]
            act_list     = [s[2] for s in seg]
            old_lp_list  = [s[3] for s in seg]
            old_val_list = [s[4] for s in seg]
            rew_list     = [s[5] for s in seg]
            done_list    = [s[6] for s in seg]
            h_init = seg[0][7].to(device)
            c_init = seg[0][8].to(device)

            obs_arr = np.array(obs_list)
            if np.isnan(obs_arr).any():
                continue

            obs_t    = torch.tensor(obs_arr,              dtype=torch.float32, device=device)
            mask_t   = torch.tensor(np.array(mask_list), dtype=torch.bool,    device=device)
            acts_t   = torch.tensor(act_list,            dtype=torch.long,    device=device)
            old_lp_t = torch.tensor(old_lp_list,         dtype=torch.float32, device=device)

            # Forward through segment with detached initial hidden state.
            # obs_t shape (T, 128) → add batch dim → (1, T, 128)
            obs_b  = obs_t.unsqueeze(0)
            mask_b = mask_t.unsqueeze(0)

            dist, values_b, _, _ = policy(obs_b, mask_b,
                                          h_init.detach(),
                                          c_init.detach())
            values = values_b.squeeze(0)  # (T,)

            # dist.logits shape is (1, T, 4096) for T>1; squeeze batch dim
            logits_t = dist.logits.squeeze(0)
            dist_t   = Categorical(logits=logits_t)

            new_lp  = dist_t.log_prob(acts_t)
            entropy = dist_t.entropy().mean()

            # Bootstrap: 0 if episode ended, else last old value estimate
            boot_val  = 0.0 if done_list[-1] else float(old_val_list[-1])
            boot_vals = old_val_list + [boot_val]

            advs, returns = compute_gae(rew_list, boot_vals, done_list, GAMMA, LAMBDA)
            advs_t    = torch.tensor(advs,    dtype=torch.float32, device=device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            # Normalise advantages within segment.
            # Use correction=0 (population std) to avoid NaN on single-element segments.
            if advs_t.numel() > 1:
                advs_t = (advs_t - advs_t.mean()) / (advs_t.std(correction=0) + 1e-8)
            else:
                advs_t = advs_t - advs_t.mean()

            ratio  = torch.exp(new_lp - old_lp_t)
            surr1  = ratio * advs_t
            surr2  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advs_t
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = (returns_t - values).pow(2).mean()
            loss   = p_loss + VALUE_COEFF * v_loss - ENTROPY_COEFF * entropy

            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                optimizer.zero_grad()
                continue
            optimizer.step()

            epoch_p_loss += p_loss.item()
            epoch_v_loss += v_loss.item()

        n = len(all_segments)
        final_p_loss = epoch_p_loss / n
        final_v_loss = epoch_v_loss / n

    return final_p_loss, final_v_loss


def train(n_updates: int = N_UPDATES,
          n_episodes: int = N_EPISODES,
          checkpoint: str = None):
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device("cpu")
    print(f"[PPO-LSTM] Device: {device}", flush=True)

    policy = FoWPolicyLSTM().to(device)
    if checkpoint and os.path.exists(checkpoint):
        policy.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        print(f"[PPO-LSTM] Warm-started from: {checkpoint}", flush=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    opponent_pool: list = [None]

    for update in range(1, n_updates + 1):
        active_pool = [None] if update <= RANDOM_ONLY_UPDATES else opponent_pool

        all_episodes_steps = []
        wins = 0
        total_steps = 0

        for _ in range(n_episodes):
            policy.eval()
            steps, won = collect_episode(policy, active_pool, device)
            all_episodes_steps.append(steps)
            if won:
                wins += 1
            total_steps += len(steps)

        mean_ep_len = total_steps / n_episodes

        policy.train()
        p_loss, v_loss = process_segments(
            all_episodes_steps, policy, optimizer, device, PPO_EPOCHS
        )

        win_rate  = wins / n_episodes
        pool_size = len(opponent_pool)
        print(
            f"[PPO-LSTM] Update {update:4d}/{n_updates}"
            f"  win_rate={win_rate:.2f}"
            f"  mean_ep_len={mean_ep_len:.1f}"
            f"  pool_size={pool_size}"
            f"  p_loss={p_loss:.4f}"
            f"  v_loss={v_loss:.4f}",
            flush=True,
        )

        if update % SAVE_EVERY == 0:
            pool_ckpt = CKPT_POOL_FMT.format(update=update)
            torch.save(policy.state_dict(), pool_ckpt)
            opponent_pool.append(pool_ckpt)
            torch.save(policy.state_dict(), CKPT_FINAL)
            print(
                f"[PPO-LSTM] Saved checkpoint: {pool_ckpt}"
                f"  (pool size now {len(opponent_pool)})",
                flush=True,
            )

    torch.save(policy.state_dict(), CKPT_FINAL)
    print(f"[PPO-LSTM] Training complete. Final checkpoint: {CKPT_FINAL}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO-LSTM self-play training for FoW Chess")
    parser.add_argument("--updates",    type=int, default=N_UPDATES,
                        help=f"Number of PPO updates (default: {N_UPDATES})")
    parser.add_argument("--episodes",   type=int, default=N_EPISODES,
                        help=f"Episodes per update (default: {N_EPISODES})")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint for warm-start")
    args = parser.parse_args()
    train(n_updates=args.updates, n_episodes=args.episodes, checkpoint=args.checkpoint)
