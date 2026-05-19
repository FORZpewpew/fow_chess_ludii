"""
Online PPO self-play training for FoW Chess.

Improvements over the baseline:
  1. Side-randomisation: each episode the PPO agent is randomly assigned P1 or P2.
  2. Opponent pool: starts with a random-move baseline (None); the current policy
     checkpoint is added to the pool every POOL_ADD_EVERY updates.  Each episode
     the opponent is sampled uniformly from the pool.

Usage:
  cd /Users/forzpewpew/Downloads/ludii
  python3 ppo/train_ppo.py                    # full run (500 updates × 20 episodes)
  python3 ppo/train_ppo.py --updates 2 --episodes 2  # quick smoke-test
"""

import argparse
import os
import random

import numpy as np
import torch

from ppo.fow_env import FoWChessEnv
from ppo.policy  import FoWPolicy

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_EPISODES      = 20        # games per PPO update
N_UPDATES       = 500       # total PPO updates
PPO_EPOCHS      = 4         # gradient steps per update
CLIP_EPS        = 0.2
GAMMA           = 0.99
LR              = 3e-4
SAVE_EVERY      = 50        # save checkpoint + add to pool every N updates
CKPT_DIR        = "/Users/forzpewpew/Downloads/ludii/ppo/checkpoints"
CKPT_FINAL      = os.path.join(CKPT_DIR, "ppo_selfplay_policy.pt")
CKPT_POOL_FMT   = os.path.join(CKPT_DIR, "ppo_selfplay_ckpt_{update}.pt")

OBS_DIM  = 128  # 64 cells × 2 channels (owner + piece_type)
ACT_DIM  = 4096

# ── Policy cache (avoid reloading the same checkpoint every step) ─────────────
_policy_cache: dict[str, FoWPolicy] = {}

def get_cached_policy(path: str, device: torch.device) -> FoWPolicy:
    """Load a FoWPolicy from *path* and cache it; reuse on subsequent calls."""
    if path not in _policy_cache:
        p = FoWPolicy(OBS_DIM, ACT_DIM).to(device)
        p.load_state_dict(torch.load(path, map_location=device))
        p.eval()
        _policy_cache[path] = p
    return _policy_cache[path]


# ── Opponent action sampler ───────────────────────────────────────────────────
def make_opponent_fn(opponent_entry, device: torch.device):
    """
    Return a callable(obs: np.ndarray, legal_mask: np.ndarray) -> int
    suitable for passing to FoWChessEnv as opponent_fn.

    opponent_entry is either:
      None  — random baseline (uniform over legal moves)
      str   — filesystem path to a saved FoWPolicy checkpoint
    """
    if opponent_entry is None:
        # Random baseline
        def _random_fn(obs: np.ndarray, legal_mask: np.ndarray) -> int:
            legal_indices = np.where(legal_mask)[0]
            if len(legal_indices) == 0:
                return 0
            return int(legal_indices[random.randrange(len(legal_indices))])
        return _random_fn
    else:
        # Checkpoint-based opponent
        def _policy_fn(obs: np.ndarray, legal_mask: np.ndarray) -> int:
            policy = get_cached_policy(opponent_entry, device)
            obs_t  = torch.tensor(obs,        dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(legal_mask, dtype=torch.bool,    device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = policy.select_action(obs_t, mask_t)
            return int(action)
        return _policy_fn


# ── Return calculation ────────────────────────────────────────────────────────
def compute_returns(rewards, dones, gamma: float) -> list:
    returns = []
    G = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0.0
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ── Episode collection ────────────────────────────────────────────────────────
def collect_episode(ppo_policy: FoWPolicy,
                    opponent_pool: list,
                    device: torch.device):
    """
    Play one full game.

    Returns:
        ep_obs, ep_masks, ep_acts, ep_lps, ep_vals, ep_rews, ep_dones, ppo_won
    """
    ppo_side = random.choice([1, 2])
    opponent_entry = random.choice(opponent_pool)

    opponent_fn = make_opponent_fn(opponent_entry, device)
    env = FoWChessEnv(player=ppo_side, opponent_fn=opponent_fn)

    obs, mask = env.reset()

    ep_obs, ep_masks, ep_acts, ep_lps, ep_vals, ep_rews, ep_dones = \
        [], [], [], [], [], [], []

    done = False
    while not done:
        obs_t  = torch.tensor(obs,  dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool,    device=device).unsqueeze(0)

        with torch.no_grad():
            act, lp, val = ppo_policy.select_action(obs_t, mask_t)

        next_obs, reward, done, _ = env.step(act)
        next_mask = env._legal_mask() if not done else mask

        ep_obs.append(obs);   ep_masks.append(mask)
        ep_acts.append(act);  ep_lps.append(lp)
        ep_vals.append(val);  ep_rews.append(reward)
        ep_dones.append(done)

        obs, mask = next_obs, next_mask

    ppo_won = ep_rews[-1] > 0
    return ep_obs, ep_masks, ep_acts, ep_lps, ep_vals, ep_rews, ep_dones, ppo_won


# ── Main training loop ────────────────────────────────────────────────────────
def train(n_updates: int = N_UPDATES, n_episodes: int = N_EPISODES,
          checkpoint: str = None):
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[PPO-SP] Device: {device}")

    policy    = FoWPolicy(OBS_DIM, ACT_DIM).to(device)
    if checkpoint and os.path.exists(checkpoint):
        policy.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"[PPO-SP] Warm-started from: {checkpoint}", flush=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    # Opponent pool: None = random baseline
    opponent_pool: list = [None]

    for update in range(1, n_updates + 1):
        # ── Collect trajectories ──────────────────────────────────────────────
        all_obs, all_masks  = [], []
        all_actions         = []
        all_log_probs       = []
        all_values          = []
        all_returns         = []
        wins = 0

        for _ in range(n_episodes):
            ep_obs, ep_masks, ep_acts, ep_lps, ep_vals, ep_rews, ep_dones, won = \
                collect_episode(policy, opponent_pool, device)

            returns = compute_returns(ep_rews, ep_dones, GAMMA)

            all_obs.extend(ep_obs);           all_masks.extend(ep_masks)
            all_actions.extend(ep_acts);      all_log_probs.extend(ep_lps)
            all_values.extend(ep_vals);       all_returns.extend(returns)
            if won:
                wins += 1

        # ── PPO update ────────────────────────────────────────────────────────
        obs_t      = torch.tensor(np.array(all_obs),   dtype=torch.float32, device=device)
        mask_t     = torch.tensor(np.array(all_masks), dtype=torch.bool,    device=device)
        acts_t     = torch.tensor(all_actions,          dtype=torch.long,    device=device)
        old_lps_t  = torch.tensor(all_log_probs,        dtype=torch.float32, device=device)
        returns_t  = torch.tensor(all_returns,          dtype=torch.float32, device=device)
        old_vals_t = torch.tensor(all_values,           dtype=torch.float32, device=device)

        advantages = returns_t - old_vals_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss = actor_l = critic_l = None
        for _ in range(PPO_EPOCHS):
            logits, values = policy(obs_t)
            logits = logits.masked_fill(~mask_t, float('-inf'))
            dist   = torch.distributions.Categorical(logits=logits)
            new_lps = dist.log_prob(acts_t)
            entropy = dist.entropy().mean()

            ratio   = torch.exp(new_lps - old_lps_t)
            surr1   = ratio * advantages
            surr2   = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            actor_l  = -torch.min(surr1, surr2).mean()
            critic_l = (returns_t - values).pow(2).mean()
            loss     = actor_l + 0.5 * critic_l - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        win_rate   = wins / n_episodes
        pool_size  = len(opponent_pool)
        print(
            f"[PPO-SP] Update {update:4d}/{n_updates}"
            f"  ppo_side=mixed"
            f"  opp=pool_size/{pool_size}"
            f"  win_rate={win_rate:.2f}"
            f"  loss={loss.item():.4f}"
        )

        # ── Checkpoint + pool update ──────────────────────────────────────────
        if update % SAVE_EVERY == 0:
            # Save pool checkpoint
            pool_ckpt = CKPT_POOL_FMT.format(update=update)
            torch.save(policy.state_dict(), pool_ckpt)
            opponent_pool.append(pool_ckpt)
            # Also save the "latest" checkpoint
            torch.save(policy.state_dict(), CKPT_FINAL)
            print(f"[PPO-SP] Saved checkpoint: {pool_ckpt}  (pool size now {len(opponent_pool)})")

    # Final save
    torch.save(policy.state_dict(), CKPT_FINAL)
    print(f"[PPO-SP] Training complete. Final checkpoint: {CKPT_FINAL}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO self-play training for FoW Chess")
    parser.add_argument("--updates",  type=int, default=N_UPDATES,
                        help=f"Number of PPO updates (default: {N_UPDATES})")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help=f"Episodes per update (default: {N_EPISODES})")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint to warm-start from")
    args = parser.parse_args()
    train(n_updates=args.updates, n_episodes=args.episodes, checkpoint=args.checkpoint)
