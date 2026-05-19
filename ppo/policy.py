"""
Simple feedforward policy network for PPO (no memory — baseline).
Input:  128-dim obs vector (64 cells × 2 channels: owner + piece_type)
Output: logits over 4096 actions + scalar value estimate
"""
import torch
import torch.nn as nn

class FoWPolicy(nn.Module):
    def __init__(self, obs_dim: int = 128, act_dim: int = 4096, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def select_action(self, obs: torch.Tensor, mask: torch.Tensor):
        """Sample action with illegal-action masking."""
        logits, value = self(obs)
        logits = logits.masked_fill(~mask, float('-inf'))
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()
