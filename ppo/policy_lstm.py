"""
FoWPolicyLSTM — Recurrent actor-critic for Fog-of-War Chess.

Architecture:
  obs (128 or legacy 192) → encoder [Linear(obs_dim→512), ReLU, Linear(512→512), ReLU]
                          → LSTM(512→512, num_layers=1)
                          → actor  Linear(512→4096)
                          → critic Linear(512→1)

  Asymmetric privileged critic (disabled by default, used when true board state
  is available during training):
                          → critic_encoder Linear(512+64→512), ReLU
                          → critic_head    Linear(512→1)

forward() accepts both:
  • single-step inference:  obs shape (B, obs_dim)       — used by ppo_lstm_server.py
  • sequence inference:     obs shape (B, T, obs_dim)    — used by train_ppo_lstm.py (BPTT)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

VERSION = "v4"

OBS_DIM        = 128
LEGACY_OBS_DIM = 192
HIDDEN_DIM     = 512
ACT_DIM        = 4096


class FoWPolicyLSTM(nn.Module):
    def __init__(self,
                 obs_dim:    int = OBS_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 act_dim:    int = ACT_DIM):
        super().__init__()
        self.obs_dim    = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim    = act_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Orthogonal init stabilises LSTM training
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.actor  = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        # Asymmetric critic with privileged observations (true board state).
        # During training, critic can see true_board (64 binary values) appended to LSTM output.
        self.critic_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.ReLU(),
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int = 1, device=None):
        """Return (h_0, c_0) zero tensors of shape (1, batch_size, hidden_dim)."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(self, obs, legal_mask, h=None, c=None):
        """
        Parameters
        ----------
        obs        : Tensor, shape (B, obs_dim) or (B, T, obs_dim)
        legal_mask : Tensor, shape (B, 4096) or (B, T, 4096)  — bool
        h          : Tensor, shape (1, B, hidden_dim)  or None  (zero-init)
        c          : Tensor, shape (1, B, hidden_dim)  or None  (zero-init)

        Returns
        -------
        dist   : Categorical distribution over legal actions
        value  : Tensor  (B,) or (B, T)
        h_new  : Tensor  (1, B, hidden_dim)
        c_new  : Tensor  (1, B, hidden_dim)
        """
        squeeze = obs.dim() == 2
        if squeeze:
            obs        = obs.unsqueeze(1)
            legal_mask = legal_mask.unsqueeze(1)

        B, T, _ = obs.shape

        if h is None or c is None:
            h, c = self.init_hidden(B, obs.device)

        # Encode every time step: (B, T, obs_dim) → (B, T, hidden_dim)
        enc = self.encoder(obs.view(B * T, self.obs_dim)).view(B, T, self.hidden_dim)

        lstm_out, (h_new, c_new) = self.lstm(enc, (h, c))

        logits = self.actor(lstm_out)   # (B, T, 4096)
        values = self.critic(lstm_out).squeeze(-1)  # (B, T)

        # Replace NaN logits (can occur if LSTM output is NaN after weight corruption)
        logits = torch.nan_to_num(logits, nan=float('-inf'))

        # Mask illegal actions; where ALL actions are masked allow uniform distribution
        # to prevent NaN in Categorical.
        illegal    = ~legal_mask
        all_masked = illegal.all(dim=-1, keepdim=True)
        illegal    = illegal & ~all_masked
        logits     = logits.masked_fill(illegal, float('-inf'))
        logits     = logits.masked_fill(all_masked, 0.0)

        if squeeze:
            logits = logits.squeeze(1)
            values = values.squeeze(1)

        dist = Categorical(logits=logits)
        return dist, values, h_new, c_new

    def select_action(self, obs, legal_mask, h=None, c=None):
        """
        Sample an action and return (action, log_prob, value, h_new, c_new).

        obs / legal_mask should be (B, obs_dim) / (B, 4096).
        """
        dist, value, h_new, c_new = self.forward(obs, legal_mask, h, c)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, h_new, c_new

    def select_action_with_privileged(self, obs, true_board, legal_mask, h=None, c=None):
        """
        Sample an action using the standard actor, but compute value using the
        privileged critic that sees true_board.

        Parameters
        ----------
        obs        : Tensor, shape (B, obs_dim)
        true_board : Tensor, shape (B, 64) — binary mask of true board state, or None
        legal_mask : Tensor, shape (B, 4096)
        h, c       : LSTM hidden states or None

        Returns
        -------
        action, log_prob, value, h_new, c_new
        """
        if true_board is None:
            return self.select_action(obs, legal_mask, h, c)

        squeeze = obs.dim() == 2
        if squeeze:
            obs        = obs.unsqueeze(1)
            legal_mask = legal_mask.unsqueeze(1)
            true_board = true_board.unsqueeze(1)

        B, T, _ = obs.shape

        if h is None or c is None:
            h, c = self.init_hidden(B, obs.device)

        enc = self.encoder(obs.view(B * T, self.obs_dim)).view(B, T, self.hidden_dim)
        lstm_out, (h_new, c_new) = self.lstm(enc, (h, c))

        logits = self.actor(lstm_out)

        # Privileged critic: concatenate true board state to LSTM output
        critic_input  = torch.cat([lstm_out, true_board.expand(B, T, 64)], dim=-1)
        values = self.critic_head(self.critic_encoder(critic_input)).squeeze(-1)

        illegal    = ~legal_mask
        all_masked = illegal.all(dim=-1, keepdim=True)
        illegal    = illegal & ~all_masked
        logits     = logits.masked_fill(illegal, float('-inf'))
        logits     = logits.masked_fill(all_masked, 0.0)

        if squeeze:
            logits = logits.squeeze(1)
            values = values.squeeze(1)

        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, values, h_new, c_new
