"""
belief_probe.py

BeliefProbeHead: small MLP that maps LSTM hidden state → per-square piece-type logits.

Architecture:
  h (512) → Linear(512→256) → ReLU → Linear(256→64*7) → view(64, 7)

Usage:
  probe = BeliefProbeHead()
  logits = probe(h)   # h: (B, 512)  →  logits: (B, 64, 7)
  preds  = logits.argmax(-1)  # (B, 64)
"""

import torch
import torch.nn as nn

NUM_PIECE_TYPES = 7   # 0=empty, 1=pawn, 2=rook, 3=bishop, 4=knight, 5=queen, 6=king
HIDDEN_DIM      = 512
NUM_SQUARES     = 64


class BeliefProbeHead(nn.Module):
    """Small MLP decoder: LSTM hidden state → belief over hidden squares."""

    def __init__(self,
                 hidden_dim:     int = HIDDEN_DIM,
                 num_squares:    int = NUM_SQUARES,
                 num_piece_types: int = NUM_PIECE_TYPES):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.num_squares     = num_squares
        self.num_piece_types = num_piece_types

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_squares * num_piece_types),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (B, hidden_dim)   — LSTM hidden state (h_t, NOT cell state)

        Returns
        -------
        logits : (B, num_squares, num_piece_types)
        """
        out = self.decoder(h)                           # (B, 64*7)
        return out.view(-1, self.num_squares, self.num_piece_types)
