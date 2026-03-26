"""Small policy/value network for AlphaZero-lite training.

Architecture:
  - State encoder MLP: flat state features → state embedding
  - Action encoder MLP: per-action features → action embedding
  - Policy head: concatenate (state, action) embeddings → one logit per
    legal action.  Variable-length; invalid positions are masked to -inf.
  - Value head: state embedding → scalar in [-1, +1]

The model scores the *current legal actions* conditionally on the state,
avoiding a giant fixed action vocabulary.  This is the key design choice
documented in the package docstring.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from catan_ai.models.action_features import ACTION_DIM, STATE_DIM


class PolicyValueNet(nn.Module):
    """Lightweight policy/value network for Catan.

    Args:
        state_dim: Length of flat state feature vector.
        action_dim: Length of per-action feature vector.
        hidden_dim: Width of the shared hidden layers.
        dropout: Dropout probability (0 to disable).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Policy: scores each (state, action) pair via concat → linear → scalar
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        state_feats: torch.Tensor,
        action_feats: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing policy logits and state value.

        Args:
            state_feats:  [B, S]        float — flat state features.
            action_feats: [B, A_max, F] float — per-action features (padded).
            action_mask:  [B, A_max]    bool  — True for valid action slots.

        Returns:
            logits: [B, A_max] float — raw policy logits (masked slots = -inf).
            value:  [B]        float — state value in [-1, +1].
        """
        B, A_max, _ = action_feats.shape

        state_embed = self.state_encoder(state_feats)           # [B, H]
        action_embed = self.action_encoder(action_feats)        # [B, A_max, H]

        # Broadcast state embedding to each action slot
        state_expanded = state_embed.unsqueeze(1).expand(B, A_max, -1)  # [B, A_max, H]
        combined = torch.cat([state_expanded, action_embed], dim=-1)    # [B, A_max, 2H]

        logits = self.policy_head(combined).squeeze(-1)         # [B, A_max]
        logits = logits.masked_fill(~action_mask, float("-inf"))

        value = self.value_head(state_embed).squeeze(-1)        # [B]

        return logits, value
