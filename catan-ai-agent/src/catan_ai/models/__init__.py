"""Feature extraction and policy/value network for AlphaZero-lite training.

This package uses *legal-action scoring* rather than a giant fixed action
vocabulary.  The model takes state features (from PublicState) and per-action
features (from EncodedAction), then produces one logit per legal action and
one scalar value estimate for the state.  This avoids needing a global action
ID mapping and naturally handles the variable-size legal action sets in Catan.
"""

from catan_ai.models.action_features import (
    ACTION_DIM,
    STATE_DIM,
    action_features,
    state_features,
)

__all__ = [
    "STATE_DIM",
    "ACTION_DIM",
    "state_features",
    "action_features",
]
