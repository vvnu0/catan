"""Belief-aware determinization layer (2-player, practical v1).

This module provides a practical determinization wrapper around the existing
single-world MCTS.  It samples multiple hidden-world assignments consistent
with publicly observable information, runs MCTS on each, and aggregates
root-action statistics across sampled worlds.

Limitations of this version:
  - 2-player only.
  - In 2-player Catan, opponent resource identities are fully determined
    by bank conservation (bank + own hand + opponent hand = 19 per type).
    Resource "sampling" is therefore a no-op; the main value comes from
    dev-card sampling and multi-world variance averaging.
  - Dev-card identity sampling is optional and off by default.
  - No production-history–weighted inference; uses conservation only.
  - This is a practical determinization wrapper, NOT a full exact
    game-theoretic solution (e.g. ISMCTS or CFR).
"""

from catan_ai.belief.determinizer import Determinizer
from catan_ai.belief.public_history import PublicEvidence, extract_public_evidence
from catan_ai.belief.resource_belief import ResourceBelief
from catan_ai.belief.devcard_belief import DevCardBelief

__all__ = [
    "Determinizer",
    "PublicEvidence",
    "extract_public_evidence",
    "ResourceBelief",
    "DevCardBelief",
]
