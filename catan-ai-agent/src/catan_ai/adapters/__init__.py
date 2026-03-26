"""Adapter layer: translates raw Catanatron state into masked representations.

Architecture rule: raw Catanatron game/state objects may only be read inside
this package.  All downstream AI logic must consume PublicState (or data
derived from it) — never the raw objects directly.
"""

from catan_ai.adapters.public_state import PublicPlayerSummary, PublicState
from catan_ai.adapters.catanatron_adapter import public_state_from_game
from catan_ai.adapters.action_codec import ActionCodec

__all__ = [
    "PublicState",
    "PublicPlayerSummary",
    "public_state_from_game",
    "ActionCodec",
]
