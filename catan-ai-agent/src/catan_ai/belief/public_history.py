"""Extract publicly observable evidence from a Catanatron game.

This module reads the *live* game state to build a snapshot of everything
the acting player is allowed to know, structured for the belief / sampling
layer.  It intentionally avoids reading opponent resource identities or
dev-card identities — only counts and publicly-revealed plays.

Public information used:
  - Bank resource counts  (state.resource_freqdeck)
  - Acting player's own resource hand  (exact)
  - Acting player's own dev-card hand  (exact)
  - Per-opponent total resource card count  (public in Catan rules)
  - Per-opponent total dev-card count  (public)
  - Per-opponent played dev-card counts  (public — plays are announced)
  - Dev-card deck size  (public — the deck is a visible pile)

Hidden information NOT read here:
  - Opponent resource identities
  - Opponent dev-card identities
  - Dev-card deck order
"""

from __future__ import annotations

from dataclasses import dataclass

from catanatron.game import Game
from catanatron.models.enums import RESOURCES
from catanatron.models.player import Color
from catanatron.state_functions import (
    get_dev_cards_in_hand,
    get_played_dev_cards,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

RESOURCE_TYPES: list[str] = list(RESOURCES)
DEV_CARD_TYPES: list[str] = [
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
]


@dataclass(frozen=True)
class OpponentEvidence:
    """Public information about one opponent."""

    color: str
    total_resource_cards: int
    total_dev_cards: int
    played_dev_cards: dict[str, int]


@dataclass(frozen=True)
class PublicEvidence:
    """Snapshot of all publicly observable evidence at one decision point.

    Enough to reconstruct feasible hidden worlds via belief sampling.
    """

    acting_color: str
    bank_resources: dict[str, int]
    own_resources: dict[str, int]
    own_dev_cards: dict[str, int]
    own_played_dev_cards: dict[str, int]
    opponent_evidence: tuple[OpponentEvidence, ...]
    dev_deck_size: int


def extract_public_evidence(game: Game, acting_color: Color) -> PublicEvidence:
    """Build a PublicEvidence snapshot from the live game.

    Only reads information that the acting player is allowed to know.
    """
    state = game.state
    act_key = player_key(state, acting_color)

    bank = {r: state.resource_freqdeck[i] for i, r in enumerate(RESOURCE_TYPES)}

    own_res = {
        r: state.player_state[f"{act_key}_{r}_IN_HAND"] for r in RESOURCE_TYPES
    }
    own_dev = {
        d: get_dev_cards_in_hand(state, acting_color, d) for d in DEV_CARD_TYPES
    }
    own_played = {
        d: get_played_dev_cards(state, acting_color, d) for d in DEV_CARD_TYPES
    }

    opponents: list[OpponentEvidence] = []
    for color in state.colors:
        if color == acting_color:
            continue
        played = {
            d: get_played_dev_cards(state, color, d) for d in DEV_CARD_TYPES
        }
        opponents.append(
            OpponentEvidence(
                color=color.value,
                total_resource_cards=player_num_resource_cards(state, color),
                total_dev_cards=player_num_dev_cards(state, color),
                played_dev_cards=played,
            )
        )

    return PublicEvidence(
        acting_color=acting_color.value,
        bank_resources=bank,
        own_resources=own_res,
        own_dev_cards=own_dev,
        own_played_dev_cards=own_played,
        opponent_evidence=tuple(opponents),
        dev_deck_size=len(state.development_listdeck),
    )
