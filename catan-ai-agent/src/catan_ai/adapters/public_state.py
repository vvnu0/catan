"""Dataclasses that represent *only* publicly observable game information.

Every field here is something the acting player is allowed to know in a
real game of Catan.  Nothing about opponents' hidden resource identities,
unplayed dev-card identities, or the dev-card deck order is included.

These structures are plain Python dataclasses composed of primitives, tuples,
and other dataclasses — intentionally kept serialisable (JSON, pickle, etc.)
so they can be logged and used as training inputs later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Per-player public summary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PublicPlayerSummary:
    """What every player at the table can observe about a single player."""

    color: str

    # Public VP: settlements + cities + longest-road + largest-army.
    # Does NOT include hidden VP dev cards.
    visible_victory_points: int

    # Piece counts on the board (visible to everyone).
    num_settlements: int
    num_cities: int
    num_roads: int

    # How many resource cards the player holds (count only — not identities).
    num_resource_cards: int
    # How many dev cards the player holds (count only — not identities).
    num_dev_cards: int

    # Publicly played dev-card counts (knights visible because of army race;
    # other plays are announced when used).
    played_knights: int
    played_year_of_plenty: int
    played_monopoly: int
    played_road_building: int

    # Award flags
    has_longest_road: bool
    has_largest_army: bool
    longest_road_length: int

    # Remaining unplayed pieces (derivable from board, but convenient).
    roads_available: int
    settlements_available: int
    cities_available: int


# ---------------------------------------------------------------------------
# Board-level structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TileSummary:
    """Public info for a single land tile."""
    coordinate: tuple
    resource: str | None
    number: int | None


@dataclass(frozen=True)
class BuildingSummary:
    """A building visible on the board."""
    node_id: int
    color: str
    building_type: str  # "SETTLEMENT" or "CITY"


@dataclass(frozen=True)
class RoadSummary:
    """A road visible on the board."""
    edge: tuple[int, int]
    color: str


# ---------------------------------------------------------------------------
# Encoded action (stable, serialisable wrapper around raw actions)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EncodedAction:
    """Stable, serialisable representation of a Catanatron Action.

    Stores only primitives so it can be hashed, sorted, logged, and used as a
    policy-target key without importing Catanatron downstream.
    """
    color: str
    action_type: str
    # value is normalised to a JSON-friendly primitive or tuple of primitives.
    value: str | None


# ---------------------------------------------------------------------------
# Top-level masked game snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PublicState:
    """Everything the acting player is allowed to know right now.

    Intentionally omits:
      - exact resource identities for opponents
      - exact dev-card identities for opponents
      - dev-card deck order / composition
      - actual (hidden) victory points for opponents
      - raw Catanatron State / Game objects
    """

    # ---- Identity / turn ----
    acting_color: str
    turn_number: int
    vps_to_win: int

    # ---- Board layout (static after setup) ----
    tiles: tuple[TileSummary, ...]
    robber_coordinate: tuple

    # ---- Structures on the board (public) ----
    buildings: tuple[BuildingSummary, ...]
    roads: tuple[RoadSummary, ...]

    # ---- Per-player summaries (public info only) ----
    player_summaries: tuple[PublicPlayerSummary, ...]

    # ---- Bank (public knowledge in real Catan) ----
    bank_resources: dict[str, int]
    bank_dev_cards_remaining: int

    # ---- Acting player's own private hand (they know their own cards) ----
    own_resources: dict[str, int]
    own_dev_cards: dict[str, int]

    # ---- Legal actions as stable encoded keys ----
    legal_actions: tuple[EncodedAction, ...]

    # ---- Phase flags ----
    is_initial_build_phase: bool
    is_discarding: bool
    is_moving_robber: bool
    is_road_building: bool
    free_roads_available: int

    # ---- Board topology (public, static after setup) ----
    # Maps node_id → producing adjacent tiles as (resource, dice_number) pairs.
    # Desert tiles (resource=None) are excluded.
    node_production: dict[int, tuple[tuple[str, int], ...]]
    # Maps port resource (str for 2:1, None for 3:1) → node_ids on that port.
    port_nodes: dict[str | None, tuple[int, ...]]
