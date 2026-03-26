"""Flat feature extraction from PublicState and EncodedAction.

All features are deterministic, derived only from publicly observable
information, and designed to be readable/debuggable.  No CNN board
tensors — just a flat vector per state and per action.

STATE_DIM and ACTION_DIM are stable constants exported for use by the
model, dataset, and tests.
"""

from __future__ import annotations

from catan_ai.adapters.public_state import EncodedAction, PublicState

# -----------------------------------------------------------------------
# Resource name ordering (must match PublicState.own_resources keys)
# -----------------------------------------------------------------------
_RESOURCE_ORDER = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")

# -----------------------------------------------------------------------
# Action-type categories for one-hot encoding
# -----------------------------------------------------------------------
_ACTION_TYPES = (
    "ROLL",
    "DISCARD",
    "BUILD_SETTLEMENT",
    "BUILD_CITY",
    "BUILD_ROAD",
    "BUY_DEVELOPMENT_CARD",
    "PLAY_KNIGHT_CARD",
    "PLAY_YEAR_OF_PLENTY",
    "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING",
    "MOVE_ROBBER",
    "MARITIME_TRADE",
    "END_TURN",
    "OTHER",
)
_ACTION_TYPE_TO_IDX = {at: i for i, at in enumerate(_ACTION_TYPES)}

_MANDATORY_TYPES = frozenset({"ROLL", "DISCARD"})
_BUILD_TYPES = frozenset({"BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD"})
_TRADE_TYPES = frozenset({"MARITIME_TRADE", "OFFER_TRADE"})
_DEV_PLAY_TYPES = frozenset({
    "PLAY_KNIGHT_CARD",
    "PLAY_YEAR_OF_PLENTY",
    "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING",
})

# Exported constants — importable by model, dataset, and tests.
STATE_DIM = 53
ACTION_DIM = len(_ACTION_TYPES) + 5  # one-hot + 5 binary flags = 19


# =====================================================================
# State features
# =====================================================================

def _pip_count(number: int) -> int:
    return 6 - abs(7 - number)


def _find_own(ps: PublicState):
    for s in ps.player_summaries:
        if s.color == ps.acting_color:
            return s
    return None


def _find_opp(ps: PublicState):
    for s in ps.player_summaries:
        if s.color != ps.acting_color:
            return s
    return None


def _production_score(ps: PublicState, color_str: str) -> float:
    total = 0.0
    for b in ps.buildings:
        if b.color != color_str:
            continue
        mult = 2.0 if b.building_type == "CITY" else 1.0
        pips = sum(_pip_count(n) for _, n in ps.node_production.get(b.node_id, ()))
        total += pips * mult
    return total


def _port_count(ps: PublicState, color_str: str) -> int:
    port_node_set = {n for nodes in ps.port_nodes.values() for n in nodes}
    return sum(1 for b in ps.buildings if b.color == color_str and b.node_id in port_node_set)


def state_features(ps: PublicState) -> list[float]:
    """Extract a flat float vector of length STATE_DIM from PublicState.

    Features are ordered deterministically and documented inline.
    Normalisation is light (divide by game-scale constants where helpful)
    so the model can learn residual scaling.
    """
    own = _find_own(ps)
    opp = _find_opp(ps)
    if own is None or opp is None:
        return [0.0] * STATE_DIM

    feats: list[float] = []

    # --- VP (4) ---
    feats.append(float(own.visible_victory_points))
    feats.append(float(ps.own_dev_cards.get("VICTORY_POINT", 0)))
    feats.append(float(ps.vps_to_win))
    feats.append(float(opp.visible_victory_points))

    # --- Own resources (6) ---
    for r in _RESOURCE_ORDER:
        feats.append(float(ps.own_resources.get(r, 0)))
    feats.append(float(own.num_resource_cards))

    # --- Opponent resource count (1) ---
    feats.append(float(opp.num_resource_cards))

    # --- Bank resources (6) ---
    for r in _RESOURCE_ORDER:
        feats.append(float(ps.bank_resources.get(r, 0)))
    feats.append(float(ps.bank_dev_cards_remaining))

    # --- Own structures (6) ---
    feats.append(float(own.num_settlements))
    feats.append(float(own.num_cities))
    feats.append(float(own.num_roads))
    feats.append(float(own.settlements_available))
    feats.append(float(own.cities_available))
    feats.append(float(own.roads_available))

    # --- Opponent structures (6) ---
    feats.append(float(opp.num_settlements))
    feats.append(float(opp.num_cities))
    feats.append(float(opp.num_roads))
    feats.append(float(opp.settlements_available))
    feats.append(float(opp.cities_available))
    feats.append(float(opp.roads_available))

    # --- Awards (6) ---
    feats.append(1.0 if own.has_longest_road else 0.0)
    feats.append(1.0 if own.has_largest_army else 0.0)
    feats.append(float(own.longest_road_length))
    feats.append(1.0 if opp.has_longest_road else 0.0)
    feats.append(1.0 if opp.has_largest_army else 0.0)
    feats.append(float(opp.longest_road_length))

    # --- Own dev cards in hand (5) ---
    for d in ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"):
        feats.append(float(ps.own_dev_cards.get(d, 0)))

    # --- Opponent dev info (2) ---
    feats.append(float(opp.num_dev_cards))
    feats.append(float(opp.played_knights))

    # --- Production scores (2) ---
    feats.append(_production_score(ps, own.color))
    feats.append(_production_score(ps, opp.color))

    # --- Port counts (2) ---
    feats.append(float(_port_count(ps, own.color)))
    feats.append(float(_port_count(ps, opp.color)))

    # --- Phase flags (5) ---
    feats.append(1.0 if ps.is_initial_build_phase else 0.0)
    feats.append(1.0 if ps.is_discarding else 0.0)
    feats.append(1.0 if ps.is_moving_robber else 0.0)
    feats.append(1.0 if ps.is_road_building else 0.0)
    feats.append(float(ps.free_roads_available))

    # --- Turn / action count (2) ---
    feats.append(float(ps.turn_number) / 500.0)
    feats.append(float(len(ps.legal_actions)) / 50.0)

    assert len(feats) == STATE_DIM, f"Expected {STATE_DIM}, got {len(feats)}"
    return feats


# =====================================================================
# Action features
# =====================================================================

def action_features(ea: EncodedAction) -> list[float]:
    """Extract a flat float vector of length ACTION_DIM from one EncodedAction.

    Features:
      [0..13]  one-hot action type
      [14]     is_mandatory
      [15]     is_build
      [16]     is_trade
      [17]     is_devcard_play
      [18]     has_value (action carries a parsed value field)
    """
    feats = [0.0] * ACTION_DIM

    # One-hot action type
    idx = _ACTION_TYPE_TO_IDX.get(ea.action_type, _ACTION_TYPE_TO_IDX["OTHER"])
    feats[idx] = 1.0

    # Binary flags
    offset = len(_ACTION_TYPES)
    feats[offset + 0] = 1.0 if ea.action_type in _MANDATORY_TYPES else 0.0
    feats[offset + 1] = 1.0 if ea.action_type in _BUILD_TYPES else 0.0
    feats[offset + 2] = 1.0 if ea.action_type in _TRADE_TYPES else 0.0
    feats[offset + 3] = 1.0 if ea.action_type in _DEV_PLAY_TYPES else 0.0
    feats[offset + 4] = 1.0 if ea.value is not None else 0.0

    assert len(feats) == ACTION_DIM
    return feats
