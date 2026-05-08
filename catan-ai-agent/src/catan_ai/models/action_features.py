"""Flat feature extraction from PublicState and EncodedAction.

All features are deterministic, derived only from publicly observable
information, and designed to be readable/debuggable.  No CNN board
tensors — just a flat vector per state and per action.

STATE_DIM and ACTION_DIM are stable constants exported for use by the
model, dataset, and tests.
"""

from __future__ import annotations

import re

from catan_ai.adapters.public_state import EncodedAction, PublicState

# -----------------------------------------------------------------------
# Resource name ordering (must match PublicState.own_resources keys)
# -----------------------------------------------------------------------
_RESOURCE_ORDER = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
_RESOURCE_TO_IDX = {r: i for i, r in enumerate(_RESOURCE_ORDER)}

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

# -----------------------------------------------------------------------
# Feature dimensions
# -----------------------------------------------------------------------
# Old layout (19):  14 one-hot + 5 binary flags
# New spatial block (11):
#   [19]     node_production_score  — pip-weighted production at target node
#   [20]     node_resource_diversity — number of distinct resources at node
#   [21]     node_has_port          — target node is on a port
#   [22..26] trade_give_resource    — 5-dim one-hot (which resource given)
#   [27..31] trade_get_resource     — 5-dim one-hot (which resource received)  [NOTE: indices shifted]
#   [32]     robber_target_pips     — pip value of robber destination hex
# Total: 14 + 5 + 14 = 33

_BASE_DIM = len(_ACTION_TYPES) + 5         # 19: one-hot + binary flags
_SPATIAL_DIM = 3 + 5 + 5 + 1              # 14: node(3) + give(5) + get(5) + robber(1)

# Exported constants — importable by model, dataset, and tests.
STATE_DIM = 53
ACTION_DIM = _BASE_DIM + _SPATIAL_DIM      # 19 + 14 = 33


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
# Action value parsing helpers
# =====================================================================
# EncodedAction.value is a string produced by ActionCodec._normalise_value.
# Examples:
#   BUILD_SETTLEMENT  → "42"                        (node_id)
#   BUILD_CITY        → "42"                        (node_id)
#   BUILD_ROAD        → "(3,7)"                     (edge tuple)
#   MARITIME_TRADE    → "(None,WOOD,ORE,None)"      (port_res, give, get, ...)
#   MOVE_ROBBER       → "((0,0,0),RED,WOOD)"        (hex coord, victim, res)
#   PLAY_KNIGHT_CARD  → "((0,0,0),RED,WOOD)"        (same as robber)
# We use safe parsing — any failure returns None/defaults.

def _try_parse_node_id(value: str | None) -> int | None:
    """Try to parse an integer node_id from the value string."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _try_parse_edge(value: str | None) -> tuple[int, int] | None:
    """Try to parse a (node_a, node_b) edge from '(3,7)' format."""
    if value is None:
        return None
    m = re.match(r"^\((\d+),(\d+)\)$", value.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _try_parse_trade_resources(value: str | None) -> tuple[str | None, str | None]:
    """Parse give/get resource names from a maritime trade value string.

    Catanatron maritime trade values are encoded as tuples like:
      (None,WOOD,ORE,None) — giving WOOD, receiving ORE
    The exact format varies, but the resource names appear in the string.
    We search for known resource names in order.
    """
    if value is None:
        return None, None
    # Find all resource names that appear in the value string
    found: list[str] = []
    for r in _RESOURCE_ORDER:
        # Find all occurrences
        start = 0
        while True:
            idx = value.find(r, start)
            if idx == -1:
                break
            found.append(r)
            start = idx + len(r)
    # Convention: first resource mentioned is given, second is received
    give = found[0] if len(found) >= 1 else None
    get = found[1] if len(found) >= 2 else None
    return give, get


def _try_parse_robber_coordinate(value: str | None) -> tuple | None:
    """Parse the robber hex coordinate from MOVE_ROBBER / PLAY_KNIGHT value.

    Format: ((x,y,z),...) — we extract the first parenthesised triple.
    """
    if value is None:
        return None
    m = re.match(r"^\(\((-?\d+),(-?\d+),(-?\d+)\)", value.strip())
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _node_production_score(ps: PublicState | None, node_id: int) -> float:
    """Total pip-weighted production at a node.  Normalised to ~[0, 1]."""
    if ps is None:
        return 0.0
    tiles = ps.node_production.get(node_id, ())
    return sum(_pip_count(n) for _, n in tiles) / 15.0  # max ~15 pips


def _node_resource_diversity(ps: PublicState | None, node_id: int) -> float:
    """Number of distinct resources produced at this node, normalised."""
    if ps is None:
        return 0.0
    tiles = ps.node_production.get(node_id, ())
    return len({r for r, _ in tiles}) / 5.0  # max 5 resources (very rare: 3 typical)


def _node_has_port(ps: PublicState | None, node_id: int) -> float:
    """1.0 if the node is on any port, else 0.0."""
    if ps is None:
        return 0.0
    for _res, nodes in ps.port_nodes.items():
        if node_id in nodes:
            return 1.0
    return 0.0


def _robber_hex_pips(ps: PublicState | None, coord: tuple | None) -> float:
    """Pip value of a robber target hex, normalised."""
    if ps is None or coord is None:
        return 0.0
    for tile in ps.tiles:
        if tile.coordinate == coord and tile.number is not None:
            return _pip_count(tile.number) / 5.0  # max 5 pips (for 6 or 8)
    return 0.0


# =====================================================================
# Action features
# =====================================================================

def action_features(ea: EncodedAction, ps: PublicState | None = None) -> list[float]:
    """Extract a flat float vector of length ACTION_DIM from one EncodedAction.

    Args:
        ea: The encoded action to featurise.
        ps: The current PublicState, used to compute spatial features.
            If None, spatial features are all zeros (backward-compatible
            with tests that construct actions without a game state).

    Features:
      [0..13]   one-hot action type
      [14]      is_mandatory
      [15]      is_build
      [16]      is_trade
      [17]      is_devcard_play
      [18]      has_value

      --- Spatial features (new) ---
      [19]      node_production_score — pip-weighted production at target node
      [20]      node_resource_diversity — distinct resources at target node
      [21]      node_has_port — target node is a port node
      [22..26]  trade_give_resource — 5-dim one-hot for resource given away
      [27..31]  trade_get_resource  — 5-dim one-hot for resource received
      [32]      robber_target_pips  — pip value of robber destination hex
    """
    feats = [0.0] * ACTION_DIM

    # --- Base features (unchanged) --- [0..18]
    idx = _ACTION_TYPE_TO_IDX.get(ea.action_type, _ACTION_TYPE_TO_IDX["OTHER"])
    feats[idx] = 1.0

    offset = len(_ACTION_TYPES)
    feats[offset + 0] = 1.0 if ea.action_type in _MANDATORY_TYPES else 0.0
    feats[offset + 1] = 1.0 if ea.action_type in _BUILD_TYPES else 0.0
    feats[offset + 2] = 1.0 if ea.action_type in _TRADE_TYPES else 0.0
    feats[offset + 3] = 1.0 if ea.action_type in _DEV_PLAY_TYPES else 0.0
    feats[offset + 4] = 1.0 if ea.value is not None else 0.0

    # --- Spatial features --- [19..32]
    spatial_offset = _BASE_DIM  # 19

    # Node-based features: BUILD_SETTLEMENT, BUILD_CITY target a node_id.
    # BUILD_ROAD targets an edge — use the average of its two endpoint nodes.
    if ea.action_type in ("BUILD_SETTLEMENT", "BUILD_CITY"):
        node_id = _try_parse_node_id(ea.value)
        if node_id is not None:
            feats[spatial_offset + 0] = _node_production_score(ps, node_id)
            feats[spatial_offset + 1] = _node_resource_diversity(ps, node_id)
            feats[spatial_offset + 2] = _node_has_port(ps, node_id)

    elif ea.action_type == "BUILD_ROAD":
        edge = _try_parse_edge(ea.value)
        if edge is not None:
            n_a, n_b = edge
            # Average production potential of the two endpoint nodes
            feats[spatial_offset + 0] = (
                _node_production_score(ps, n_a) + _node_production_score(ps, n_b)
            ) / 2.0
            feats[spatial_offset + 1] = (
                _node_resource_diversity(ps, n_a) + _node_resource_diversity(ps, n_b)
            ) / 2.0
            feats[spatial_offset + 2] = max(
                _node_has_port(ps, n_a), _node_has_port(ps, n_b)
            )

    # Trade features: which resource is given, which is received.
    elif ea.action_type == "MARITIME_TRADE":
        give, get = _try_parse_trade_resources(ea.value)
        give_offset = spatial_offset + 3     # [22..26]
        get_offset = spatial_offset + 3 + 5  # [27..31]
        if give is not None and give in _RESOURCE_TO_IDX:
            feats[give_offset + _RESOURCE_TO_IDX[give]] = 1.0
        if get is not None and get in _RESOURCE_TO_IDX:
            feats[get_offset + _RESOURCE_TO_IDX[get]] = 1.0

    # Robber features: pip value of the target hex.
    elif ea.action_type in ("MOVE_ROBBER", "PLAY_KNIGHT_CARD"):
        coord = _try_parse_robber_coordinate(ea.value)
        feats[spatial_offset + 3 + 5 + 5] = _robber_hex_pips(ps, coord)  # [32]

    assert len(feats) == ACTION_DIM, f"Expected {ACTION_DIM}, got {len(feats)}"
    return feats
