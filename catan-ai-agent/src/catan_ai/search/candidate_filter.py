"""Candidate filter to keep MCTS branching factor manageable.

Filters the encoded action list at each tree node so that MCTS only
expands the most promising candidates.  Always keeps mandatory, winning,
structural-build, and END_TURN actions.  Limits roads and trades to
top-K by a lightweight score.
"""

from __future__ import annotations

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.public_state import EncodedAction, PublicState

# Action types that are always kept (no filtering).
_ALWAYS_KEEP = frozenset({
    "ROLL",
    "DISCARD",
    "BUILD_CITY",
    "BUILD_SETTLEMENT",
    "BUY_DEVELOPMENT_CARD",
    "PLAY_KNIGHT_CARD",
    "PLAY_YEAR_OF_PLENTY",
    "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING",
    "END_TURN",
    # Trade actions that are not MARITIME_TRADE are rare but keep them.
    "OFFER_TRADE",
    "ACCEPT_TRADE",
    "REJECT_TRADE",
    "CONFIRM_TRADE",
    "CANCEL_TRADE",
})


def _pip_count(number: int) -> int:
    return 6 - abs(7 - number)


def _road_score(ps: PublicState, ea: EncodedAction) -> float:
    """Quick heuristic score for a BUILD_ROAD action."""
    if ps.is_initial_build_phase or ps.is_road_building:
        return 1000.0  # keep all roads during placement or free road phase

    try:
        inner = ea.value.strip("()")  # type: ignore[union-attr]
        a_str, b_str = inner.split(",")
        a, b = int(a_str), int(b_str)
    except (ValueError, AttributeError):
        return 0.0

    pip_a = sum(_pip_count(n) for _, n in ps.node_production.get(a, ()))
    pip_b = sum(_pip_count(n) for _, n in ps.node_production.get(b, ()))
    return max(pip_a, pip_b)


def _trade_score(_ps: PublicState, _ea: EncodedAction) -> float:
    """Flat score for MARITIME_TRADE; no deep evaluation in v1."""
    return 0.0


def _robber_score(ps: PublicState, ea: EncodedAction) -> float:
    """Prefer robber placements that hurt strongest opponents on high-pip tiles."""
    try:
        parts = ea.value.split("),")  # type: ignore[union-attr]
        coord_str = parts[0].lstrip("(")
        target_str = parts[1].rstrip(")")
        coord = tuple(int(x) for x in coord_str.split(","))
        target = None if target_str == "None" else target_str
    except (ValueError, AttributeError, IndexError):
        return 0.0

    tile_pip = 0.0
    for tile in ps.tiles:
        if tile.coordinate == coord and tile.number is not None:
            tile_pip = _pip_count(tile.number)
            break

    if target == ps.acting_color:
        return -100.0

    strongest = max(
        (s for s in ps.player_summaries if s.color != ps.acting_color),
        key=lambda s: s.visible_victory_points,
        default=None,
    )
    strongest_bonus = 8.0 if strongest is not None and target == strongest.color else 0.0
    return tile_pip + strongest_bonus


class CandidateFilter:
    """Configurable filter that trims legal actions for MCTS expansion.

    Args:
        top_k_roads: Keep at most this many BUILD_ROAD actions (by score).
        top_k_trades: Keep at most this many MARITIME_TRADE actions.
    """

    def __init__(
        self,
        top_k_roads: int = 3,
        top_k_trades: int = 2,
        top_k_robber: int = 4,
    ):
        self.top_k_roads = top_k_roads
        self.top_k_trades = top_k_trades
        self.top_k_robber = top_k_robber

    def __call__(
        self, ps: PublicState, encoded_actions: tuple[EncodedAction, ...]
    ) -> list[EncodedAction]:
        return self.filter(ps, encoded_actions)

    def filter(
        self, ps: PublicState, encoded_actions: tuple[EncodedAction, ...]
    ) -> list[EncodedAction]:
        """Return a filtered, deterministically ordered list of actions."""
        kept: list[EncodedAction] = []
        roads: list[tuple[float, EncodedAction]] = []
        trades: list[tuple[float, EncodedAction]] = []
        robbers: list[tuple[float, EncodedAction]] = []

        for ea in encoded_actions:
            if ea.action_type in _ALWAYS_KEEP:
                kept.append(ea)
            elif ea.action_type == "BUILD_ROAD":
                roads.append((_road_score(ps, ea), ea))
            elif ea.action_type == "MARITIME_TRADE":
                trades.append((_trade_score(ps, ea), ea))
            elif ea.action_type == "MOVE_ROBBER":
                robbers.append((_robber_score(ps, ea), ea))
            else:
                kept.append(ea)

        # Keep top-K roads (sorted by score descending, ties by encoded key).
        if ps.is_initial_build_phase or ps.is_road_building:
            kept.extend(ea for _, ea in roads)
        else:
            roads.sort(key=lambda t: (-t[0], ActionCodec.sort_key(t[1])))
            kept.extend(ea for _, ea in roads[: self.top_k_roads])

        trades.sort(key=lambda t: (-t[0], ActionCodec.sort_key(t[1])))
        kept.extend(ea for _, ea in trades[: self.top_k_trades])

        robbers.sort(key=lambda t: (-t[0], ActionCodec.sort_key(t[1])))
        kept.extend(ea for _, ea in robbers[: self.top_k_robber])

        return kept
