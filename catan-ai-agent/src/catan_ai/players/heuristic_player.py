"""First real strategy bot, built entirely on PublicState.

This is a simple heuristic player — no search, no learning, no belief
sampling.  It scores each legal EncodedAction using only information
available in PublicState, picks the highest-scoring one, and returns the
corresponding raw action through the DecisionContext bridge.

Scoring philosophy (easy to read and edit, not optimal):
  - Mandatory actions (ROLL, DISCARD) are always taken immediately.
  - Winning moves get the highest score.
  - Cities > Settlements > Dev cards > Roads > Trading > END_TURN.
  - Initial placement uses pip-count and resource diversity.
  - Robber targets the strongest opponent on high-value tiles.
  - Ties are broken deterministically by EncodedAction sort order.
"""

from __future__ import annotations

import logging

from catanatron.models.player import Player

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.public_state import EncodedAction, PublicState
from catan_ai.players.decision_context import DecisionContext

log = logging.getLogger(__name__)


# =====================================================================
# Helpers
# =====================================================================

def _pip_count(number: int) -> int:
    """Dots on the dice for *number* (proportional to probability)."""
    return 6 - abs(7 - number)


def _node_pip_total(ps: PublicState, node_id: int) -> int:
    return sum(_pip_count(n) for _, n in ps.node_production.get(node_id, ()))


def _node_diversity(ps: PublicState, node_id: int) -> int:
    return len({r for r, _ in ps.node_production.get(node_id, ())})


def _own_summary(ps: PublicState):
    for s in ps.player_summaries:
        if s.color == ps.acting_color:
            return s
    raise ValueError("acting_color not found in player_summaries")


def _own_actual_vp(ps: PublicState) -> int:
    """Visible VP + hidden VP dev cards the acting player holds."""
    return (
        _own_summary(ps).visible_victory_points
        + ps.own_dev_cards.get("VICTORY_POINT", 0)
    )


def _strongest_opponent(ps: PublicState) -> str | None:
    """Color of the opponent with the most visible VP."""
    best_color, best_vp = None, -1
    for s in ps.player_summaries:
        if s.color == ps.acting_color:
            continue
        if s.visible_victory_points > best_vp:
            best_vp = s.visible_victory_points
            best_color = s.color
    return best_color


def _is_on_port(ps: PublicState, node_id: int) -> tuple[bool, str | None]:
    for res, nodes in ps.port_nodes.items():
        if node_id in nodes:
            return True, res
    return False, None


# =====================================================================
# Value parsing (EncodedAction.value is always a string)
# =====================================================================

def _parse_node_id(value: str | None) -> int:
    return int(value)  # type: ignore[arg-type]


def _parse_edge(value: str | None) -> tuple[int, int]:
    inner = value.strip("()")  # type: ignore[union-attr]
    a, b = inner.split(",")
    return int(a), int(b)


def _parse_robber_target(value: str | None) -> tuple[tuple[int, ...], str | None]:
    """Parse '((1,-1,0),BLUE)' → ((1,-1,0), 'BLUE')."""
    parts = value.split("),")  # type: ignore[union-attr]
    coord_str = parts[0].lstrip("(")
    target_str = parts[1].rstrip(")")
    coords = tuple(int(x) for x in coord_str.split(","))
    target = None if target_str == "None" else target_str
    return coords, target


# =====================================================================
# Scoring
# =====================================================================

_SCORE_MANDATORY = 10_000
_SCORE_WIN = 50_000


class HeuristicBot(Player):
    """Heuristic strategy bot that decides from PublicState only."""

    def __init__(self, color, is_bot=True):
        super().__init__(color, is_bot=is_bot)
        self._calls = 0

    # -----------------------------------------------------------------
    # Catanatron Player interface
    # -----------------------------------------------------------------
    def decide(self, game, playable_actions):
        self._calls += 1
        ctx = DecisionContext(game, playable_actions, self.color)
        ps = ctx.public_state

        scored: list[tuple[float, str, EncodedAction]] = []
        for ea in ps.legal_actions:
            score, rule = self._score(ps, ea)
            scored.append((score, rule, ea))

        # Highest score first; deterministic tie-break via encoded sort key.
        scored.sort(key=lambda t: (-t[0], ActionCodec.sort_key(t[2])))

        if log.isEnabledFor(logging.DEBUG):
            top = scored[:3]
            log.debug(
                "HeuristicBot %s call #%d — top 3: %s",
                self.color.value,
                self._calls,
                [(f"{s:.0f}", r, ActionCodec.decode_to_str(ea)) for s, r, ea in top],
            )

        best_score, best_rule, best_ea = scored[0]

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "  → chose %s (score=%.0f, rule=%s)",
                ActionCodec.decode_to_str(best_ea),
                best_score,
                best_rule,
            )

        return ctx.get_raw_action(best_ea)

    def reset_state(self):
        self._calls = 0

    # -----------------------------------------------------------------
    # Top-level dispatcher — returns (score, rule_name)
    # -----------------------------------------------------------------
    def _score(self, ps: PublicState, ea: EncodedAction) -> tuple[float, str]:
        at = ea.action_type
        if at == "ROLL":
            return _SCORE_MANDATORY, "mandatory"
        if at == "DISCARD":
            return _SCORE_MANDATORY, "mandatory"
        if at == "BUILD_CITY":
            return self._score_city(ps, ea), "build_city"
        if at == "BUILD_SETTLEMENT":
            return self._score_settlement(ps, ea), "build_settlement"
        if at == "BUY_DEVELOPMENT_CARD":
            return self._score_buy_dev(ps), "buy_dev"
        if at == "BUILD_ROAD":
            return self._score_road(ps, ea), "build_road"
        if at == "PLAY_KNIGHT_CARD":
            return self._score_play_knight(ps), "play_knight"
        if at == "PLAY_ROAD_BUILDING":
            return 150.0, "play_road_building"
        if at == "PLAY_YEAR_OF_PLENTY":
            return 180.0, "play_year_of_plenty"
        if at == "PLAY_MONOPOLY":
            return 180.0, "play_monopoly"
        if at == "MOVE_ROBBER":
            return self._score_robber(ps, ea), "move_robber"
        if at == "MARITIME_TRADE":
            return self._score_trade(ps, ea), "trade"
        if at == "END_TURN":
            return 0.0, "end_turn"
        # Unknown action types: above END_TURN so they're at least tried.
        return 1.0, "unknown"

    # -----------------------------------------------------------------
    # BUILD_CITY — +1 VP, doubles production at this node
    # -----------------------------------------------------------------
    def _score_city(self, ps: PublicState, ea: EncodedAction) -> float:
        node_id = _parse_node_id(ea.value)
        pip = _node_pip_total(ps, node_id)

        if _own_actual_vp(ps) + 1 >= ps.vps_to_win:
            return _SCORE_WIN

        # Cities are high value: VP gain + production boost.
        return 500.0 + pip * 20

    # -----------------------------------------------------------------
    # BUILD_SETTLEMENT — +1 VP, new production
    # -----------------------------------------------------------------
    def _score_settlement(self, ps: PublicState, ea: EncodedAction) -> float:
        node_id = _parse_node_id(ea.value)
        pip = _node_pip_total(ps, node_id)
        diversity = _node_diversity(ps, node_id)

        if _own_actual_vp(ps) + 1 >= ps.vps_to_win:
            return _SCORE_WIN

        on_port, port_res = _is_on_port(ps, node_id)
        port_bonus = 0.0
        if on_port:
            # 2:1 port matching a produced resource is extra useful.
            port_bonus = 15.0 if port_res is not None else 10.0

        if ps.is_initial_build_phase:
            # Position is critical in the opening — weight production heavily.
            return 400.0 + pip * 25 + diversity * 15 + port_bonus

        return 350.0 + pip * 15 + diversity * 10 + port_bonus

    # -----------------------------------------------------------------
    # BUY_DEVELOPMENT_CARD
    # -----------------------------------------------------------------
    def _score_buy_dev(self, ps: PublicState) -> float:
        # More valuable when close to winning (chance of VP card) or
        # when we need knights for largest army.
        own = _own_summary(ps)
        if _own_actual_vp(ps) + 1 >= ps.vps_to_win:
            # Could be a winning VP card — speculative but high-value.
            return 300.0
        army_bonus = 30.0 if not own.has_largest_army and own.played_knights >= 2 else 0.0
        return 200.0 + army_bonus

    # -----------------------------------------------------------------
    # BUILD_ROAD
    # -----------------------------------------------------------------
    def _score_road(self, ps: PublicState, ea: EncodedAction) -> float:
        if ps.is_initial_build_phase:
            return self._score_initial_road(ps, ea)
        if ps.is_road_building:
            # Free road from dev card — always take it.
            return 300.0

        own = _own_summary(ps)
        # Roads are mainly useful for expanding toward settlement spots.
        # Give a modest bonus if we have settlements to place.
        expansion_bonus = 30.0 if own.settlements_available > 0 else 0.0
        return 80.0 + expansion_bonus

    def _score_initial_road(self, ps: PublicState, ea: EncodedAction) -> float:
        """During initial placement, prefer roads toward high-production nodes."""
        a, b = _parse_edge(ea.value)
        # The endpoint that is NOT our just-placed settlement is the
        # expansion direction.  Score that node's production potential.
        score_a = _node_pip_total(ps, a) + _node_diversity(ps, a) * 5
        score_b = _node_pip_total(ps, b) + _node_diversity(ps, b) * 5
        expansion_score = max(score_a, score_b)
        return 300.0 + expansion_score * 3

    # -----------------------------------------------------------------
    # PLAY_KNIGHT_CARD
    # -----------------------------------------------------------------
    def _score_play_knight(self, ps: PublicState) -> float:
        own = _own_summary(ps)
        # More valuable if we're close to largest army.
        army_gap = 3 - own.played_knights
        if army_gap <= 1 and not own.has_largest_army:
            return 300.0  # one knight away from the army award
        return 250.0

    # -----------------------------------------------------------------
    # MOVE_ROBBER
    # -----------------------------------------------------------------
    def _score_robber(self, ps: PublicState, ea: EncodedAction) -> float:
        coord, steal_target = _parse_robber_target(ea.value)

        # Look up the tile's production value.
        tile_pip = 0.0
        for t in ps.tiles:
            if t.coordinate == coord and t.number is not None:
                tile_pip = _pip_count(t.number)
                break

        # Avoid blocking our own buildings (coord = robber destination tile;
        # if steal_target is our colour, we'd be hurting ourselves).
        if steal_target == ps.acting_color:
            return 100.0  # last resort only

        # Prefer stealing from the strongest opponent.
        strongest = _strongest_opponent(ps)
        target_bonus = 20.0 if steal_target == strongest else 0.0

        # Prefer high-production tiles (hurts the blocked player more).
        return 300.0 + tile_pip * 15 + target_bonus

    # -----------------------------------------------------------------
    # MARITIME_TRADE
    # -----------------------------------------------------------------
    def _score_trade(self, ps: PublicState, ea: EncodedAction) -> float:
        # Maritime trades are low priority unless they enable an immediate
        # build.  Full evaluation is complex, so keep it simple for now.
        return 50.0
