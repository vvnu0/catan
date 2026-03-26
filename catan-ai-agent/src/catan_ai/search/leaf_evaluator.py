"""Heuristic leaf evaluator for MCTS v1.

Returns a value in [-1, +1] from the *root player's* perspective using
only PublicState features.  No random rollouts, no neural network.

Does NOT read raw hidden opponent identities — only uses information
available through the PublicState adapter.
"""

from __future__ import annotations

from catanatron.game import Game
from catanatron.models.player import Color

from catan_ai.adapters.catanatron_adapter import public_state_from_game
from catan_ai.adapters.public_state import PublicState


def _pip_count(number: int) -> int:
    return 6 - abs(7 - number)


def evaluate_leaf(game: Game, root_color: Color) -> float:
    """Estimate the game value from *root_color*'s perspective.

    Returns a float in [-1.0, +1.0]:
      +1.0 = root_color wins
      -1.0 = opponent wins
       0.0 = perfectly even
    """
    winner = game.winning_color()
    if winner is not None:
        return 1.0 if winner == root_color else -1.0

    # Build PublicState from root's perspective so we can see our own
    # hidden VP cards but NOT opponent hidden state.
    ps = public_state_from_game(game, root_color)

    own = _find_summary(ps, root_color.value)
    opp = _find_opponent(ps, root_color.value)
    if own is None or opp is None:
        return 0.0

    # --- Feature 1: VP difference (strongest signal) ---
    own_vp = own.visible_victory_points + ps.own_dev_cards.get("VICTORY_POINT", 0)
    opp_vp = opp.visible_victory_points
    vp_diff = (own_vp - opp_vp) / ps.vps_to_win

    # --- Feature 2: production quality difference ---
    own_prod = _production_score(ps, root_color.value)
    opp_prod = _production_score(ps, opp.color)
    prod_total = own_prod + opp_prod
    prod_diff = (own_prod - opp_prod) / max(prod_total, 1.0)

    # --- Feature 3: longest road / largest army awards ---
    road_bonus = 0.0
    if own.has_longest_road:
        road_bonus = 0.10
    elif opp.has_longest_road:
        road_bonus = -0.10

    army_bonus = 0.0
    if own.has_largest_army:
        army_bonus = 0.10
    elif opp.has_largest_army:
        army_bonus = -0.10

    # --- Feature 4: city / settlement structure counts ---
    structure_diff = (
        (own.num_cities - opp.num_cities) * 0.04
        + (own.num_settlements - opp.num_settlements) * 0.02
    )

    # --- Feature 5: port access ---
    own_ports = _port_access_count(ps, own.color)
    opp_ports = _port_access_count(ps, opp.color)
    port_diff = (own_ports - opp_ports) * 0.015

    # --- Feature 6: useful build opportunities at this node ---
    useful_build_diff = _useful_build_action_diff(ps, own.color, opp.color)

    # --- Feature 7: settlement expansion potential ---
    settle_diff = (own.settlements_available - opp.settlements_available) * 0.02

    # --- Feature 8: resource card count advantage (count only, not identity) ---
    res_diff = (own.num_resource_cards - opp.num_resource_cards) * 0.01

    # --- Feature 9: robber pressure (avoid self-blocking) ---
    robber_diff = _robber_pressure_diff(ps, own.color, opp.color)

    # --- Weighted combination ---
    raw = (
        0.95 * vp_diff
        + 0.28 * prod_diff
        + road_bonus
        + army_bonus
        + structure_diff
        + port_diff
        + useful_build_diff
        + settle_diff
        + res_diff
        + robber_diff
    )
    return max(-1.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_summary(ps: PublicState, color_str: str):
    for s in ps.player_summaries:
        if s.color == color_str:
            return s
    return None


def _find_opponent(ps: PublicState, root_color_str: str):
    for s in ps.player_summaries:
        if s.color != root_color_str:
            return s
    return None


def _production_score(ps: PublicState, color_str: str) -> float:
    """Sum of pip-weighted production across all buildings owned by *color_str*."""
    total = 0.0
    for b in ps.buildings:
        if b.color != color_str:
            continue
        multiplier = 2.0 if b.building_type == "CITY" else 1.0
        pips = sum(_pip_count(n) for _, n in ps.node_production.get(b.node_id, ()))
        total += pips * multiplier
    return total


def _port_access_count(ps: PublicState, color_str: str) -> int:
    port_nodes = {
        node for nodes in ps.port_nodes.values() for node in nodes
    }
    return sum(
        1
        for b in ps.buildings
        if b.color == color_str and b.node_id in port_nodes
    )


def _useful_build_action_diff(ps: PublicState, own_color: str, opp_color: str) -> float:
    own_build = 0
    opp_build = 0
    for ea in ps.legal_actions:
        if ea.action_type not in {"BUILD_CITY", "BUILD_SETTLEMENT"}:
            continue
        if ea.color == own_color:
            own_build += 1
        elif ea.color == opp_color:
            opp_build += 1
    return (own_build - opp_build) * 0.015


def _robber_pressure_diff(ps: PublicState, own_color: str, opp_color: str) -> float:
    """Positive if robber hurts opponent more than root player."""
    own_block = 0.0
    opp_block = 0.0

    robber_coord = ps.robber_coordinate
    robber_resource = None
    robber_number = None
    for tile in ps.tiles:
        if tile.coordinate != robber_coord or tile.number is None:
            continue
        robber_resource = tile.resource
        robber_number = tile.number
        robber_pip = _pip_count(tile.number)
        break
    else:
        robber_pip = 0.0

    if robber_pip <= 0 or robber_resource is None or robber_number is None:
        return 0.0

    for b in ps.buildings:
        adjacent = ps.node_production.get(b.node_id, ())
        if any(
            num == robber_number and res == robber_resource for res, num in adjacent
        ):
            if b.color == own_color:
                own_block += robber_pip
            elif b.color == opp_color:
                opp_block += robber_pip

    return (opp_block - own_block) * 0.01
