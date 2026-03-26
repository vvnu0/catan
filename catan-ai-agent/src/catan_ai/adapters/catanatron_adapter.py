"""Bridge between raw Catanatron game objects and our PublicState.

This is the *only* module that should import and read raw Catanatron state
fields.  Everything downstream receives a PublicState instead.
"""

from __future__ import annotations

from catanatron.game import Game
from catanatron.models.enums import (
    CITY,
    DEVELOPMENT_CARDS,
    RESOURCES,
    ROAD,
    SETTLEMENT,
)
from catanatron.models.player import Color
from catanatron.state_functions import (
    get_dev_cards_in_hand,
    get_longest_road_length,
    get_played_dev_cards,
    get_visible_victory_points,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.public_state import (
    BuildingSummary,
    PublicPlayerSummary,
    PublicState,
    RoadSummary,
    TileSummary,
)


def public_state_from_game(game: Game, acting_color: Color) -> PublicState:
    """Build a PublicState snapshot from a live Catanatron Game.

    Only information that is legally observable by *acting_color* in a real
    game of Catan is included.
    """
    state = game.state

    tiles = _extract_tiles(state)
    buildings = _extract_buildings(state)
    roads = _extract_roads(state)
    player_summaries = tuple(
        _build_player_summary(state, color) for color in state.colors
    )

    bank_resources = dict(zip(
        [r for r in RESOURCES],
        state.resource_freqdeck,
    ))

    # The acting player knows their own hand (private to them).
    own_resources = _own_resources(state, acting_color)
    own_dev_cards = _own_dev_cards(state, acting_color)

    # Encode legal actions into stable keys.
    encoded_actions = ActionCodec.sorted_actions(
        ActionCodec.encode_many(game.playable_actions)
    )

    node_production = _extract_node_production(state)
    port_nodes = _extract_port_nodes(state)

    return PublicState(
        acting_color=acting_color.value,
        turn_number=state.num_turns,
        vps_to_win=game.vps_to_win,
        tiles=tiles,
        robber_coordinate=tuple(state.board.robber_coordinate),
        buildings=buildings,
        roads=roads,
        player_summaries=player_summaries,
        bank_resources=bank_resources,
        bank_dev_cards_remaining=len(state.development_listdeck),
        own_resources=own_resources,
        own_dev_cards=own_dev_cards,
        legal_actions=tuple(encoded_actions),
        is_initial_build_phase=state.is_initial_build_phase,
        is_discarding=state.is_discarding,
        is_moving_robber=state.is_moving_knight,
        is_road_building=state.is_road_building,
        free_roads_available=state.free_roads_available,
        node_production=node_production,
        port_nodes=port_nodes,
    )


# ---------------------------------------------------------------------------
# Private extraction helpers
# ---------------------------------------------------------------------------

def _extract_tiles(state) -> tuple[TileSummary, ...]:
    """Board tile layout — fully public, static after setup."""
    summaries = []
    for coord, tile in state.board.map.land_tiles.items():
        summaries.append(TileSummary(
            coordinate=tuple(coord),
            resource=getattr(tile, "resource", None),
            number=getattr(tile, "number", None),
        ))
    return tuple(summaries)


def _extract_buildings(state) -> tuple[BuildingSummary, ...]:
    """Buildings on the board — visible to everyone."""
    result = []
    for node_id, (color, btype) in state.board.buildings.items():
        result.append(BuildingSummary(
            node_id=node_id,
            color=color.value,
            building_type=btype,
        ))
    return tuple(sorted(result, key=lambda b: b.node_id))


def _extract_roads(state) -> tuple[RoadSummary, ...]:
    """Roads on the board — visible to everyone.

    Catanatron stores both orientations of each edge; we deduplicate here.
    """
    seen: set[tuple[int, int]] = set()
    result = []
    for edge, color in state.board.roads.items():
        canonical = tuple(sorted(edge))
        if canonical not in seen:
            seen.add(canonical)
            result.append(RoadSummary(edge=canonical, color=color.value))
    return tuple(sorted(result, key=lambda r: r.edge))


def _build_player_summary(state, color: Color) -> PublicPlayerSummary:
    """Public-only summary for one player."""
    key = player_key(state, color)
    ps = state.player_state

    return PublicPlayerSummary(
        color=color.value,
        # VICTORY_POINTS is the public count (excludes hidden VP dev cards).
        visible_victory_points=get_visible_victory_points(state, color),
        num_settlements=len(state.buildings_by_color[color].get(SETTLEMENT, [])),
        num_cities=len(state.buildings_by_color[color].get(CITY, [])),
        num_roads=len(state.buildings_by_color[color].get(ROAD, [])),
        # Counts only — not which resources or which dev cards.
        num_resource_cards=player_num_resource_cards(state, color),
        num_dev_cards=player_num_dev_cards(state, color),
        # Played dev cards are public (announced when used).
        played_knights=get_played_dev_cards(state, color, "KNIGHT"),
        played_year_of_plenty=get_played_dev_cards(state, color, "YEAR_OF_PLENTY"),
        played_monopoly=get_played_dev_cards(state, color, "MONOPOLY"),
        played_road_building=get_played_dev_cards(state, color, "ROAD_BUILDING"),
        has_longest_road=ps[f"{key}_HAS_ROAD"],
        has_largest_army=ps[f"{key}_HAS_ARMY"],
        longest_road_length=get_longest_road_length(state, color),
        roads_available=ps[f"{key}_ROADS_AVAILABLE"],
        settlements_available=ps[f"{key}_SETTLEMENTS_AVAILABLE"],
        cities_available=ps[f"{key}_CITIES_AVAILABLE"],
    )


def _extract_node_production(state) -> dict[int, tuple[tuple[str, int], ...]]:
    """Per-node production: (resource, dice_number) for each adjacent producing tile.

    This is public board-layout info (visible to everyone).  Desert tiles
    (resource=None) are excluded.
    """
    result: dict[int, tuple[tuple[str, int], ...]] = {}
    for node_id, tiles in state.board.map.adjacent_tiles.items():
        producing = tuple(
            (t.resource, t.number)
            for t in tiles
            if t.resource is not None and t.number is not None
        )
        result[node_id] = producing
    return result


def _extract_port_nodes(state) -> dict[str | None, tuple[int, ...]]:
    """Port access: which nodes can use each port type.

    Key is the resource string for 2:1 ports, or None for 3:1 ports.
    """
    return {
        res: tuple(sorted(node_ids))
        for res, node_ids in state.board.map.port_nodes.items()
    }


def _own_resources(state, color: Color) -> dict[str, int]:
    """The acting player's own resource hand (private to them)."""
    key = player_key(state, color)
    return {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES}


def _own_dev_cards(state, color: Color) -> dict[str, int]:
    """The acting player's own dev-card hand (private to them)."""
    key = player_key(state, color)
    return {d: state.player_state[f"{key}_{d}_IN_HAND"] for d in DEVELOPMENT_CARDS}
