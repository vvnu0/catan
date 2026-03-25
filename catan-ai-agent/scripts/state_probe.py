"""State probe: dump raw game/state fields for manual inspection.

Plays a short game (50 ticks) then prints structured summaries of:
  - board tiles and robber position
  - buildings on the board
  - roads on the board
  - per-player state (resources, dev cards, victory points, piece counts)
  - bank state (resource supply, remaining dev cards)
  - game-level flags

This is meant for understanding what information is public vs hidden.

Run:
    python scripts/state_probe.py
"""

from collections import defaultdict

from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import (
    CITY,
    DEVELOPMENT_CARDS,
    RESOURCES,
    ROAD,
    SETTLEMENT,
)

TICKS = 50


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_board_summary(state) -> None:
    section("Board Summary")
    board = state.board

    print(f"\nRobber location: {board.robber_coordinate}")

    print(f"\nLand tiles ({len(board.map.land_tiles)}):")
    for coord, tile in board.map.land_tiles.items():
        res = getattr(tile, "resource", None)
        num = getattr(tile, "number", None)
        print(f"  {coord}  resource={res}  number={num}")


def print_buildings_summary(state) -> None:
    section("Buildings Summary")

    # From board.buildings (public — visible on the board)
    board = state.board
    print(f"\nboard.buildings ({len(board.buildings)} occupied nodes):")
    for node_id, (color, building_type) in sorted(board.buildings.items()):
        print(f"  node {node_id:>3}: {color.value} {building_type}")

    # From state.buildings_by_color (same data, grouped by colour)
    print("\nstate.buildings_by_color:")
    for color, btype_map in state.buildings_by_color.items():
        for btype in (SETTLEMENT, CITY):
            nodes = btype_map.get(btype, [])
            if nodes:
                print(f"  {color.value:>8} {btype}: {nodes}")


def print_roads_summary(state) -> None:
    section("Roads Summary")
    board = state.board

    roads_by_color: dict[str, list] = defaultdict(list)
    seen = set()
    for edge, color in board.roads.items():
        canonical = tuple(sorted(edge))
        if canonical not in seen:
            seen.add(canonical)
            roads_by_color[color.value].append(canonical)

    for color_name, edges in roads_by_color.items():
        print(f"\n  {color_name} ({len(edges)} roads):")
        for e in edges:
            print(f"    {e}")


def print_player_summaries(state) -> None:
    section("Player State Summaries")

    for i, player in enumerate(state.players):
        prefix = f"P{i}_"
        color = player.color.value
        print(f"\n--- Player {i}: {color} ---")

        # Victory points (actual vs visible)
        actual_vp = state.player_state[f"{prefix}ACTUAL_VICTORY_POINTS"]
        visible_vp = state.player_state[f"{prefix}VICTORY_POINTS"]
        print(f"  Victory points: visible={visible_vp}  actual={actual_vp}")

        # Resources in hand (hidden from other players)
        hand = {r: state.player_state[f"{prefix}{r}_IN_HAND"] for r in RESOURCES}
        print(f"  Resources in hand: {hand}")

        # Dev cards in hand (hidden)
        dev_hand = {
            d: state.player_state[f"{prefix}{d}_IN_HAND"] for d in DEVELOPMENT_CARDS
        }
        dev_played = {
            d: state.player_state[f"{prefix}PLAYED_{d}"] for d in DEVELOPMENT_CARDS
        }
        print(f"  Dev cards in hand:  {dev_hand}")
        print(f"  Dev cards played:   {dev_played}")

        # Piece counts
        roads_avail = state.player_state[f"{prefix}ROADS_AVAILABLE"]
        sett_avail = state.player_state[f"{prefix}SETTLEMENTS_AVAILABLE"]
        cities_avail = state.player_state[f"{prefix}CITIES_AVAILABLE"]
        print(f"  Pieces remaining: roads={roads_avail}  settlements={sett_avail}  cities={cities_avail}")

        # Longest road / largest army
        has_road = state.player_state[f"{prefix}HAS_ROAD"]
        has_army = state.player_state[f"{prefix}HAS_ARMY"]
        road_len = state.player_state[f"{prefix}LONGEST_ROAD_LENGTH"]
        print(f"  Longest road length: {road_len}  (has_road_award={has_road})")
        print(f"  Has largest army: {has_army}")


def print_bank_and_flags(state) -> None:
    section("Bank & Game Flags")

    resource_names = list(RESOURCES)
    bank = dict(zip(resource_names, state.resource_freqdeck))
    print(f"\nBank resources: {bank}")
    print(f"Dev cards remaining in bank: {len(state.development_listdeck)}")

    print(f"\nTurn number:           {state.num_turns}")
    print(f"Current player index:  {state.current_player_index}")
    print(f"Current prompt:        {state.current_prompt}")
    print(f"Initial build phase:   {state.is_initial_build_phase}")
    print(f"Is discarding:         {state.is_discarding}")
    print(f"Is moving robber:      {state.is_moving_knight}")
    print(f"Free roads available:  {state.free_roads_available}")


def main() -> None:
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=42)

    # Advance the game a bit so there is interesting state to inspect
    for _ in range(TICKS):
        if game.winning_color() is not None:
            break
        game.play_tick()

    print(f"Inspecting game {game.id} after up to {TICKS} ticks")
    print(f"Winner so far: {game.winning_color()}")

    print_board_summary(game.state)
    print_buildings_summary(game.state)
    print_roads_summary(game.state)
    print_player_summaries(game.state)
    print_bank_and_flags(game.state)


if __name__ == "__main__":
    main()
