"""Pretty-print a PublicState snapshot from a live Catanatron game.

This script advances a game partway, builds a PublicState for the acting
player, and prints every field — clearly labelling what information is
intentionally omitted by the masking layer.

Run:
    python scripts/public_state_probe.py
"""

from catanatron import Color, Game, RandomPlayer

from catan_ai.adapters import ActionCodec, public_state_from_game

TICKS = 50
SEED = 42


def section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def main() -> None:
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=SEED)

    for _ in range(TICKS):
        if game.winning_color() is not None:
            break
        game.play_tick()

    acting_color = game.state.current_color()
    ps = public_state_from_game(game, acting_color)

    section("PublicState Snapshot")
    print(f"  Acting color:  {ps.acting_color}")
    print(f"  Turn number:   {ps.turn_number}")
    print(f"  VPs to win:    {ps.vps_to_win}")

    section("Board Tiles (static after setup)")
    for t in ps.tiles:
        print(f"  {t.coordinate}  resource={t.resource}  number={t.number}")
    print(f"\n  Robber at: {ps.robber_coordinate}")

    section("Buildings on Board (public)")
    for b in ps.buildings:
        print(f"  node {b.node_id:>3}: {b.color} {b.building_type}")

    section("Roads on Board (public)")
    for r in ps.roads:
        print(f"  {r.edge}: {r.color}")

    section("Player Summaries (public info only)")
    for p in ps.player_summaries:
        marker = " <-- acting" if p.color == ps.acting_color else ""
        print(f"\n  --- {p.color}{marker} ---")
        print(f"  Visible VP:       {p.visible_victory_points}")
        print(f"  Settlements:      {p.num_settlements}")
        print(f"  Cities:           {p.num_cities}")
        print(f"  Roads:            {p.num_roads}")
        print(f"  Resource cards:   {p.num_resource_cards}  (count only, identities hidden)")
        print(f"  Dev cards:        {p.num_dev_cards}  (count only, identities hidden)")
        print(f"  Played knights:   {p.played_knights}")
        print(f"  Played YoP:       {p.played_year_of_plenty}")
        print(f"  Played Monopoly:  {p.played_monopoly}")
        print(f"  Played RoadBldg:  {p.played_road_building}")
        print(f"  Longest road:     {p.longest_road_length}  (award={p.has_longest_road})")
        print(f"  Largest army:     {p.has_largest_army}")
        print(f"  Pieces left:      roads={p.roads_available}  "
              f"settlements={p.settlements_available}  cities={p.cities_available}")

    section("Bank (public)")
    print(f"  Resources: {ps.bank_resources}")
    print(f"  Dev cards remaining: {ps.bank_dev_cards_remaining}")

    section("Acting Player's Own Hand (private to them)")
    print(f"  Resources: {ps.own_resources}")
    print(f"  Dev cards: {ps.own_dev_cards}")

    section("Phase Flags")
    print(f"  Initial build phase: {ps.is_initial_build_phase}")
    print(f"  Is discarding:       {ps.is_discarding}")
    print(f"  Is moving robber:    {ps.is_moving_robber}")
    print(f"  Is road building:    {ps.is_road_building}")
    print(f"  Free roads left:     {ps.free_roads_available}")

    section(f"Legal Actions ({len(ps.legal_actions)} encoded)")
    for ea in ps.legal_actions[:15]:
        print(f"  {ActionCodec.decode_to_str(ea)}")
    if len(ps.legal_actions) > 15:
        print(f"  ... +{len(ps.legal_actions) - 15} more")

    section("Intentionally Omitted Information")
    print("  - Opponent exact resource identities")
    print("  - Opponent exact dev-card identities")
    print("  - Dev-card deck order / composition")
    print("  - ACTUAL_VICTORY_POINTS for opponents (hidden VP cards)")
    print("  - Raw Catanatron State/Game objects")


if __name__ == "__main__":
    main()
