"""Simulation probe: step through a Catanatron game tick-by-tick.

Creates a small (2-player) game and steps up to 30 ticks, printing:
  - tick number
  - current player colour
  - number of legal actions
  - the action chosen by the random player

Run:
    python scripts/sim_probe.py
"""

from catanatron import Color, Game, RandomPlayer

MAX_TICKS = 30


def main() -> None:
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=42)

    print(f"Game {game.id}  |  players: {[p.color.value for p in players]}")
    print(f"{'tick':>4}  {'player':<8}  {'#actions':>8}  action")
    print("-" * 60)

    for tick in range(1, MAX_TICKS + 1):
        color = game.state.current_color()
        actions = game.playable_actions
        num_actions = len(actions)

        # play_tick lets the current player's decide() pick an action,
        # executes it, and returns the ActionRecord.
        record = game.play_tick()
        action = record.action

        action_desc = f"{action.action_type.value}"
        if action.value is not None:
            action_desc += f"  value={action.value}"

        print(f"{tick:>4}  {color.value:<8}  {num_actions:>8}  {action_desc}")

        if game.winning_color() is not None:
            print(f"\nGame over! Winner: {game.winning_color().value}")
            break
    else:
        print(f"\n(stopped after {MAX_TICKS} ticks — game still in progress)")


if __name__ == "__main__":
    main()
