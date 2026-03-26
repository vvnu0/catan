"""Run matches showcasing the HeuristicBot against baseline opponents.

Runs two matches:
  1. HeuristicBot (RED) vs DebugPlayer (BLUE)
  2. HeuristicBot (RED) vs RandomPlayer (BLUE)

Prints winner, turn count, and how many times each bot was called.

Run:
    python scripts/run_heuristic_match.py
"""

from catanatron import Color, Game, RandomPlayer

from catan_ai.players import DebugPlayer, HeuristicBot

SEED = 42


def run_match(player_red, player_blue, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    game = Game([player_red, player_blue], seed=SEED)
    winner = game.play()

    print(f"  Winner:     {winner.value if winner else 'None (turn limit)'}")
    print(f"  Turns:      {game.state.num_turns}")
    print(f"  RED calls:  {player_red._calls}")
    print(f"  BLUE calls: {player_blue._calls}")


def main() -> None:
    # Match 1: HeuristicBot vs DebugPlayer
    heuristic1 = HeuristicBot(Color.RED)
    debug = DebugPlayer(Color.BLUE)
    run_match(heuristic1, debug, "HeuristicBot vs DebugPlayer")

    # Match 2: HeuristicBot vs RandomPlayer
    heuristic2 = HeuristicBot(Color.RED)
    random_player = RandomPlayer(Color.BLUE)
    # RandomPlayer doesn't track _calls, so we wrap it.
    random_player._calls = 0
    original_decide = random_player.decide

    def counting_decide(game, actions):
        random_player._calls += 1
        return original_decide(game, actions)

    random_player.decide = counting_decide
    run_match(heuristic2, random_player, "HeuristicBot vs RandomPlayer")


if __name__ == "__main__":
    main()
