"""Run matches showcasing MCTSPlayer against baseline opponents.

Runs three matches:
  1. MCTSPlayer vs DebugPlayer
  2. MCTSPlayer vs HeuristicBot
  3. MCTSPlayer vs RandomPlayer

Run:
    python scripts/run_mcts_match.py
"""

from catanatron import Color, Game, RandomPlayer

from catan_ai.players import DebugPlayer, HeuristicBot, MCTSPlayer

SEED = 42
MCTS_SIMS = 50
MCTS_DEPTH = 10
TOP_K_ROADS = 3
TOP_K_TRADES = 2
TOP_K_ROBBER = 4


def run_match(player_red, player_blue, label: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {label}")
    print(f"{'=' * 64}")

    game = Game([player_red, player_blue], seed=SEED)
    winner = game.play()

    winner_str = winner.value if winner else "None (turn limit)"
    print(f"  Winner:           {winner_str}")
    print(f"  Turns:            {game.state.num_turns}")

    red_calls = getattr(player_red, "_calls", "?")
    blue_calls = getattr(player_blue, "_calls", "?")
    print(f"  RED calls:        {red_calls}")
    print(f"  BLUE calls:       {blue_calls}")

    if isinstance(player_red, MCTSPlayer):
        print(f"  RED avg move ms:  {player_red.avg_move_ms:.1f}")
    if isinstance(player_blue, MCTSPlayer):
        print(f"  BLUE avg move ms: {player_blue.avg_move_ms:.1f}")


def main() -> None:
    print(
        "MCTS budget: "
        f"{MCTS_SIMS} sims, depth={MCTS_DEPTH}, "
        f"k_roads={TOP_K_ROADS}, k_trades={TOP_K_TRADES}, k_robber={TOP_K_ROBBER}"
    )

    # Match 1
    run_match(
        MCTSPlayer(
            Color.RED,
            max_simulations=MCTS_SIMS,
            max_depth=MCTS_DEPTH,
            top_k_roads=TOP_K_ROADS,
            top_k_trades=TOP_K_TRADES,
            top_k_robber=TOP_K_ROBBER,
            seed=0,
        ),
        DebugPlayer(Color.BLUE),
        "MCTSPlayer vs DebugPlayer",
    )

    # Match 2
    run_match(
        MCTSPlayer(
            Color.RED,
            max_simulations=MCTS_SIMS,
            max_depth=MCTS_DEPTH,
            top_k_roads=TOP_K_ROADS,
            top_k_trades=TOP_K_TRADES,
            top_k_robber=TOP_K_ROBBER,
            seed=0,
        ),
        HeuristicBot(Color.BLUE),
        "MCTSPlayer vs HeuristicBot",
    )

    # Match 3
    run_match(
        MCTSPlayer(
            Color.RED,
            max_simulations=MCTS_SIMS,
            max_depth=MCTS_DEPTH,
            top_k_roads=TOP_K_ROADS,
            top_k_trades=TOP_K_TRADES,
            top_k_robber=TOP_K_ROBBER,
            seed=0,
        ),
        RandomPlayer(Color.BLUE),
        "MCTSPlayer vs RandomPlayer",
    )


if __name__ == "__main__":
    main()
