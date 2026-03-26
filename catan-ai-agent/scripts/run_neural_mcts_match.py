"""Run matches comparing NeuralMCTSPlayer against all baselines.

Usage:
    python scripts/run_neural_mcts_match.py --checkpoint PATH [--games N]

If no checkpoint is provided, NeuralMCTSPlayer falls back to heuristic-only
MCTS (equivalent to MCTSPlayer with PUCT exploration constant).
"""

from __future__ import annotations

import argparse
import logging

from catanatron import Color, RandomPlayer

from catan_ai.eval.arena import Arena
from catan_ai.players import HeuristicBot, MCTSPlayer
from catan_ai.players.neural_mcts_player import NeuralMCTSConfig, NeuralMCTSPlayer
from catan_ai.training.checkpoints import load_checkpoint

log = logging.getLogger(__name__)

NEURAL_CFG = NeuralMCTSConfig(
    max_simulations=50,
    max_depth=10,
    puct_c=2.5,
    top_k_roads=3,
    top_k_trades=2,
    top_k_robber=4,
    seed=0,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NeuralMCTSPlayer")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=10, help="Games per pairing")
    parser.add_argument("--seed", type=int, default=3000, help="Base game seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    model = None
    if args.checkpoint:
        model, _ckpt = load_checkpoint(args.checkpoint)
        print(f"Loaded model from {args.checkpoint}")
    else:
        print("No checkpoint — using heuristic-only fallback")

    arena = Arena(num_games=args.games, base_seed=args.seed)

    def make_neural(color: Color) -> NeuralMCTSPlayer:
        return NeuralMCTSPlayer(color, model=model, config=NEURAL_CFG)

    pairings = [
        (make_neural, lambda c: RandomPlayer(c), "NeuralMCTS vs RandomPlayer"),
        (make_neural, lambda c: HeuristicBot(c), "NeuralMCTS vs HeuristicBot"),
        (
            make_neural,
            lambda c: MCTSPlayer(c, max_simulations=50, max_depth=10, seed=0),
            "NeuralMCTS vs MCTSPlayer",
        ),
    ]

    print(f"\n{'=' * 60}")
    print(f"  NeuralMCTSPlayer benchmark ({args.games} games per pairing)")
    print(f"{'=' * 60}")

    for make_cand, make_base, label in pairings:
        result = arena.compare(make_cand, make_base, label)
        print(f"\n  {result.summary()}")

    print()


if __name__ == "__main__":
    main()
