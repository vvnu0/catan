"""Generate self-play training data using the MCTS teacher.

Usage:
    python scripts/run_self_play.py [--games N] [--sims S] [--output DIR]
"""

from __future__ import annotations

import argparse
import logging

from catan_ai.training.self_play import SelfPlayConfig, run_self_play


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCTS self-play data generation")
    parser.add_argument("--games", type=int, default=10, help="Number of self-play games")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per decision")
    parser.add_argument("--depth", type=int, default=10, help="MCTS max search depth")
    parser.add_argument("--output", type=str, default="data/self_play", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--shard-size", type=int, default=512, help="Samples per shard file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = SelfPlayConfig(
        num_games=args.games,
        output_dir=args.output,
        seed=args.seed,
        max_simulations=args.sims,
        max_depth=args.depth,
        shard_size=args.shard_size,
    )

    print(
        f"Self-play config: {cfg.num_games} games, "
        f"{cfg.max_simulations} sims/decision, "
        f"depth={cfg.max_depth}, seed={cfg.seed}"
    )

    out_dir = run_self_play(cfg)
    print(f"\nData written to: {out_dir}")


if __name__ == "__main__":
    main()
