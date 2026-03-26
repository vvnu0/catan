"""Train the policy/value network from self-play data.

Usage:
    python scripts/train_policy_value.py [--data DIR] [--epochs N] [--lr F]
"""

from __future__ import annotations

import argparse
import logging

from catan_ai.training.train_policy_value import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train policy/value network")
    parser.add_argument("--data", type=str, default="data/self_play", help="Self-play data directory")
    parser.add_argument("--checkpoints", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer width")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument(
        "--overfit-batches",
        type=int,
        default=0,
        help="If >0, train on this many batches only (debugging)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = TrainConfig(
        data_dir=args.data,
        checkpoint_dir=args.checkpoints,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
        overfit_batches=args.overfit_batches,
    )

    print(
        f"Training config: data={cfg.data_dir}, "
        f"epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
        f"lr={cfg.lr}, hidden={cfg.hidden_dim}"
    )

    best_path = train(cfg)
    print(f"\nBest checkpoint: {best_path}")


if __name__ == "__main__":
    main()
