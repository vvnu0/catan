"""Seed management for reproducible games and experiments.

Usage:
    from catan_ai.utils.seeding import make_seed
    seed = make_seed()        # random seed
    seed = make_seed(42)      # deterministic seed
"""

import random


def make_seed(seed: int | None = None) -> int:
    """Return *seed* unchanged if provided, otherwise generate a fresh one."""
    if seed is not None:
        return seed
    return random.randint(0, 2**31 - 1)
