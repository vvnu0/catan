"""PyTorch Dataset for self-play training samples.

Each sample has a variable number of legal actions, so the dataset
returns raw (un-padded) tensors.  Padding and masking happen in
``collate_fn`` (see ``catan_ai.training.collate``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class SelfPlayDataset(Dataset):
    """Loads all shards from a directory into memory.

    Each element is a dict with keys:
      - state_feats:    Tensor[S]
      - action_feats:   Tensor[A, F]  (variable A per sample)
      - target_policy:  Tensor[A]
      - target_value:   Tensor[]      (scalar)
      - encoded_actions: list[str]
      - meta:           dict
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.samples: list[dict] = []

        shard_paths = sorted(self.data_dir.glob("shard_*.pt"))
        if not shard_paths:
            log.warning("No shards found in %s", self.data_dir)
            return

        for path in shard_paths:
            shard = torch.load(path, weights_only=False)
            self.samples.extend(shard)

        log.info(
            "Loaded %d samples from %d shards in %s",
            len(self.samples),
            len(shard_paths),
            self.data_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
