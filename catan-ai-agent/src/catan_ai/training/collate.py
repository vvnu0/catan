"""Collate function that pads variable-length action sets for batching.

Each sample has a different number of legal actions.  This function:
  1. Pads ``action_feats`` to the max action count in the batch.
  2. Pads ``target_policy`` likewise (with 0.0 for invalid slots).
  3. Creates an ``action_mask`` tensor (True = valid action slot).
  4. Stacks ``state_feats`` and ``target_value`` normally.
"""

from __future__ import annotations

import torch


def collate_fn(batch: list[dict]) -> dict:
    """Collate a list of sample dicts into a padded batch dict.

    Returns dict with keys:
      - state_feats:   [B, S]      float
      - action_feats:  [B, A_max, F] float (zero-padded)
      - action_mask:   [B, A_max]  bool
      - target_policy: [B, A_max]  float (zero-padded)
      - target_value:  [B]         float
    """
    B = len(batch)
    a_max = max(sample["action_feats"].shape[0] for sample in batch)
    F = batch[0]["action_feats"].shape[1]

    state_feats = torch.stack([s["state_feats"] for s in batch])           # [B, S]
    target_value = torch.stack([s["target_value"] for s in batch])         # [B]

    action_feats = torch.zeros(B, a_max, F)
    target_policy = torch.zeros(B, a_max)
    action_mask = torch.zeros(B, a_max, dtype=torch.bool)

    for i, sample in enumerate(batch):
        n = sample["action_feats"].shape[0]
        action_feats[i, :n] = sample["action_feats"]
        target_policy[i, :n] = sample["target_policy"]
        action_mask[i, :n] = True

    return {
        "state_feats": state_feats,
        "action_feats": action_feats,
        "action_mask": action_mask,
        "target_policy": target_policy,
        "target_value": target_value,
    }
