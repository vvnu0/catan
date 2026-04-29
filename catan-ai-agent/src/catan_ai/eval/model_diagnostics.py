"""Compact diagnostics for policy/value checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from catan_ai.models.policy_value_net import PolicyValueNet
from catan_ai.training.collate import collate_fn
from catan_ai.training.dataset import SelfPlayDataset


@torch.no_grad()
def compute_model_diagnostics(
    *,
    model: PolicyValueNet,
    data_dir: str | Path,
    checkpoint: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    max_samples: int = 256,
    batch_size: int = 64,
    flat_entropy_threshold: float = 0.95,
) -> dict[str, Any]:
    """Compute held-out policy/value diagnostics for one checkpoint."""
    dataset = SelfPlayDataset(data_dir)
    sample_count = len(dataset)
    if sample_count == 0:
        raise RuntimeError(f"No diagnostic data in {data_dir}")

    eval_count = min(max_samples, sample_count)
    start = max(0, sample_count - eval_count)
    subset = Subset(dataset, list(range(start, sample_count)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    entropies: list[float] = []
    normalized_entropies: list[float] = []
    top1_matches = 0
    states = 0
    value_abs_errors: list[float] = []
    value_sq_errors: list[float] = []
    max_probs: list[float] = []

    model.eval()
    for batch in loader:
        logits, values = model(
            batch["state_feats"],
            batch["action_feats"],
            batch["action_mask"],
        )
        probs = torch.softmax(logits, dim=-1).masked_fill(~batch["action_mask"], 0.0)
        target = batch["target_policy"]
        valid_counts = batch["action_mask"].sum(dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

        for i in range(probs.shape[0]):
            n = int(valid_counts[i].item())
            denom = math.log(max(n, 2))
            ent = float(entropy[i].item())
            entropies.append(ent)
            normalized_entropies.append(ent / denom if denom > 0 else 0.0)
            max_probs.append(float(probs[i].max().item()))
            if int(torch.argmax(probs[i]).item()) == int(torch.argmax(target[i]).item()):
                top1_matches += 1
            states += 1

        err = values - batch["target_value"]
        value_abs_errors.extend(torch.abs(err).tolist())
        value_sq_errors.extend((err * err).tolist())

    flat_count = sum(1 for h in normalized_entropies if h >= flat_entropy_threshold)
    diagnostics = {
        "self_play_sample_count": sample_count,
        "diagnostic_sample_count": states,
        "checkpoint_epoch": (checkpoint or {}).get("epoch"),
        "checkpoint_metrics": (checkpoint or {}).get("metrics", {}),
        "mean_policy_entropy": _mean(entropies),
        "mean_normalized_policy_entropy": _mean(normalized_entropies),
        "flat_policy_fraction": flat_count / states if states else 0.0,
        "nonflat_policy_fraction": 1.0 - (flat_count / states if states else 0.0),
        "mean_max_policy_probability": _mean(max_probs),
        "top1_match_rate": top1_matches / states if states else 0.0,
        "value_mae": _mean(value_abs_errors),
        "value_mse": _mean(value_sq_errors),
    }
    if checkpoint_path is not None:
        history_path = Path(checkpoint_path).parent / "training_history.json"
        if history_path.exists():
            diagnostics["train_loss_history"] = json.loads(
                history_path.read_text(encoding="utf-8")
            )
    return diagnostics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
