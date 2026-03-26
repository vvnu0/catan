"""Checkpoint save/load utilities for PolicyValueNet."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from catan_ai.models.policy_value_net import PolicyValueNet

log = logging.getLogger(__name__)


def save_checkpoint(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict,
    path: str | Path,
) -> Path:
    """Save a training checkpoint to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "model_config": {
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "hidden_dim": model.hidden_dim,
        },
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, path)
    log.info("Saved checkpoint to %s (epoch %d)", path, epoch)
    return path


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[PolicyValueNet, dict]:
    """Load a model from a checkpoint file.

    Returns ``(model, checkpoint_dict)`` where ``checkpoint_dict`` contains
    ``epoch``, ``metrics``, and optionally ``optimizer_state_dict``.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    cfg = ckpt.get("model_config", {})
    model = PolicyValueNet(
        state_dim=cfg.get("state_dim", 53),
        action_dim=cfg.get("action_dim", 19),
        hidden_dim=cfg.get("hidden_dim", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    log.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1))
    return model, ckpt
