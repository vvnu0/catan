"""Training loop for the policy/value network.

Losses:
  - Policy:  KL divergence against the normalised MCTS visit distribution.
  - Value:   MSE against the final game outcome in [-1, +1].

Supports:
  - train/validation split
  - early stopping by validation loss
  - best-checkpoint saving
  - a tiny "overfit" mode for debugging (``overfit_batches``)
"""

from __future__ import annotations

import logging
import math
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from catan_ai.models.policy_value_net import PolicyValueNet
from catan_ai.training.checkpoints import save_checkpoint
from catan_ai.training.collate import collate_fn
from catan_ai.training.dataset import SelfPlayDataset

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    data_dir: str = "data/self_play"
    checkpoint_dir: str = "checkpoints"
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    validation_split: float = 0.1
    value_loss_weight: float = 1.0
    patience: int = 5
    overfit_batches: int = 0

    # Model architecture
    hidden_dim: int = 64
    dropout: float = 0.0


def train(cfg: TrainConfig) -> Path:
    """Run the full training loop.  Returns path to best checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Training on device: %s", device)

    dataset = SelfPlayDataset(cfg.data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No training data in {cfg.data_dir}")

    # --- Train / val split ---
    n_val = max(1, int(len(dataset) * cfg.validation_split))
    n_train = len(dataset) - n_val
    indices = list(range(len(dataset)))
    train_ds = Subset(dataset, indices[:n_train])
    val_ds = Subset(dataset, indices[n_train:])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = PolicyValueNet(
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = math.inf
    best_path = ckpt_dir / "best.pt"
    patience_left = cfg.patience
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        # --- Training ---
        model.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if cfg.overfit_batches > 0 and batch_idx >= cfg.overfit_batches:
                break

            s = batch["state_feats"].to(device)
            a = batch["action_feats"].to(device)
            m = batch["action_mask"].to(device)
            tp = batch["target_policy"].to(device)
            tv = batch["target_value"].to(device)

            logits, value = model(s, a, m)

            p_loss = _policy_loss(logits, tp, m)
            v_loss = F.mse_loss(value, tv)
            loss = p_loss + cfg.value_loss_weight * v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_policy_loss += p_loss.item()
            train_value_loss += v_loss.item()
            train_batches += 1

        avg_tp = train_policy_loss / max(train_batches, 1)
        avg_tv = train_value_loss / max(train_batches, 1)

        # --- Validation ---
        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                s = batch["state_feats"].to(device)
                a = batch["action_feats"].to(device)
                m = batch["action_mask"].to(device)
                tp = batch["target_policy"].to(device)
                tv = batch["target_value"].to(device)

                logits, value = model(s, a, m)
                p_loss = _policy_loss(logits, tp, m)
                v_loss = F.mse_loss(value, tv)

                val_policy_loss += p_loss.item()
                val_value_loss += v_loss.item()
                val_batches += 1

        avg_vp = val_policy_loss / max(val_batches, 1)
        avg_vv = val_value_loss / max(val_batches, 1)
        val_total = avg_vp + cfg.value_loss_weight * avg_vv
        history.append({
            "epoch": epoch,
            "train_policy": avg_tp,
            "train_value": avg_tv,
            "val_policy": avg_vp,
            "val_value": avg_vv,
            "val_total": val_total,
        })

        log.info(
            "Epoch %d/%d — train: policy=%.4f value=%.4f | val: policy=%.4f value=%.4f",
            epoch,
            cfg.epochs,
            avg_tp,
            avg_tv,
            avg_vp,
            avg_vv,
        )

        # --- Checkpoint / early stopping ---
        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_left = cfg.patience
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {
                    "train_policy": avg_tp,
                    "train_value": avg_tv,
                    "val_policy": avg_vp,
                    "val_value": avg_vv,
                    "val_total": val_total,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("Early stopping at epoch %d", epoch)
                break

        # Periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"train_policy": avg_tp, "train_value": avg_tv},
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

    log.info("Training complete.  Best checkpoint: %s", best_path)
    (ckpt_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    return best_path


def _policy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between MCTS visit distribution and model policy.

    ``target`` is already a normalised probability distribution from the
    MCTS teacher.  We compute cross-entropy: -sum(target * log_softmax(logits))
    over valid (masked) positions only.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Replace -inf with 0.0 in log_probs before multiplication to avoid
    # nan from 0 * -inf at padded positions.
    log_probs = log_probs.masked_fill(~mask, 0.0)
    target_masked = target * mask.float()
    loss = -(target_masked * log_probs).sum(dim=-1).mean()
    return loss
