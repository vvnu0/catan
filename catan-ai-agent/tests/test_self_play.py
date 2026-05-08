"""Tests for self-play data generation pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from catan_ai.models.action_features import ACTION_DIM
from catan_ai.training.self_play import SelfPlayConfig, run_self_play


@pytest.fixture
def tiny_config(tmp_path: Path) -> SelfPlayConfig:
    """Minimal self-play config for fast testing."""
    return SelfPlayConfig(
        num_games=2,
        output_dir=str(tmp_path / "sp_data"),
        seed=99,
        max_simulations=5,
        max_depth=4,
        shard_size=9999,
    )


class TestSelfPlay:
    """Self-play data generation tests."""

    def test_produces_shards(self, tiny_config: SelfPlayConfig):
        """run_self_play writes at least one shard file."""
        out = run_self_play(tiny_config)
        shards = list(Path(out).glob("shard_*.pt"))
        assert len(shards) >= 1

    def test_sample_structure(self, tiny_config: SelfPlayConfig):
        """Each sample has the expected keys and tensor shapes."""
        out = run_self_play(tiny_config)
        shard = torch.load(next(Path(out).glob("shard_*.pt")), weights_only=False)
        assert len(shard) > 0

        sample = shard[0]
        assert "state_feats" in sample
        assert "action_feats" in sample
        assert "target_policy" in sample
        assert "target_value" in sample
        assert "encoded_actions" in sample
        assert "meta" in sample

        s = sample["state_feats"]
        assert s.ndim == 1
        assert s.shape[0] == 53

        a = sample["action_feats"]
        assert a.ndim == 2
        assert a.shape[1] == ACTION_DIM

        p = sample["target_policy"]
        assert p.shape[0] == a.shape[0]

    def test_target_policy_sums_to_one(self, tiny_config: SelfPlayConfig):
        """MCTS visit distribution should be normalised."""
        out = run_self_play(tiny_config)
        shard = torch.load(next(Path(out).glob("shard_*.pt")), weights_only=False)
        for sample in shard[:10]:
            total = sample["target_policy"].sum().item()
            assert abs(total - 1.0) < 1e-5, f"Policy sums to {total}"

    def test_target_value_filled(self, tiny_config: SelfPlayConfig):
        """After game ends, target_value is non-zero for at least one player."""
        out = run_self_play(tiny_config)
        shard = torch.load(next(Path(out).glob("shard_*.pt")), weights_only=False)
        values = [s["target_value"].item() for s in shard]
        assert any(v != 0.0 for v in values), "All target values are zero"

    def test_deterministic_across_runs(self, tmp_path: Path):
        """Same seed → same samples."""
        cfg1 = SelfPlayConfig(
            num_games=1,
            output_dir=str(tmp_path / "run1"),
            seed=42,
            max_simulations=5,
            max_depth=4,
        )
        cfg2 = SelfPlayConfig(
            num_games=1,
            output_dir=str(tmp_path / "run2"),
            seed=42,
            max_simulations=5,
            max_depth=4,
        )
        run_self_play(cfg1)
        run_self_play(cfg2)

        s1 = torch.load(next(Path(cfg1.output_dir).glob("shard_*.pt")), weights_only=False)
        s2 = torch.load(next(Path(cfg2.output_dir).glob("shard_*.pt")), weights_only=False)

        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            assert torch.allclose(a["state_feats"], b["state_feats"])
            assert torch.allclose(a["target_policy"], b["target_policy"])

    def test_metadata_present(self, tiny_config: SelfPlayConfig):
        """Metadata fields are populated."""
        out = run_self_play(tiny_config)
        shard = torch.load(next(Path(out).glob("shard_*.pt")), weights_only=False)
        meta = shard[0]["meta"]
        assert "game_id" in meta
        assert "ply" in meta
        assert "color" in meta
        assert "teacher" in meta
        assert "config_hash" in meta
