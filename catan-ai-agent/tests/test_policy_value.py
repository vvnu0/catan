"""Tests for feature extraction, model, collation, and neural MCTS integration."""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import pytest
import torch

from catanatron import Color, Game, RandomPlayer

from catan_ai.adapters.catanatron_adapter import public_state_from_game
from catan_ai.adapters.public_state import EncodedAction
from catan_ai.models.action_features import (
    ACTION_DIM,
    STATE_DIM,
    action_features,
    state_features,
)
from catan_ai.models.policy_value_net import PolicyValueNet
from catan_ai.training.collate import collate_fn
from catan_ai.training.checkpoints import load_checkpoint, save_checkpoint


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_game():
    """Create a 2-player game advanced a few ticks."""
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)
    for _ in range(30):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


def _make_ps():
    game = _make_game()
    return public_state_from_game(game, Color.RED)


# -----------------------------------------------------------------------
# Feature extraction tests
# -----------------------------------------------------------------------

class TestStateFeatures:
    def test_length(self):
        ps = _make_ps()
        feats = state_features(ps)
        assert len(feats) == STATE_DIM

    def test_all_float(self):
        ps = _make_ps()
        feats = state_features(ps)
        assert all(isinstance(f, float) for f in feats)

    def test_deterministic(self):
        ps = _make_ps()
        assert state_features(ps) == state_features(ps)


class TestActionFeatures:
    def test_length(self):
        ea = EncodedAction(color="RED", action_type="BUILD_CITY", value="42")
        feats = action_features(ea)
        assert len(feats) == ACTION_DIM

    def test_one_hot_set(self):
        ea = EncodedAction(color="RED", action_type="BUILD_CITY", value="42")
        feats = action_features(ea)
        assert feats[3] == 1.0  # BUILD_CITY is index 3

    def test_unknown_type_maps_to_other(self):
        ea = EncodedAction(color="RED", action_type="UNKNOWN_TYPE", value=None)
        feats = action_features(ea)
        assert feats[13] == 1.0  # OTHER is index 13

    def test_flags(self):
        ea = EncodedAction(color="RED", action_type="ROLL", value=None)
        feats = action_features(ea)
        assert feats[14] == 1.0  # is_mandatory
        assert feats[18] == 0.0  # has_value (ROLL has no value)


# -----------------------------------------------------------------------
# Model tests
# -----------------------------------------------------------------------

class TestPolicyValueNet:
    def test_forward_shapes(self):
        model = PolicyValueNet(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=32)
        B, A_max = 4, 7
        s = torch.randn(B, STATE_DIM)
        a = torch.randn(B, A_max, ACTION_DIM)
        m = torch.ones(B, A_max, dtype=torch.bool)
        m[0, 5:] = False  # some masked

        logits, value = model(s, a, m)
        assert logits.shape == (B, A_max)
        assert value.shape == (B,)

    def test_masked_logits_are_neg_inf(self):
        model = PolicyValueNet(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=32)
        B, A_max = 2, 5
        s = torch.randn(B, STATE_DIM)
        a = torch.randn(B, A_max, ACTION_DIM)
        m = torch.ones(B, A_max, dtype=torch.bool)
        m[0, 3:] = False
        m[1, 4:] = False

        logits, _ = model(s, a, m)
        assert logits[0, 3].item() == float("-inf")
        assert logits[0, 4].item() == float("-inf")
        assert logits[1, 4].item() == float("-inf")
        assert logits[1, 3].item() != float("-inf")

    def test_value_in_range(self):
        model = PolicyValueNet(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=32)
        s = torch.randn(8, STATE_DIM)
        a = torch.randn(8, 3, ACTION_DIM)
        m = torch.ones(8, 3, dtype=torch.bool)

        _, value = model(s, a, m)
        assert (value >= -1.0).all() and (value <= 1.0).all()

    def test_softmax_over_valid_sums_to_one(self):
        model = PolicyValueNet(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=32)
        B, A_max = 3, 6
        s = torch.randn(B, STATE_DIM)
        a = torch.randn(B, A_max, ACTION_DIM)
        m = torch.ones(B, A_max, dtype=torch.bool)
        m[0, 4:] = False

        logits, _ = model(s, a, m)
        probs = torch.softmax(logits, dim=-1)
        # Valid positions should sum to ~1.0
        valid_sum = (probs[0, :4]).sum().item()
        assert abs(valid_sum - 1.0) < 1e-5


# -----------------------------------------------------------------------
# Collation tests
# -----------------------------------------------------------------------

class TestCollate:
    def test_pads_to_max_actions(self):
        samples = [
            {
                "state_feats": torch.randn(STATE_DIM),
                "action_feats": torch.randn(3, ACTION_DIM),
                "target_policy": torch.tensor([0.5, 0.3, 0.2]),
                "target_value": torch.tensor(1.0),
            },
            {
                "state_feats": torch.randn(STATE_DIM),
                "action_feats": torch.randn(5, ACTION_DIM),
                "target_policy": torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]),
                "target_value": torch.tensor(-1.0),
            },
        ]
        batch = collate_fn(samples)

        assert batch["state_feats"].shape == (2, STATE_DIM)
        assert batch["action_feats"].shape == (2, 5, ACTION_DIM)
        assert batch["action_mask"].shape == (2, 5)
        assert batch["target_policy"].shape == (2, 5)
        assert batch["target_value"].shape == (2,)

        assert batch["action_mask"][0, 2].item() is True
        assert batch["action_mask"][0, 3].item() is False
        assert batch["action_mask"][1, 4].item() is True

    def test_padding_values_are_zero(self):
        samples = [
            {
                "state_feats": torch.randn(STATE_DIM),
                "action_feats": torch.randn(2, ACTION_DIM),
                "target_policy": torch.tensor([0.6, 0.4]),
                "target_value": torch.tensor(0.0),
            },
            {
                "state_feats": torch.randn(STATE_DIM),
                "action_feats": torch.randn(4, ACTION_DIM),
                "target_policy": torch.tensor([0.25, 0.25, 0.25, 0.25]),
                "target_value": torch.tensor(1.0),
            },
        ]
        batch = collate_fn(samples)
        # Padded action features should be zero
        assert batch["action_feats"][0, 2:].abs().sum().item() == 0.0
        # Padded policy should be zero
        assert batch["target_policy"][0, 2:].abs().sum().item() == 0.0


# -----------------------------------------------------------------------
# Checkpoint tests
# -----------------------------------------------------------------------

class TestCheckpoints:
    def test_save_and_load(self, tmp_path: Path):
        model = PolicyValueNet(hidden_dim=32)
        path = tmp_path / "test_ckpt.pt"
        save_checkpoint(model, None, epoch=5, metrics={"loss": 0.5}, path=path)

        loaded, ckpt = load_checkpoint(path)
        assert ckpt["epoch"] == 5
        assert ckpt["metrics"]["loss"] == 0.5

        # Weights should be identical
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), loaded.state_dict().items()
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2)


# -----------------------------------------------------------------------
# Tiny overfit test
# -----------------------------------------------------------------------

class TestTinyOverfit:
    def test_loss_decreases(self):
        """Train on a tiny batch and verify loss goes down."""
        model = PolicyValueNet(hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Synthetic batch
        B, A = 8, 4
        s = torch.randn(B, STATE_DIM)
        a_feats = torch.randn(B, A, ACTION_DIM)
        mask = torch.ones(B, A, dtype=torch.bool)

        target_p = torch.softmax(torch.randn(B, A), dim=-1)
        target_v = torch.randn(B).clamp(-1, 1)

        losses = []
        for _ in range(50):
            logits, value = model(s, a_feats, mask)
            log_probs = torch.log_softmax(logits, dim=-1)
            p_loss = -(target_p * log_probs).sum(dim=-1).mean()
            v_loss = torch.nn.functional.mse_loss(value, target_v)
            loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


# -----------------------------------------------------------------------
# NeuralMCTSPlayer integration tests
# -----------------------------------------------------------------------

class TestNeuralMCTSPlayer:
    def test_returns_legal_action(self):
        from catan_ai.players.neural_mcts_player import NeuralMCTSConfig, NeuralMCTSPlayer

        model = PolicyValueNet(hidden_dim=32)
        cfg = NeuralMCTSConfig(max_simulations=5, max_depth=4, seed=0)
        player = NeuralMCTSPlayer(Color.RED, model=model, config=cfg)

        game = Game([player, RandomPlayer(Color.BLUE)], seed=1)
        for _ in range(20):
            if game.winning_color() is not None:
                break
            game.play_tick()

    def test_no_model_fallback(self):
        from catan_ai.players.neural_mcts_player import NeuralMCTSConfig, NeuralMCTSPlayer

        cfg = NeuralMCTSConfig(max_simulations=5, max_depth=4, seed=0)
        player = NeuralMCTSPlayer(Color.RED, model=None, config=cfg)

        game = Game([player, RandomPlayer(Color.BLUE)], seed=1)
        for _ in range(20):
            if game.winning_color() is not None:
                break
            game.play_tick()

    def test_does_not_mutate_game(self):
        from catan_ai.players.neural_mcts_player import NeuralMCTSConfig, NeuralMCTSPlayer

        model = PolicyValueNet(hidden_dim=32)
        cfg = NeuralMCTSConfig(max_simulations=5, max_depth=4, seed=0)

        game = Game(
            [NeuralMCTSPlayer(Color.RED, model=model, config=cfg), RandomPlayer(Color.BLUE)],
            seed=1,
        )

        # Advance to a non-trivial state
        for _ in range(20):
            if game.winning_color() is not None:
                break
            game.play_tick()

        turns_before = game.state.num_turns
        actions = game.playable_actions
        if len(actions) > 1:
            # Call decide directly to test non-mutation
            player = game.state.current_player()
            if hasattr(player, "decide"):
                player.decide(game, actions)
                assert game.state.num_turns == turns_before


class TestArena:
    def test_small_batch(self):
        from catan_ai.eval.arena import Arena

        arena = Arena(num_games=2, base_seed=100)
        result = arena.compare(
            lambda c: RandomPlayer(c),
            lambda c: RandomPlayer(c),
            "random vs random",
        )
        assert result.games == 2
        assert result.wins + result.losses + result.draws == 2
