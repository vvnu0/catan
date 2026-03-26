"""Tests for the belief-aware determinized search layer."""

from __future__ import annotations

import random

from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import RESOURCES
from catanatron.state_functions import (
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

from catan_ai.belief.determinizer import Determinizer
from catan_ai.belief.devcard_belief import DevCardBelief
from catan_ai.belief.public_history import (
    RESOURCE_TYPES,
    extract_public_evidence,
)
from catan_ai.belief.resource_belief import PER_RESOURCE_TOTAL, ResourceBelief
from catan_ai.players.belief_mcts_player import BeliefMCTSConfig, BeliefMCTSPlayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game(seed=0, ticks=0):
    p1 = RandomPlayer(Color.RED)
    p2 = RandomPlayer(Color.BLUE)
    game = Game([p1, p2], seed=seed)
    for _ in range(ticks):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


def _advance_past_initial(game, n=30):
    for _ in range(n):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


# ---------------------------------------------------------------------------
# 1. Belief model construction from a real game
# ---------------------------------------------------------------------------

def test_public_evidence_construction():
    game = _make_game(seed=5, ticks=20)
    if game.winning_color() is not None:
        return
    ev = extract_public_evidence(game, Color.RED)

    assert ev.acting_color == "RED"
    assert set(ev.bank_resources.keys()) == set(RESOURCE_TYPES)
    assert set(ev.own_resources.keys()) == set(RESOURCE_TYPES)
    assert all(v >= 0 for v in ev.bank_resources.values())
    assert all(v >= 0 for v in ev.own_resources.values())
    assert len(ev.opponent_evidence) == 1
    assert ev.opponent_evidence[0].color == "BLUE"
    assert ev.dev_deck_size >= 0


def test_resource_belief_conservation():
    game = _make_game(seed=7, ticks=20)
    if game.winning_color() is not None:
        return
    ev = extract_public_evidence(game, Color.RED)
    rb = ResourceBelief(ev, mode="conservation")
    rng = random.Random(42)
    hands = rb.sample_opponent_hands(rng)

    assert "BLUE" in hands
    blue_hand = hands["BLUE"]
    for r in RESOURCE_TYPES:
        expected = PER_RESOURCE_TOTAL - ev.bank_resources[r] - ev.own_resources[r]
        assert blue_hand[r] == expected, f"{r}: {blue_hand[r]} != {expected}"


# ---------------------------------------------------------------------------
# 2. Determinized worlds preserve total resource counts
# ---------------------------------------------------------------------------

def test_sampled_world_total_resources():
    game = _make_game(seed=10, ticks=30)
    if game.winning_color() is not None:
        return
    det = Determinizer(acting_color=Color.RED, belief_mode="conservation")
    rng = random.Random(99)
    world = det.sample_world(game, rng)
    assert world is not None

    for i, r in enumerate(RESOURCE_TYPES):
        total = world.state.resource_freqdeck[i]
        for c in world.state.colors:
            key = player_key(world.state, c)
            total += world.state.player_state[f"{key}_{r}_IN_HAND"]
        assert total == PER_RESOURCE_TOTAL, f"{r} total={total} != {PER_RESOURCE_TOTAL}"


# ---------------------------------------------------------------------------
# 3. Determinized worlds preserve opponent total hand sizes
# ---------------------------------------------------------------------------

def test_sampled_world_opponent_hand_size():
    game = _make_game(seed=12, ticks=30)
    if game.winning_color() is not None:
        return
    original_blue_count = player_num_resource_cards(game.state, Color.BLUE)

    det = Determinizer(acting_color=Color.RED, belief_mode="conservation")
    rng = random.Random(77)
    world = det.sample_world(game, rng)
    assert world is not None

    sampled_count = player_num_resource_cards(world.state, Color.BLUE)
    assert sampled_count == original_blue_count


# ---------------------------------------------------------------------------
# 4. Acting player's own hand is unchanged
# ---------------------------------------------------------------------------

def test_own_hand_unchanged():
    game = _make_game(seed=15, ticks=30)
    if game.winning_color() is not None:
        return
    key = player_key(game.state, Color.RED)
    original = {
        r: game.state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCE_TYPES
    }

    det = Determinizer(acting_color=Color.RED, belief_mode="conservation")
    rng = random.Random(55)
    world = det.sample_world(game, rng)
    assert world is not None

    wkey = player_key(world.state, Color.RED)
    for r in RESOURCE_TYPES:
        assert world.state.player_state[f"{wkey}_{r}_IN_HAND"] == original[r]


# ---------------------------------------------------------------------------
# 5. Sampling with fixed seed is deterministic
# ---------------------------------------------------------------------------

def test_sampling_deterministic():
    game = _make_game(seed=20, ticks=30)
    if game.winning_color() is not None:
        return
    det = Determinizer(acting_color=Color.RED, belief_mode="conservation")

    def _sample():
        rng = random.Random(123)
        w = det.sample_world(game, rng)
        assert w is not None
        blue_key = player_key(w.state, Color.BLUE)
        return tuple(
            w.state.player_state[f"{blue_key}_{r}_IN_HAND"]
            for r in RESOURCE_TYPES
        )

    a = _sample()
    b = _sample()
    assert a == b


# ---------------------------------------------------------------------------
# 6. Invalid samples are rejected cleanly
# ---------------------------------------------------------------------------

def test_invalid_sample_rejection():
    """count_only mode can theoretically produce invalid samples in edge
    cases; the determinizer should handle rejection gracefully."""
    game = _make_game(seed=25, ticks=30)
    if game.winning_color() is not None:
        return
    det = Determinizer(
        acting_color=Color.RED,
        belief_mode="count_only",
        max_invalid_samples=5,
    )
    rng = random.Random(99)
    world = det.sample_world(game, rng)
    # Should either produce a valid world or None — not crash.
    if world is not None:
        for i, r in enumerate(RESOURCE_TYPES):
            assert world.state.resource_freqdeck[i] >= 0, f"Negative bank for {r}"


# ---------------------------------------------------------------------------
# 7. BeliefMCTSPlayer returns a legal raw action
# ---------------------------------------------------------------------------

def test_belief_player_returns_legal_action():
    cfg = BeliefMCTSConfig(num_worlds=2, sims_per_world=5, belief_seed=0)
    player = BeliefMCTSPlayer(Color.RED, config=cfg)
    opp = RandomPlayer(Color.BLUE)
    game = Game([player, opp], seed=30)

    # Advance to a state where the belief player has a real choice.
    for _ in range(4):
        if game.winning_color() is not None:
            break
        game.play_tick()
    if game.winning_color() is not None:
        return

    current = game.state.current_player()
    if isinstance(current, BeliefMCTSPlayer):
        action = current.decide(game, game.playable_actions)
        assert action in game.playable_actions


# ---------------------------------------------------------------------------
# 8. Live game is not mutated
# ---------------------------------------------------------------------------

def test_live_game_not_mutated():
    cfg = BeliefMCTSConfig(num_worlds=3, sims_per_world=5, belief_seed=0)
    player = BeliefMCTSPlayer(Color.RED, config=cfg)
    opp = RandomPlayer(Color.BLUE)
    game = Game([player, opp], seed=33)

    for _ in range(4):
        if game.winning_color() is not None:
            break
        game.play_tick()
    if game.winning_color() is not None:
        return

    turns_before = game.state.num_turns
    records_before = len(game.state.action_records)
    buildings_before = len(game.state.board.buildings)

    current = game.state.current_player()
    if isinstance(current, BeliefMCTSPlayer):
        current.decide(game, game.playable_actions)

    assert game.state.num_turns == turns_before
    assert len(game.state.action_records) == records_before
    assert len(game.state.board.buildings) == buildings_before


# ---------------------------------------------------------------------------
# 9. Small batch match runs
# ---------------------------------------------------------------------------

def test_small_batch_match():
    cfg = BeliefMCTSConfig(num_worlds=2, sims_per_world=3, belief_seed=0)
    player = BeliefMCTSPlayer(Color.RED, config=cfg)
    opp = RandomPlayer(Color.BLUE)
    game = Game([player, opp], seed=40)
    game.play()
    assert game.state.num_turns > 0
    assert player._calls > 0


# ---------------------------------------------------------------------------
# 10. Dev-card belief (when enabled)
# ---------------------------------------------------------------------------

def test_devcard_belief_pool():
    game = _make_game(seed=50, ticks=30)
    if game.winning_color() is not None:
        return
    ev = extract_public_evidence(game, Color.RED)
    db = DevCardBelief(ev)
    assert db.total_unseen >= 0


def test_devcard_sampling_preserves_counts():
    game = _make_game(seed=55, ticks=30)
    if game.winning_color() is not None:
        return
    original_blue_dev = player_num_dev_cards(game.state, Color.BLUE)

    det = Determinizer(
        acting_color=Color.RED,
        belief_mode="conservation",
        enable_devcard_sampling=True,
    )
    rng = random.Random(42)
    world = det.sample_world(game, rng)
    if world is None:
        return

    sampled_blue_dev = player_num_dev_cards(world.state, Color.BLUE)
    assert sampled_blue_dev == original_blue_dev
