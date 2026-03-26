"""Tests for MCTS v1 — tree search, player, candidate filter."""

import copy

from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import ActionType

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.public_state import EncodedAction
from catan_ai.players import MCTSPlayer
from catan_ai.players.decision_context import DecisionContext
from catan_ai.search.candidate_filter import CandidateFilter
from catan_ai.search.mcts import MCTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game(seed=0, ticks=0):
    players = [MCTSPlayer(Color.RED, max_simulations=5, seed=0),
               RandomPlayer(Color.BLUE)]
    game = Game(players, seed=seed)
    for _ in range(ticks):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


# ---------------------------------------------------------------------------
# MCTSPlayer instantiation
# ---------------------------------------------------------------------------

def test_mcts_player_instantiation():
    p = MCTSPlayer(Color.RED, max_simulations=10, seed=42)
    assert p.color == Color.RED
    assert p.is_bot is True
    assert p._calls == 0


# ---------------------------------------------------------------------------
# Always returns a legal raw action
# ---------------------------------------------------------------------------

def test_returns_legal_action():
    game = _make_game(seed=0, ticks=0)
    player = game.state.current_player()
    if not isinstance(player, MCTSPlayer):
        game.play_tick()
        player = game.state.current_player()

    if isinstance(player, MCTSPlayer):
        action = player.decide(game, game.playable_actions)
        assert action in game.playable_actions


def test_returns_legal_action_mid_game():
    for seed in range(3):
        game = _make_game(seed=seed, ticks=20)
        if game.winning_color() is not None:
            continue
        player = game.state.current_player()
        if isinstance(player, MCTSPlayer):
            action = player.decide(game, game.playable_actions)
            assert action in game.playable_actions


# ---------------------------------------------------------------------------
# Deterministic for fixed seed and budget
# ---------------------------------------------------------------------------

def test_deterministic_choice():
    """Same game + same MCTS seed + same budget → same encoded action."""
    def run_once():
        game = Game(
            [MCTSPlayer(Color.RED, max_simulations=10, seed=99),
             RandomPlayer(Color.BLUE)],
            seed=7,
        )
        # Advance past initial placements to a more interesting state.
        for _ in range(12):
            if game.winning_color() is not None:
                break
            game.play_tick()
        if game.winning_color() is not None:
            return None
        player = game.state.current_player()
        if isinstance(player, MCTSPlayer):
            action = player.decide(game, game.playable_actions)
            return ActionCodec.encode(action)
        return None

    a = run_once()
    b = run_once()
    if a is not None:
        assert a == b


# ---------------------------------------------------------------------------
# Search does not mutate the original live game
# ---------------------------------------------------------------------------

def test_search_does_not_mutate_game():
    game = _make_game(seed=0, ticks=0)
    player = game.state.current_player()
    if not isinstance(player, MCTSPlayer):
        game.play_tick()
        player = game.state.current_player()

    if isinstance(player, MCTSPlayer):
        turns_before = game.state.num_turns
        actions_before = len(game.state.action_records)
        buildings_before = len(game.state.board.buildings)

        player.decide(game, game.playable_actions)

        assert game.state.num_turns == turns_before
        assert len(game.state.action_records) == actions_before
        assert len(game.state.board.buildings) == buildings_before


# ---------------------------------------------------------------------------
# Full tiny game completes
# ---------------------------------------------------------------------------

def test_full_game_completes():
    bot = MCTSPlayer(Color.RED, max_simulations=5, seed=0)
    baseline = RandomPlayer(Color.BLUE)
    game = Game([bot, baseline], seed=3)
    game.play()
    assert game.state.num_turns > 0
    assert bot._calls > 0


# ---------------------------------------------------------------------------
# Candidate filter is deterministic
# ---------------------------------------------------------------------------

def test_candidate_filter_deterministic():
    game = _make_game(seed=0, ticks=20)
    if game.winning_color() is not None:
        return

    ctx = DecisionContext(
        game, game.playable_actions, game.state.current_color()
    )
    cf = CandidateFilter(top_k_roads=3, top_k_trades=2)

    a = cf(ctx.public_state, ctx.encoded_actions)
    b = cf(ctx.public_state, ctx.encoded_actions)
    assert a == b


# ---------------------------------------------------------------------------
# Root child stats populated after search
# ---------------------------------------------------------------------------

def test_root_children_populated():
    game = _make_game(seed=0, ticks=0)

    mcts = MCTS(
        root_color=game.state.current_color(),
        max_simulations=10,
        seed=0,
        candidate_filter=CandidateFilter(),
    )
    best_ea, stats = mcts.search(game)

    assert stats["simulations"] > 0
    assert stats["max_depth"] >= 0
    assert isinstance(best_ea, EncodedAction)
    assert len(stats["root_children"]) > 0

    for key, child_stats in stats["root_children"].items():
        assert child_stats["visits"] >= 0
