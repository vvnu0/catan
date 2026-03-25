"""Tests for DebugPlayer integration with Catanatron."""

from catanatron import Color, Game, RandomPlayer

from catan_ai.players import DebugPlayer
from catan_ai.players.debug_player import _action_sort_key


def test_instantiation():
    player = DebugPlayer(Color.RED)
    assert player.color == Color.RED
    assert player.is_bot is True
    assert player._calls == 0


def test_returns_legal_action():
    """DebugPlayer.decide() must return one of the playable_actions."""
    players = [DebugPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=0)

    actions = game.playable_actions
    chosen = players[0].decide(game, actions)
    assert chosen in actions


def test_returns_first_sorted_action():
    """DebugPlayer must pick the first action in sorted order, not raw order."""
    players = [DebugPlayer(Color.RED), DebugPlayer(Color.BLUE)]
    game = Game(players, seed=0)

    actions = game.playable_actions
    expected = sorted(actions, key=_action_sort_key)[0]
    chosen = players[0].decide(game, actions)
    assert chosen == expected


def test_decide_increments_call_counter():
    players = [DebugPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=0)

    players[0].decide(game, game.playable_actions)
    players[0].decide(game, game.playable_actions)
    assert players[0]._calls == 2


def test_reset_state():
    player = DebugPlayer(Color.RED)
    player._calls = 5
    player.reset_state()
    assert player._calls == 0


def test_debug_match_runs_to_completion():
    """A full game with DebugPlayer must finish without crashing."""
    debug = DebugPlayer(Color.RED)
    baseline = RandomPlayer(Color.BLUE)
    game = Game([debug, baseline], seed=1)

    game.play()

    assert game.state.num_turns > 0
    assert debug._calls > 0


def test_debug_vs_debug_reproducible():
    """Two DebugPlayer-vs-DebugPlayer games with the same seed must
    produce the same turn count, confirming sorted selection removes
    any ordering sensitivity on our side."""
    def play_once(seed):
        players = [DebugPlayer(Color.RED), DebugPlayer(Color.BLUE)]
        game = Game(players, seed=seed)
        game.play()
        return game.state.num_turns

    assert play_once(99) == play_once(99)
