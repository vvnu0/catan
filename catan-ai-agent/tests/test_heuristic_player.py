"""Tests for HeuristicBot and DecisionContext."""

from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import ActionType

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.public_state import EncodedAction
from catan_ai.players import HeuristicBot
from catan_ai.players.decision_context import DecisionContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game(seed=0, ticks=0):
    players = [HeuristicBot(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=seed)
    for _ in range(ticks):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_can_be_instantiated():
    bot = HeuristicBot(Color.RED)
    assert bot.color == Color.RED
    assert bot.is_bot is True
    assert bot._calls == 0


# ---------------------------------------------------------------------------
# Always returns a legal action
# ---------------------------------------------------------------------------

def test_returns_legal_action():
    """decide() must return one of the raw playable_actions."""
    game = _make_game(seed=0, ticks=0)
    player = game.state.current_player()
    if not isinstance(player, HeuristicBot):
        # The other player is up — step once to get our player's turn.
        game.play_tick()
        player = game.state.current_player()

    if isinstance(player, HeuristicBot):
        action = player.decide(game, game.playable_actions)
        assert action in game.playable_actions


def test_returns_legal_action_mid_game():
    """Check legality after the game has progressed."""
    for seed in range(5):
        game = _make_game(seed=seed, ticks=30)
        if game.winning_color() is not None:
            continue
        player = game.state.current_player()
        if isinstance(player, HeuristicBot):
            action = player.decide(game, game.playable_actions)
            assert action in game.playable_actions


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_for_fixed_seed():
    """Two games with the same seed must produce the same first decision."""
    def first_decision(seed):
        game = _make_game(seed=seed)
        player = game.state.current_player()
        if isinstance(player, HeuristicBot):
            return ActionCodec.encode(player.decide(game, game.playable_actions))
        game.play_tick()
        player = game.state.current_player()
        if isinstance(player, HeuristicBot):
            return ActionCodec.encode(player.decide(game, game.playable_actions))
        return None

    a = first_decision(42)
    b = first_decision(42)
    assert a is not None
    assert a == b


# ---------------------------------------------------------------------------
# Prefers build over END_TURN
# ---------------------------------------------------------------------------

def _find_state_with_build_and_end_turn():
    """Step through games until we find a state where the HeuristicBot has
    both a BUILD action and END_TURN available."""
    for seed in range(20):
        game = _make_game(seed=seed, ticks=0)
        for _ in range(200):
            if game.winning_color() is not None:
                break
            player = game.state.current_player()
            actions = game.playable_actions
            types = {a.action_type for a in actions}
            if (
                isinstance(player, HeuristicBot)
                and ActionType.END_TURN in types
                and types & {
                    ActionType.BUILD_SETTLEMENT,
                    ActionType.BUILD_CITY,
                    ActionType.BUILD_ROAD,
                    ActionType.BUY_DEVELOPMENT_CARD,
                }
            ):
                return game, player, actions
            game.play_tick()
    return None, None, None


def test_prefers_build_over_end_turn():
    game, player, actions = _find_state_with_build_and_end_turn()
    if game is None:
        # Could not find a suitable state — skip rather than fail.
        return

    chosen = player.decide(game, actions)
    assert chosen.action_type != ActionType.END_TURN, (
        "HeuristicBot chose END_TURN when a build action was available"
    )


# ---------------------------------------------------------------------------
# DecisionContext mapping
# ---------------------------------------------------------------------------

def test_decision_context_mapping():
    """EncodedAction → raw action round-trip must work."""
    game = _make_game(seed=0)
    acting = game.state.current_color()
    ctx = DecisionContext(game, game.playable_actions, acting)

    assert len(ctx.encoded_actions) == len(game.playable_actions)

    for ea in ctx.encoded_actions:
        raw = ctx.get_raw_action(ea)
        assert raw in game.playable_actions
        # Re-encoding the raw action should give back the same EncodedAction.
        assert ActionCodec.encode(raw) == ea


# ---------------------------------------------------------------------------
# Full game runs to completion
# ---------------------------------------------------------------------------

def test_full_game_completes():
    """A complete game with HeuristicBot must finish without crashing."""
    bot = HeuristicBot(Color.RED)
    baseline = RandomPlayer(Color.BLUE)
    game = Game([bot, baseline], seed=7)
    game.play()
    assert game.state.num_turns > 0
    assert bot._calls > 0
