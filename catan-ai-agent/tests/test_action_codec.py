"""Tests for ActionCodec: encoding, deterministic sorting, and decode_to_str."""

from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import Action, ActionType

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.public_state import EncodedAction


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def test_encode_simple_action():
    action = Action(Color.RED, ActionType.ROLL, None)
    ea = ActionCodec.encode(action)
    assert ea == EncodedAction(color="RED", action_type="ROLL", value=None)


def test_encode_action_with_int_value():
    action = Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 42)
    ea = ActionCodec.encode(action)
    assert ea.color == "BLUE"
    assert ea.action_type == "BUILD_SETTLEMENT"
    assert ea.value == "42"


def test_encode_action_with_tuple_value():
    action = Action(Color.RED, ActionType.BUILD_ROAD, (3, 7))
    ea = ActionCodec.encode(action)
    assert ea.value == "(3,7)"


def test_encode_action_with_nested_tuple():
    action = Action(
        Color.RED,
        ActionType.MOVE_ROBBER,
        ((1, -1, 0), Color.BLUE),
    )
    ea = ActionCodec.encode(action)
    assert ea.value == "((1,-1,0),BLUE)"


def test_encode_many():
    actions = [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.END_TURN, None),
    ]
    encoded = ActionCodec.encode_many(actions)
    assert len(encoded) == 2
    assert all(isinstance(e, EncodedAction) for e in encoded)


# ---------------------------------------------------------------------------
# Deterministic sorting
# ---------------------------------------------------------------------------

def test_sort_is_deterministic_across_calls():
    """Sorting the same set of encoded actions must produce identical order."""
    game = Game(
        [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
        seed=0,
    )
    encoded = ActionCodec.encode_many(game.playable_actions)

    sorted_a = ActionCodec.sorted_actions(encoded)
    sorted_b = ActionCodec.sorted_actions(encoded)

    assert sorted_a == sorted_b


def test_sort_is_deterministic_across_games():
    """Two games with the same seed should produce the same sorted actions."""
    def get_sorted_actions(seed):
        game = Game(
            [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
            seed=seed,
        )
        return ActionCodec.sorted_actions(
            ActionCodec.encode_many(game.playable_actions)
        )

    assert get_sorted_actions(123) == get_sorted_actions(123)


def test_sort_order_is_stable():
    """Verify that sort order groups by color, then action_type, then value."""
    actions = [
        EncodedAction("RED", "END_TURN", None),
        EncodedAction("RED", "BUILD_ROAD", "(3,7)"),
        EncodedAction("BLUE", "ROLL", None),
        EncodedAction("RED", "BUILD_ROAD", "(1,2)"),
    ]
    result = ActionCodec.sorted_actions(actions)
    keys = [ActionCodec.sort_key(e) for e in result]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Decode to string
# ---------------------------------------------------------------------------

def test_decode_to_str_no_value():
    ea = EncodedAction("RED", "ROLL", None)
    assert ActionCodec.decode_to_str(ea) == "RED:ROLL"


def test_decode_to_str_with_value():
    ea = EncodedAction("RED", "BUILD_SETTLEMENT", "42")
    assert ActionCodec.decode_to_str(ea) == "RED:BUILD_SETTLEMENT(42)"


# ---------------------------------------------------------------------------
# EncodedAction is hashable (needed for sets, dict keys, dedup)
# ---------------------------------------------------------------------------

def test_encoded_action_is_hashable():
    ea = EncodedAction("RED", "ROLL", None)
    s = {ea, ea}
    assert len(s) == 1
