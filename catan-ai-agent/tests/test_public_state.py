"""Tests for the PublicState adapter layer.

Verifies that:
  - PublicState can be built from a real game
  - hidden opponent resource/dev-card identities never appear
  - the structure contains only simple serialisable types
"""

import dataclasses

from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES

from catan_ai.adapters import public_state_from_game
from catan_ai.adapters.public_state import (
    BuildingSummary,
    EncodedAction,
    PublicPlayerSummary,
    PublicState,
    RoadSummary,
    TileSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game(seed=0, ticks=40):
    """Create a 2-player game advanced by *ticks* steps."""
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=seed)
    for _ in range(ticks):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


def _acting_color(game):
    return game.state.current_color()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_public_state_can_be_built():
    game = _make_game()
    ps = public_state_from_game(game, _acting_color(game))
    assert isinstance(ps, PublicState)


def test_public_state_has_all_fields():
    game = _make_game()
    ps = public_state_from_game(game, _acting_color(game))
    field_names = {f.name for f in dataclasses.fields(PublicState)}
    expected = {
        "acting_color", "turn_number", "vps_to_win",
        "tiles", "robber_coordinate",
        "buildings", "roads",
        "player_summaries",
        "bank_resources", "bank_dev_cards_remaining",
        "own_resources", "own_dev_cards",
        "legal_actions",
        "is_initial_build_phase", "is_discarding",
        "is_moving_robber", "is_road_building", "free_roads_available",
        "node_production", "port_nodes",
    }
    assert expected == field_names


# ---------------------------------------------------------------------------
# Hidden information must NOT appear
# ---------------------------------------------------------------------------

def _collect_all_values(obj, depth=0):
    """Recursively yield every leaf value reachable from *obj*."""
    if depth > 20:
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _collect_all_values(getattr(obj, f.name), depth + 1)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _collect_all_values(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _collect_all_values(item, depth + 1)
    else:
        yield obj


def test_no_opponent_resource_identities_in_public_state():
    """PublicState must not contain per-resource counts for opponents."""
    game = _make_game()
    acting = _acting_color(game)
    ps = public_state_from_game(game, acting)

    for summary in ps.player_summaries:
        if summary.color == acting.value:
            continue
        # The summary should only have a *count* of resources, not breakdown.
        assert hasattr(summary, "num_resource_cards")
        assert not hasattr(summary, "resources")
        assert not hasattr(summary, "own_resources")

    # Scan all string leaf values: none should be a player_state key
    # like "P0_WOOD_IN_HAND" or similar.
    all_vals = list(_collect_all_values(ps))
    string_vals = [v for v in all_vals if isinstance(v, str)]
    for s in string_vals:
        assert "_IN_HAND" not in s, f"Leaked hidden key: {s}"


def test_no_opponent_dev_card_identities_in_public_state():
    """PublicState must not reveal which dev cards opponents hold."""
    game = _make_game()
    acting = _acting_color(game)
    ps = public_state_from_game(game, acting)

    for summary in ps.player_summaries:
        if summary.color == acting.value:
            continue
        assert hasattr(summary, "num_dev_cards")
        assert not hasattr(summary, "dev_cards")
        assert not hasattr(summary, "own_dev_cards")


def test_no_dev_deck_order_in_public_state():
    """The development-card deck ordering must never be exposed."""
    game = _make_game()
    ps = public_state_from_game(game, _acting_color(game))

    all_vals = list(_collect_all_values(ps))
    # The dev deck is a list of strings like "KNIGHT"; as a sequence it
    # should not appear.  We check that no field contains an ordered list of
    # dev-card type names.
    for v in all_vals:
        if isinstance(v, (list, tuple)) and len(v) > 1:
            # Should never be a raw dev deck slice.
            if all(item in DEVELOPMENT_CARDS for item in v):
                raise AssertionError(f"Possible dev-deck leak: {v[:5]}...")


def test_no_actual_vp_for_opponents():
    """ACTUAL_VICTORY_POINTS (which includes hidden VP cards) must not
    appear for opponents — only visible VP is allowed."""
    game = _make_game()
    acting = _acting_color(game)
    ps = public_state_from_game(game, acting)

    all_vals = list(_collect_all_values(ps))
    string_vals = [v for v in all_vals if isinstance(v, str)]
    for s in string_vals:
        assert "ACTUAL_VICTORY" not in s, f"Leaked hidden VP key: {s}"


# ---------------------------------------------------------------------------
# Serialisability: only simple types or our own dataclasses
# ---------------------------------------------------------------------------

_ALLOWED_LEAF_TYPES = (int, float, str, bool, type(None))

_ALLOWED_DATACLASSES = (
    PublicState,
    PublicPlayerSummary,
    TileSummary,
    BuildingSummary,
    RoadSummary,
    EncodedAction,
)


def _check_serialisable(obj, path="root", depth=0):
    """Walk *obj* and assert every node is a permitted type."""
    if depth > 30:
        return
    if isinstance(obj, _ALLOWED_LEAF_TYPES):
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        assert type(obj) in _ALLOWED_DATACLASSES, (
            f"Unexpected dataclass at {path}: {type(obj)}"
        )
        for f in dataclasses.fields(obj):
            _check_serialisable(
                getattr(obj, f.name), f"{path}.{f.name}", depth + 1
            )
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_serialisable(k, f"{path}[key]", depth + 1)
            _check_serialisable(v, f"{path}[{k!r}]", depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _check_serialisable(item, f"{path}[{i}]", depth + 1)
        return
    raise AssertionError(
        f"Non-serialisable type at {path}: {type(obj)} = {obj!r}"
    )


def test_public_state_is_serialisable():
    game = _make_game()
    ps = public_state_from_game(game, _acting_color(game))
    _check_serialisable(ps)


# ---------------------------------------------------------------------------
# Smoke: acting player's own hand is populated
# ---------------------------------------------------------------------------

def test_own_hand_is_populated():
    game = _make_game()
    acting = _acting_color(game)
    ps = public_state_from_game(game, acting)

    assert set(ps.own_resources.keys()) == set(RESOURCES)
    assert set(ps.own_dev_cards.keys()) == set(DEVELOPMENT_CARDS)
    assert all(isinstance(v, int) for v in ps.own_resources.values())
    assert all(isinstance(v, int) for v in ps.own_dev_cards.values())


def test_legal_actions_are_encoded():
    game = _make_game(ticks=0)
    ps = public_state_from_game(game, _acting_color(game))
    assert len(ps.legal_actions) > 0
    for ea in ps.legal_actions:
        assert isinstance(ea, EncodedAction)
        assert isinstance(ea.action_type, str)
