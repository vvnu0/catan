"""Verify that catanatron and catan_ai can be imported from this repo."""

import importlib


def test_catanatron_top_level():
    mod = importlib.import_module("catanatron")
    assert hasattr(mod, "Game")
    assert hasattr(mod, "RandomPlayer")
    assert hasattr(mod, "Color")


def test_catanatron_models():
    from catanatron.models.enums import ActionType, RESOURCES, SETTLEMENT, CITY, ROAD

    assert len(RESOURCES) == 5
    assert hasattr(ActionType, "ROLL")
    assert SETTLEMENT == "SETTLEMENT"
    assert CITY == "CITY"
    assert ROAD == "ROAD"


def test_catanatron_state():
    from catanatron.state import State

    assert callable(State)


def test_catanatron_state_functions():
    from catanatron.state_functions import (
        get_actual_victory_points,
        player_key,
    )

    assert callable(get_actual_victory_points)
    assert callable(player_key)


def test_catanatron_game_creation():
    from catanatron import Color, Game, RandomPlayer

    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=0)
    assert game.playable_actions is not None
    assert len(game.playable_actions) > 0


def test_catan_ai_package():
    import catan_ai

    assert hasattr(catan_ai, "__version__")


def test_catan_ai_utils():
    from catan_ai.utils.logging import get_logger
    from catan_ai.utils.seeding import make_seed

    logger = get_logger("test")
    assert logger is not None

    seed = make_seed(42)
    assert seed == 42

    random_seed = make_seed()
    assert isinstance(random_seed, int)
