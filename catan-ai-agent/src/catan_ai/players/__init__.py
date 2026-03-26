"""Custom player implementations owned by this repo."""

from catan_ai.players.debug_player import DebugPlayer
from catan_ai.players.heuristic_player import HeuristicBot

__all__ = [
    "DebugPlayer",
    "HeuristicBot",
    "MCTSPlayer",
    "BeliefMCTSPlayer",
    "NeuralMCTSPlayer",
]


def __getattr__(name: str):
    """Lazy import to avoid package-level circular imports."""
    if name == "MCTSPlayer":
        from catan_ai.players.mcts_player import MCTSPlayer

        return MCTSPlayer
    if name == "BeliefMCTSPlayer":
        from catan_ai.players.belief_mcts_player import BeliefMCTSPlayer

        return BeliefMCTSPlayer
    if name == "NeuralMCTSPlayer":
        from catan_ai.players.neural_mcts_player import NeuralMCTSPlayer

        return NeuralMCTSPlayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
