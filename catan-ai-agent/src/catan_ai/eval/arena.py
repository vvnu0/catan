"""Arena — round-robin match runner for checkpoint comparison.

Runs batches of games between two player factories, collects aggregate
statistics, and optionally swaps seats to reduce first-player bias.
"""

from __future__ import annotations

import contextlib
import io
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

from catanatron import Color, Game
from catanatron.models.player import Player
from catanatron.state_functions import get_visible_victory_points

log = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Aggregate statistics from one batch of games."""

    label: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    turn_counts: list[int] = field(default_factory=list)
    move_times: list[float] = field(default_factory=list)
    final_vps: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games > 0 else 0.0

    @property
    def avg_turns(self) -> float:
        return statistics.mean(self.turn_counts) if self.turn_counts else 0.0

    @property
    def avg_move_ms(self) -> float:
        return statistics.mean(self.move_times) if self.move_times else 0.0

    @property
    def avg_final_vp(self) -> float:
        return statistics.mean(self.final_vps) if self.final_vps else 0.0

    def summary(self) -> str:
        return (
            f"{self.label}: {self.wins}W / {self.losses}L / {self.draws}D "
            f"(win rate {self.win_rate:.0%}, "
            f"avg VP {self.avg_final_vp:.1f}, "
            f"avg turns {self.avg_turns:.1f}, "
            f"avg move {self.avg_move_ms:.1f} ms)"
        )


PlayerFactory = Callable[[Color], Player]


class Arena:
    """Run fixed-seed matches between two player factories.

    Args:
        num_games: Games per pairing (per seat if swapping).
        base_seed: Starting seed for game RNG.
        swap_seats: If True, run each seed twice with swapped colours.
    """

    def __init__(
        self,
        num_games: int = 20,
        base_seed: int = 3000,
        swap_seats: bool = False,
    ):
        self.num_games = num_games
        self.base_seed = base_seed
        self.swap_seats = swap_seats

    def compare(
        self,
        make_candidate: PlayerFactory,
        make_baseline: PlayerFactory,
        label: str = "candidate vs baseline",
    ) -> MatchResult:
        """Run a batch of games and return aggregate stats.

        The *candidate* is always evaluated — wins/losses are from the
        candidate's perspective.
        """
        result = MatchResult(label=label)

        for i in range(self.num_games):
            seed = self.base_seed + i
            self._play_one(make_candidate, make_baseline, seed, result, candidate_is_red=True)
            if self.swap_seats:
                self._play_one(make_candidate, make_baseline, seed + 10000, result, candidate_is_red=False)

        log.info(result.summary())
        return result

    def _play_one(
        self,
        make_candidate: PlayerFactory,
        make_baseline: PlayerFactory,
        seed: int,
        result: MatchResult,
        candidate_is_red: bool,
    ) -> None:
        if candidate_is_red:
            candidate = make_candidate(Color.RED)
            baseline = make_baseline(Color.BLUE)
            players = [candidate, baseline]
            cand_color = Color.RED
        else:
            baseline = make_baseline(Color.RED)
            candidate = make_candidate(Color.BLUE)
            players = [baseline, candidate]
            cand_color = Color.BLUE

        game = Game(players, seed=seed)
        with contextlib.redirect_stdout(io.StringIO()):
            winner = game.play()

        result.games += 1
        if winner == cand_color:
            result.wins += 1
        elif winner is None:
            result.draws += 1
        else:
            result.losses += 1

        result.turn_counts.append(game.state.num_turns)
        result.final_vps.append(get_visible_victory_points(game.state, cand_color))

        if hasattr(candidate, "avg_move_ms"):
            result.move_times.append(candidate.avg_move_ms)
