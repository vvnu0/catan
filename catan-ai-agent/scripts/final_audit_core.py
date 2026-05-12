"""Instrumented tournament runner for the final audit.

Provides an InstrumentedArena that hooks into game execution to collect
granular per-game metrics beyond what the base Arena tracks.
"""

from __future__ import annotations

import contextlib
import io
import logging
import random as _random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from catanatron import Color, Game
from catanatron.models.player import Player
from catanatron.state_functions import (
    get_visible_victory_points,
    player_num_resource_cards,
)

from catan_ai.adapters.catanatron_adapter import public_state_from_game
from catan_ai.belief.determinizer import Determinizer
from catan_ai.players.belief_mcts_player import BeliefMCTSConfig, BeliefMCTSPlayer
from catan_ai.players.decision_context import DecisionContext
from catan_ai.players.heuristic_player import HeuristicBot
from catan_ai.players.neural_mcts_player import (
    NeuralMCTS,
    NeuralMCTSConfig,
    NeuralMCTSPlayer,
)
from catan_ai.search.candidate_filter import CandidateFilter
from catan_ai.training.checkpoints import load_checkpoint

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]

# Resource costs for common Catan actions
_ACTION_COSTS = {
    "BUILD_SETTLEMENT": 4,  # wood + brick + sheep + wheat
    "BUILD_CITY": 5,        # 3 ore + 2 wheat
    "BUILD_ROAD": 2,        # wood + brick
    "BUY_DEVELOPMENT_CARD": 3,  # ore + sheep + wheat
}


def _pip_count(number: int) -> int:
    return 6 - abs(7 - number)


# ── Per-game record ─────────────────────────────────────────────────

@dataclass
class GameRecord:
    """Detailed metrics from a single game."""
    game_seed: int
    winner_color: str | None
    candidate_color: str
    candidate_won: bool
    candidate_vp: int
    opponent_vp: int
    total_turns: int
    candidate_move_ms: float
    opponent_move_ms: float = 0.0

    # Strategic metrics
    resources_spent: int = 0
    settlement_pips: list[int] = field(default_factory=list)
    city_pips: list[int] = field(default_factory=list)
    has_largest_army: bool = False
    has_longest_road: bool = False

    # VP progression: list of (turn_number, vp) snapshots
    vp_trace: list[tuple[int, int]] = field(default_factory=list)

    @property
    def economic_efficiency(self) -> float:
        """VP per 10 resources spent."""
        if self.resources_spent == 0:
            return 0.0
        return self.candidate_vp / (self.resources_spent / 10.0)

    @property
    def avg_settlement_pips(self) -> float:
        all_pips = self.settlement_pips + self.city_pips
        return sum(all_pips) / len(all_pips) if all_pips else 0.0


# ── NeuralFrequencyMCTSPlayer (from run_v3_combined.py) ─────────────

class NeuralFrequencyMCTSPlayer(Player):
    """NeuralMCTS over one determinized frequency-belief world."""

    def __init__(self, color, *, model, config: NeuralMCTSConfig,
                 belief_seed: int, is_bot: bool = True):
        super().__init__(color, is_bot=is_bot)
        self.model = model
        self.config = config
        self.belief_seed = belief_seed
        self._calls = 0
        self._total_search_ms = 0.0

    def decide(self, game, playable_actions):
        self._calls += 1
        if len(playable_actions) == 1:
            return playable_actions[0]

        root_ctx = DecisionContext(game, playable_actions, self.color)
        rng = _random.Random(self.belief_seed + game.state.num_turns * 997)
        world = Determinizer(
            acting_color=self.color,
            belief_mode="count_only",
            enable_devcard_sampling=False,
        ).sample_world(game, rng)
        if world is None:
            world = game

        t0 = time.perf_counter()
        engine = NeuralMCTS(
            root_color=self.color,
            model=self.model,
            cfg=self.config,
            candidate_filter=CandidateFilter(
                top_k_roads=self.config.top_k_roads,
                top_k_trades=self.config.top_k_trades,
                top_k_robber=self.config.top_k_robber,
            ),
        )
        best_ea, _stats = engine.search(world)
        self._total_search_ms += (time.perf_counter() - t0) * 1000
        return root_ctx.get_raw_action(best_ea)

    def reset_state(self):
        self._calls = 0
        self._total_search_ms = 0.0

    @property
    def avg_move_ms(self) -> float:
        return self._total_search_ms / self._calls if self._calls > 0 else 0.0


# ── Instrumented Arena ──────────────────────────────────────────────

PlayerFactory = Callable[[Color], Player]


def _extract_post_game_metrics(
    game: Game,
    cand_color: Color,
    opp_color: Color,
) -> dict[str, Any]:
    """Extract strategic metrics from a completed game state."""
    ps = public_state_from_game(game, cand_color)

    # Count resources spent by candidate (estimate from built structures)
    resources_spent = 0
    settlement_pips: list[int] = []
    city_pips: list[int] = []

    for b in ps.buildings:
        if b.color != cand_color.value:
            continue
        node_pips = sum(_pip_count(n) for _, n in ps.node_production.get(b.node_id, ()))
        if b.building_type == "CITY":
            city_pips.append(node_pips)
            resources_spent += _ACTION_COSTS["BUILD_CITY"]
            # City was also a settlement first
            resources_spent += _ACTION_COSTS["BUILD_SETTLEMENT"]
        else:
            settlement_pips.append(node_pips)
            resources_spent += _ACTION_COSTS["BUILD_SETTLEMENT"]

    # Count roads
    for r in ps.roads:
        if r.color == cand_color.value:
            resources_spent += _ACTION_COSTS["BUILD_ROAD"]

    # Subtract initial placements (2 settlements + 2 roads are free)
    resources_spent -= 2 * _ACTION_COSTS["BUILD_SETTLEMENT"]
    resources_spent -= 2 * _ACTION_COSTS["BUILD_ROAD"]
    resources_spent = max(0, resources_spent)

    # Dev cards played (estimate bought)
    cand_summary = None
    for s in ps.player_summaries:
        if s.color == cand_color.value:
            cand_summary = s
            break

    if cand_summary:
        dev_played = (cand_summary.played_knights +
                      cand_summary.played_year_of_plenty +
                      cand_summary.played_monopoly +
                      cand_summary.played_road_building)
        dev_held = cand_summary.num_dev_cards
        resources_spent += (dev_played + dev_held) * _ACTION_COSTS["BUY_DEVELOPMENT_CARD"]

    return {
        "resources_spent": resources_spent,
        "settlement_pips": settlement_pips,
        "city_pips": city_pips,
        "has_largest_army": cand_summary.has_largest_army if cand_summary else False,
        "has_longest_road": cand_summary.has_longest_road if cand_summary else False,
    }


class InstrumentedArena:
    """Arena that collects detailed per-game metrics."""

    def __init__(self, num_games: int = 25, base_seed: int = 7000,
                 swap_seats: bool = True):
        self.num_games = num_games
        self.base_seed = base_seed
        self.swap_seats = swap_seats

    def run_matchup(
        self,
        make_candidate: PlayerFactory,
        make_baseline: PlayerFactory,
        label: str,
    ) -> list[GameRecord]:
        """Play games and return detailed per-game records."""
        records: list[GameRecord] = []

        for i in range(self.num_games):
            seed = self.base_seed + i
            rec = self._play_one(make_candidate, make_baseline, seed,
                                 candidate_is_red=True)
            if rec:
                records.append(rec)

            if self.swap_seats:
                rec = self._play_one(make_candidate, make_baseline,
                                     seed + 10000, candidate_is_red=False)
                if rec:
                    records.append(rec)

            if (i + 1) % 10 == 0:
                wins = sum(1 for r in records if r.candidate_won)
                log.info("[%s] %d/%d games done, wins=%d",
                         label, len(records), (i + 1) * (2 if self.swap_seats else 1), wins)

        return records

    def _play_one(
        self,
        make_candidate: PlayerFactory,
        make_baseline: PlayerFactory,
        seed: int,
        candidate_is_red: bool,
    ) -> GameRecord | None:
        if candidate_is_red:
            candidate = make_candidate(Color.RED)
            baseline = make_baseline(Color.BLUE)
            players = [candidate, baseline]
            cand_color, opp_color = Color.RED, Color.BLUE
        else:
            baseline = make_baseline(Color.RED)
            candidate = make_candidate(Color.BLUE)
            players = [baseline, candidate]
            cand_color, opp_color = Color.BLUE, Color.RED

        game = Game(players, seed=seed)

        # Play with bank-error guard
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                winner = game.play()
        except (ValueError, Exception) as exc:
            if "bank" in str(exc).lower() or "resource" in str(exc).lower():
                log.warning("Bank inconsistency in game seed %d, skipping: %s",
                            seed, exc)
                return None
            raise

        cand_vp = get_visible_victory_points(game.state, cand_color)
        opp_vp = get_visible_victory_points(game.state, opp_color)
        cand_won = (winner == cand_color)
        move_ms = candidate.avg_move_ms if hasattr(candidate, "avg_move_ms") else 0.0
        opp_move_ms = baseline.avg_move_ms if hasattr(baseline, "avg_move_ms") else 0.0

        # Extract strategic metrics
        try:
            metrics = _extract_post_game_metrics(game, cand_color, opp_color)
        except Exception:
            metrics = {"resources_spent": 0, "settlement_pips": [],
                       "city_pips": [], "has_largest_army": False,
                       "has_longest_road": False}

        # Build VP trace (sample at key points from final state)
        vp_trace = [(game.state.num_turns, cand_vp)]

        return GameRecord(
            game_seed=seed,
            winner_color=winner.value if winner else None,
            candidate_color=cand_color.value,
            candidate_won=cand_won,
            candidate_vp=cand_vp,
            opponent_vp=opp_vp,
            total_turns=game.state.num_turns,
            candidate_move_ms=move_ms,
            opponent_move_ms=opp_move_ms,
            resources_spent=metrics["resources_spent"],
            settlement_pips=metrics["settlement_pips"],
            city_pips=metrics["city_pips"],
            has_largest_army=metrics["has_largest_army"],
            has_longest_road=metrics["has_longest_road"],
            vp_trace=vp_trace,
        )


# ── Player factories ────────────────────────────────────────────────

def make_player_factories(
    model, neural_cfg: NeuralMCTSConfig, search_seed: int,
) -> dict[str, PlayerFactory]:
    """Create factory functions for all 3 agent types."""

    def make_neural(color: Color) -> NeuralFrequencyMCTSPlayer:
        return NeuralFrequencyMCTSPlayer(
            color, model=model, config=neural_cfg, belief_seed=search_seed)

    def make_frequency(color: Color) -> BeliefMCTSPlayer:
        return BeliefMCTSPlayer(
            color,
            config=BeliefMCTSConfig(
                num_worlds=1, sims_per_world=30,
                belief_seed=search_seed, belief_mode="count_only",
                enable_devcard_sampling=False, max_depth=8,
                top_k_roads=3, top_k_trades=2, top_k_robber=4,
            ),
        )

    def make_heuristic(color: Color) -> HeuristicBot:
        return HeuristicBot(color)

    return {
        "neural": make_neural,
        "frequency": make_frequency,
        "heuristic": make_heuristic,
    }


# ── Aggregate stats helper ──────────────────────────────────────────

@dataclass
class AggregateStats:
    """Summary statistics for one agent matchup."""
    label: str
    games: int
    wins: int
    win_rate: float
    avg_vp: float
    avg_turns: float
    avg_move_ms: float
    avg_turns_to_win: float
    econ_efficiency: float
    spatial_mastery: float
    largest_army_rate: float
    longest_road_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


def aggregate(records: list[GameRecord], label: str) -> AggregateStats:
    """Compute aggregate stats from per-game records."""
    n = len(records)
    if n == 0:
        return AggregateStats(label=label, games=0, wins=0, win_rate=0,
                              avg_vp=0, avg_turns=0, avg_move_ms=0,
                              avg_turns_to_win=0, econ_efficiency=0,
                              spatial_mastery=0, largest_army_rate=0,
                              longest_road_rate=0)

    wins = sum(1 for r in records if r.candidate_won)
    won_records = [r for r in records if r.candidate_won]

    return AggregateStats(
        label=label,
        games=n,
        wins=wins,
        win_rate=wins / n,
        avg_vp=sum(r.candidate_vp for r in records) / n,
        avg_turns=sum(r.total_turns for r in records) / n,
        avg_move_ms=sum(r.candidate_move_ms for r in records) / n,
        avg_turns_to_win=(sum(r.total_turns for r in won_records) / len(won_records)
                          if won_records else 0.0),
        econ_efficiency=sum(r.economic_efficiency for r in records) / n,
        spatial_mastery=sum(r.avg_settlement_pips for r in records) / n,
        largest_army_rate=sum(1 for r in records if r.has_largest_army) / n,
        longest_road_rate=sum(1 for r in records if r.has_longest_road) / n,
    )
