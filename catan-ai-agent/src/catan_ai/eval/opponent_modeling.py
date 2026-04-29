"""Player factories for opponent-modeling ablations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from catanatron.models.player import Color, Player

from catan_ai.players.belief_mcts_player import BeliefMCTSConfig, BeliefMCTSPlayer
from catan_ai.players.mcts_player import MCTSPlayer


class OpponentModelMode(str, Enum):
    """Opponent-modeling variants used in proposal-aligned ablations."""

    NONE = "none"
    FREQUENCY = "frequency"
    PARTICLE = "particle"


@dataclass(frozen=True)
class OpponentModelEvalConfig:
    """Compute-matched search settings for all ablation modes."""

    total_simulations: int = 24
    max_depth: int = 8
    exploration_c: float = 1.41
    top_k_roads: int = 3
    top_k_trades: int = 2
    top_k_robber: int = 4
    search_seed: int = 2026
    particle_worlds: int = 4
    max_invalid_samples: int = 20
    enable_particle_devcards: bool = True


def make_opponent_model_player(
    mode: OpponentModelMode | str,
    color: Color,
    cfg: OpponentModelEvalConfig,
) -> Player:
    """Instantiate a player for one ablation mode."""
    mode = OpponentModelMode(mode)
    if mode is OpponentModelMode.NONE:
        return MCTSPlayer(
            color,
            max_simulations=cfg.total_simulations,
            max_depth=cfg.max_depth,
            exploration_c=cfg.exploration_c,
            top_k_roads=cfg.top_k_roads,
            top_k_trades=cfg.top_k_trades,
            top_k_robber=cfg.top_k_robber,
            seed=cfg.search_seed,
        )

    if mode is OpponentModelMode.FREQUENCY:
        # Single count-only determinization: simple frequency/resource-count belief,
        # with the same total simulation budget as plain MCTS.
        return BeliefMCTSPlayer(
            color,
            config=BeliefMCTSConfig(
                num_worlds=1,
                sims_per_world=cfg.total_simulations,
                max_invalid_samples=cfg.max_invalid_samples,
                belief_seed=cfg.search_seed,
                belief_mode="count_only",
                enable_devcard_sampling=False,
                max_depth=cfg.max_depth,
                exploration_c=cfg.exploration_c,
                top_k_roads=cfg.top_k_roads,
                top_k_trades=cfg.top_k_trades,
                top_k_robber=cfg.top_k_robber,
            ),
        )

    sims_per_world = max(1, cfg.total_simulations // max(1, cfg.particle_worlds))
    return BeliefMCTSPlayer(
        color,
        config=BeliefMCTSConfig(
            num_worlds=cfg.particle_worlds,
            sims_per_world=sims_per_world,
            max_invalid_samples=cfg.max_invalid_samples,
            belief_seed=cfg.search_seed,
            belief_mode="conservation",
            enable_devcard_sampling=cfg.enable_particle_devcards,
            max_depth=cfg.max_depth,
            exploration_c=cfg.exploration_c,
            top_k_roads=cfg.top_k_roads,
            top_k_trades=cfg.top_k_trades,
            top_k_robber=cfg.top_k_robber,
        ),
    )
