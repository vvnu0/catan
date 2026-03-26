"""BeliefMCTSPlayer — multi-world determinized search.

Wraps the existing single-world MCTS v1 engine: for each decision, it
samples several determinized hidden worlds consistent with public
information, runs MCTS on each, and aggregates root-action statistics
to choose the final action.

This is a practical determinization wrapper (2-player only for now),
not a full information-set MCTS redesign.

WARNING: the inner MCTS still uses Game.copy() within each world.
The belief layer controls *which* hidden world each inner MCTS sees,
but does not prevent the inner MCTS from reading that world's hidden
state during its search.  This reduces hidden-information leakage
(averaging over possible worlds) but does not eliminate it.
"""

from __future__ import annotations

import logging
import random as _random
import time
from collections import defaultdict
from dataclasses import dataclass

from catanatron.models.player import Player

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.belief.determinizer import Determinizer
from catan_ai.players.decision_context import DecisionContext
from catan_ai.search.candidate_filter import CandidateFilter
from catan_ai.search.mcts import MCTS

log = logging.getLogger(__name__)


@dataclass
class BeliefMCTSConfig:
    """All tuneable parameters for BeliefMCTSPlayer."""

    # --- Belief / determinization ---
    num_worlds: int = 8
    sims_per_world: int = 15
    max_invalid_samples: int = 20
    belief_seed: int | None = 0
    belief_mode: str = "conservation"
    enable_devcard_sampling: bool = False
    move_time_ms: int | None = None

    # --- Inner MCTS ---
    max_depth: int = 10
    exploration_c: float = 1.41
    top_k_roads: int = 3
    top_k_trades: int = 2
    top_k_robber: int = 4


class BeliefMCTSPlayer(Player):
    """MCTS player with belief-aware multi-world determinization.

    For each ``decide()`` call:
      1. Build root DecisionContext (encoded ↔ raw action mapping).
      2. Extract public evidence for belief models.
      3. Sample ``num_worlds`` determinized game copies.
      4. Run single-world MCTS v1 on each with ``sims_per_world`` budget.
      5. Aggregate root child visit counts across worlds.
      6. Choose the action with the highest total visits (deterministic
         tie-break via EncodedAction ordering).
      7. Map back to the exact raw playable action via DecisionContext.

    Args:
        color: Player colour.
        config: Tuning parameters (see BeliefMCTSConfig).
        is_bot: Whether this is a bot player.
    """

    def __init__(self, color, *, config: BeliefMCTSConfig | None = None, is_bot: bool = True):
        super().__init__(color, is_bot=is_bot)
        self.config = config or BeliefMCTSConfig()
        self._calls = 0
        self._total_search_ms = 0.0
        self._worlds_attempted_total = 0
        self._worlds_used_total = 0
        self._invalid_samples_total = 0
        self._fallback_count = 0

    # ------------------------------------------------------------------
    # Catanatron Player interface
    # ------------------------------------------------------------------
    def decide(self, game, playable_actions):
        self._calls += 1
        cfg = self.config

        if len(playable_actions) == 1:
            return playable_actions[0]

        root_ctx = DecisionContext(game, playable_actions, self.color)

        # Reverse map: decode_to_str → EncodedAction for aggregation.
        str_to_ea: dict[str, object] = {
            ActionCodec.decode_to_str(ea): ea for ea in root_ctx.encoded_actions
        }

        t0 = time.perf_counter()

        # Seed for this decision's sampling (deterministic across calls
        # at the same game state if belief_seed is set).
        if cfg.belief_seed is not None:
            decision_seed = cfg.belief_seed + game.state.num_turns * 997
        else:
            decision_seed = None

        rng = _random.Random(decision_seed)

        determinizer = Determinizer(
            acting_color=self.color,
            belief_mode=cfg.belief_mode,
            enable_devcard_sampling=cfg.enable_devcard_sampling,
            max_invalid_samples=cfg.max_invalid_samples,
        )

        candidate_filter = CandidateFilter(
            top_k_roads=cfg.top_k_roads,
            top_k_trades=cfg.top_k_trades,
            top_k_robber=cfg.top_k_robber,
        )

        # --- Sample worlds and run inner MCTS on each ---
        agg_visits: dict[str, int] = defaultdict(int)
        agg_value_sum: dict[str, float] = defaultdict(float)
        worlds_used = 0
        invalid_count = 0
        worlds_attempted = 0

        for world_idx in range(cfg.num_worlds):
            remaining_ms = None
            if cfg.move_time_ms is not None:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if elapsed_ms >= cfg.move_time_ms:
                    break
                remaining_ms = max(1, int(cfg.move_time_ms - elapsed_ms))

            worlds_attempted += 1
            world = determinizer.sample_world(game, rng)
            if world is None:
                invalid_count += 1
                continue

            # Derive a per-world MCTS seed for determinism.
            world_seed = (
                (decision_seed * 31 + world_idx) if decision_seed is not None else None
            )

            inner_mcts = MCTS(
                root_color=self.color,
                max_simulations=cfg.sims_per_world,
                max_depth=cfg.max_depth,
                exploration_c=cfg.exploration_c,
                candidate_filter=candidate_filter,
                move_time_ms=remaining_ms,
                seed=world_seed,
            )
            _best_ea, stats = inner_mcts.search(world)

            for action_str, child_stats in stats["root_children"].items():
                agg_visits[action_str] += child_stats["visits"]
                agg_value_sum[action_str] += (
                    child_stats["avg_value"] * child_stats["visits"]
                )

            worlds_used += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_search_ms += elapsed_ms
        self._worlds_attempted_total += worlds_attempted
        self._worlds_used_total += worlds_used
        self._invalid_samples_total += invalid_count

        # --- Choose the root action with highest aggregated visits ---
        best_ea, used_fallback = self._pick_best(agg_visits, str_to_ea, root_ctx)
        if used_fallback:
            self._fallback_count += 1

        if log.isEnabledFor(logging.INFO):
            log.info(
                "BeliefMCTS %s call #%d — worlds: attempted=%d used=%d invalid=%d fallback=%s, %.0f ms",
                self.color.value,
                self._calls,
                worlds_attempted,
                worlds_used,
                invalid_count,
                used_fallback,
                elapsed_ms,
            )
        if log.isEnabledFor(logging.DEBUG):
            top = sorted(agg_visits.items(), key=lambda kv: -kv[1])[:5]
            for action_str, visits in top:
                avg = (
                    agg_value_sum[action_str] / visits if visits > 0 else 0.0
                )
                log.debug("  %s  visits=%d  avg=%.4f", action_str, visits, avg)
            log.debug(
                "  → chose %s",
                ActionCodec.decode_to_str(best_ea) if best_ea else "None",
            )

        return root_ctx.get_raw_action(best_ea)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pick_best(self, agg_visits, str_to_ea, root_ctx):
        """Select the encoded action with the most aggregated visits."""
        if not agg_visits:
            return root_ctx.encoded_actions[0], True

        # Sort by (-visits, sort_key) for deterministic tie-breaking.
        candidates = []
        for action_str, visits in agg_visits.items():
            ea = str_to_ea.get(action_str)
            if ea is None:
                continue
            candidates.append((visits, ea))

        if not candidates:
            return root_ctx.encoded_actions[0], True

        candidates.sort(key=lambda t: (-t[0], ActionCodec.sort_key(t[1])))
        return candidates[0][1], False

    def reset_state(self):
        self._calls = 0
        self._total_search_ms = 0.0
        self._worlds_attempted_total = 0
        self._worlds_used_total = 0
        self._invalid_samples_total = 0
        self._fallback_count = 0

    @property
    def avg_move_ms(self) -> float:
        return self._total_search_ms / self._calls if self._calls > 0 else 0.0
