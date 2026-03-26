"""MCTSPlayer — Catanatron Player that uses MCTS v1 for decisions.

WARNING: This is a single-world search scaffold.  Game.copy() carries the
full hidden world (opponent hands, deck order).  A future version will
replace this with belief sampling or information-set MCTS to be
hidden-information safe.

For mandatory / single-option actions (ROLL, DISCARD with one choice), the
player short-circuits without running a full search.
"""

from __future__ import annotations

import logging
import time

from catanatron.models.player import Player

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.players.decision_context import DecisionContext
from catan_ai.search.candidate_filter import CandidateFilter
from catan_ai.search.mcts import MCTS

log = logging.getLogger(__name__)


class MCTSPlayer(Player):
    """MCTS v1 strategy bot (2-player, single-world scaffold).

    Args:
        color: Player colour.
        max_simulations: MCTS iteration budget per decision.
        max_depth: Maximum search depth.
        exploration_c: UCT exploration constant.
        top_k_roads: Candidate filter — max road actions to keep.
        top_k_trades: Candidate filter — max trade actions to keep.
        top_k_robber: Candidate filter — max robber actions to keep.
        move_time_ms: Optional soft wall-clock budget (ms).
        seed: Seed for deterministic search.
    """

    def __init__(
        self,
        color,
        *,
        max_simulations: int = 50,
        max_depth: int = 10,
        exploration_c: float = 1.41,
        top_k_roads: int = 3,
        top_k_trades: int = 2,
        top_k_robber: int = 4,
        move_time_ms: int | None = None,
        seed: int | None = None,
        is_bot: bool = True,
    ):
        super().__init__(color, is_bot=is_bot)
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.exploration_c = exploration_c
        self.top_k_roads = top_k_roads
        self.top_k_trades = top_k_trades
        self.top_k_robber = top_k_robber
        self.move_time_ms = move_time_ms
        self.seed = seed

        self._calls = 0
        self._total_search_ms = 0.0

    # ------------------------------------------------------------------
    # Catanatron Player interface
    # ------------------------------------------------------------------
    def decide(self, game, playable_actions):
        self._calls += 1

        # Short-circuit for forced single-option actions.
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Build root DecisionContext (maps encoded ↔ raw actions at root).
        root_ctx = DecisionContext(game, playable_actions, self.color)

        t0 = time.perf_counter()

        mcts = MCTS(
            root_color=self.color,
            max_simulations=self.max_simulations,
            max_depth=self.max_depth,
            exploration_c=self.exploration_c,
            candidate_filter=CandidateFilter(
                top_k_roads=self.top_k_roads,
                top_k_trades=self.top_k_trades,
                top_k_robber=self.top_k_robber,
            ),
            move_time_ms=self.move_time_ms,
            seed=self.seed,
        )
        best_ea, stats = mcts.search(game)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_search_ms += elapsed_ms

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "MCTSPlayer %s call #%d — %s (%.0f ms, %d sims, depth %d)",
                self.color.value,
                self._calls,
                ActionCodec.decode_to_str(best_ea),
                elapsed_ms,
                stats["simulations"],
                stats["max_depth"],
            )

        return root_ctx.get_raw_action(best_ea)

    def reset_state(self):
        self._calls = 0
        self._total_search_ms = 0.0

    @property
    def avg_move_ms(self) -> float:
        return self._total_search_ms / self._calls if self._calls > 0 else 0.0
