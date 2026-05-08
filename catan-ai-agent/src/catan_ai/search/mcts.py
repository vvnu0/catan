"""MCTS v1 — UCT search with heuristic leaf evaluation.

Single-world scaffold (2-player only).  Game.copy() carries the full
hidden world; this will be replaced with belief sampling in a future
version.  Clearly labelled as temporary.

Algorithm:
  1. Selection  — walk the tree via UCT, with perspective flip for the
                  opponent in a 2-player setting.
  2. Expansion  — expand one candidate action per simulation.
  3. Evaluation — heuristic leaf value (no random rollout).
  4. Backup     — propagate value (always from root player's perspective).
"""

from __future__ import annotations

import logging
import math
import random
import time

from catanatron.game import Game
from catanatron.models.player import Color

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.public_state import EncodedAction
from catan_ai.search.candidate_filter import CandidateFilter
from catan_ai.search.leaf_evaluator import evaluate_leaf
from catan_ai.search.tree_node import TreeNode

log = logging.getLogger(__name__)

_DEFAULT_C = 1.41  # exploration constant ≈ √2


class MCTS:
    """Monte-Carlo Tree Search (v1 — single-world, heuristic eval).

    Args:
        root_color: The colour we are optimising for.
        max_simulations: Hard budget on number of tree iterations.
        max_depth: Maximum tree depth before forced leaf evaluation.
        exploration_c: UCT exploration constant.
        candidate_filter: Action filter to limit branching.
        move_time_ms: Optional soft wall-clock budget (ms).  Search stops
            early if exceeded, even if max_simulations is not reached.
        seed: Seed for deterministic search.  If None, uses current random
            state (non-deterministic across calls).
    """

    def __init__(
        self,
        root_color: Color,
        *,
        max_simulations: int = 50,
        max_depth: int = 12,
        exploration_c: float = _DEFAULT_C,
        candidate_filter: CandidateFilter | None = None,
        move_time_ms: int | None = None,
        seed: int | None = None,
    ):
        self.root_color = root_color
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.c = exploration_c
        self.filter = candidate_filter or CandidateFilter()
        self.move_time_ms = move_time_ms
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, game: Game) -> tuple[EncodedAction, dict]:
        """Run MCTS from the current game state.

        Returns:
            (best_encoded_action, stats_dict)
        """
        # Deterministic random state for the search tree
        saved_rng = random.getstate()
        if self.seed is not None:
            random.seed(self.seed + game.state.num_turns * 997)

        try:
            return self._run(game)
        finally:
            random.setstate(saved_rng)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def _run(self, game: Game) -> tuple[EncodedAction, dict]:
        root = TreeNode(game.copy(), depth=0)
        root.init_unexpanded(self.filter)

        t0 = time.perf_counter()
        max_depth_seen = 0

        for sim in range(self.max_simulations):
            # Time budget check
            if self.move_time_ms is not None:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if elapsed_ms >= self.move_time_ms:
                    break

            # --- Selection ---
            node = root
            while (
                not node.is_terminal
                and node.depth < self.max_depth
                and node.children
                and (
                    not node.has_unexpanded
                    or len(node.children) >= self._expansion_limit(node.visits)
                )
            ):
                node = self._select_child(node)

            # --- Expansion ---
            if (
                not node.is_terminal
                and node.depth < self.max_depth
                and node.has_unexpanded
            ):
                ea = node.pop_unexpanded()
                if ea is not None:
                    child_game = node.game.copy()
                    raw_action = node.context.get_raw_action(ea)
                    try:
                        child_game.execute(raw_action)
                    except (ValueError, Exception):
                        # Action may be illegal in determinized/copied state
                        # (e.g. bank out of cards). Skip it.
                        continue

                    child = TreeNode(
                        child_game,
                        parent=node,
                        incoming_action=ea,
                        depth=node.depth + 1,
                    )
                    node.children[ea] = child

                    if child.depth < self.max_depth and not child.is_terminal:
                        child.init_unexpanded(self.filter)

                    node = child

            max_depth_seen = max(max_depth_seen, node.depth)

            # --- Evaluation (from root player's perspective) ---
            value = evaluate_leaf(node.game, self.root_color)

            # --- Backup ---
            self._backup(node, value)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        best_ea = self._best_action(root)

        stats = {
            "simulations": root.visits,
            "max_depth": max_depth_seen,
            "elapsed_ms": elapsed_ms,
            "root_children": {
                ActionCodec.decode_to_str(ea): {
                    "visits": ch.visits,
                    "avg_value": round(ch.value_avg, 4),
                }
                for ea, ch in sorted(
                    root.children.items(),
                    key=lambda p: (-p[1].visits, ActionCodec.sort_key(p[0])),
                )
            },
        }

        if log.isEnabledFor(logging.DEBUG):
            top = sorted(
                root.children.items(),
                key=lambda p: (-p[1].visits, ActionCodec.sort_key(p[0])),
            )[:5]
            log.debug(
                "MCTS %s — %d sims, depth=%d, %.0f ms  |  best=%s",
                self.root_color.value,
                root.visits,
                max_depth_seen,
                elapsed_ms,
                ActionCodec.decode_to_str(best_ea),
            )
            for ea, ch in top:
                log.debug(
                    "  %s  visits=%d  avg=%.4f",
                    ActionCodec.decode_to_str(ea),
                    ch.visits,
                    ch.value_avg,
                )

        return best_ea, stats

    # ------------------------------------------------------------------
    # Selection — UCT with perspective flip for 2-player
    # ------------------------------------------------------------------
    def _select_child(self, node: TreeNode) -> TreeNode:
        is_root_turn = node.game.state.current_color() == self.root_color
        parent_log_visits = math.log(node.visits) if node.visits > 0 else 0.0

        best_child: TreeNode | None = None
        best_uct = -math.inf

        # Deterministic iteration order via sorted encoded-action keys.
        for ea, child in sorted(
            node.children.items(), key=lambda p: ActionCodec.sort_key(p[0])
        ):
            if child.visits == 0:
                uct = math.inf
            else:
                avg = child.value_avg
                exploit = avg if is_root_turn else -avg
                explore = self.c * math.sqrt(parent_log_visits / child.visits)
                uct = exploit + explore

            if uct > best_uct:
                best_uct = uct
                best_child = child

        assert best_child is not None
        return best_child

    # ------------------------------------------------------------------
    # Backup — values are always from root player's perspective
    # ------------------------------------------------------------------
    @staticmethod
    def _backup(node: TreeNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Best root action — highest visit count, deterministic tie-break
    # ------------------------------------------------------------------
    @staticmethod
    def _best_action(root: TreeNode) -> EncodedAction:
        if not root.children:
            # Fallback: if no children were expanded, return first legal action.
            return root.context.encoded_actions[0]

        best_ea: EncodedAction | None = None
        best_visits = -1

        for ea, child in sorted(
            root.children.items(), key=lambda p: ActionCodec.sort_key(p[0])
        ):
            if child.visits > best_visits:
                best_visits = child.visits
                best_ea = ea

        assert best_ea is not None
        return best_ea

    @staticmethod
    def _expansion_limit(node_visits: int) -> int:
        """Progressive widening cap for number of expanded children."""
        return max(1, int(math.sqrt(max(1, node_visits))))
