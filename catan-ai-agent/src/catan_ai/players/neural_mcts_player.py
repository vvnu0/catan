"""NeuralMCTSPlayer — MCTS guided by a learned policy/value network.

Extends the existing single-world MCTS with two optional neural components:
  1. **Priors** — the model's policy head provides PUCT-style prior
     probabilities that bias action selection toward promising moves.
  2. **Value** — the model's value head replaces (or blends with) the
     hand-crafted heuristic leaf evaluator.

Everything else (tree structure, candidate filtering, backup, root
action selection) is reused from MCTS v1.

The model scores the *current legal actions* at each node, avoiding a
fixed global action vocabulary.  Feature extraction uses PublicState
and EncodedAction only.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass

import torch

from catanatron.game import Game
from catanatron.models.player import Color, Player

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.public_state import EncodedAction
from catan_ai.models.action_features import ACTION_DIM, action_features, state_features
from catan_ai.models.policy_value_net import PolicyValueNet
from catan_ai.players.decision_context import DecisionContext
from catan_ai.search.candidate_filter import CandidateFilter
from catan_ai.search.leaf_evaluator import evaluate_leaf
from catan_ai.search.tree_node import TreeNode

log = logging.getLogger(__name__)


@dataclass
class NeuralMCTSConfig:
    """All tuneable parameters for NeuralMCTSPlayer."""

    max_simulations: int = 50
    max_depth: int = 10
    puct_c: float = 1.5
    top_k_roads: int = 3
    top_k_trades: int = 2
    top_k_robber: int = 4
    move_time_ms: int | None = None
    seed: int | None = None

    use_model_priors: bool = True
    use_model_value: bool = True
    heuristic_value_weight: float = 0.1


class NeuralMCTS:
    """MCTS with neural priors (PUCT) and neural leaf evaluation.

    This is a standalone search engine (not a Player subclass).  It holds
    a reference to a ``PolicyValueNet`` and uses it at each node for
    action priors and leaf values.
    """

    def __init__(
        self,
        root_color: Color,
        model: PolicyValueNet,
        cfg: NeuralMCTSConfig,
        candidate_filter: CandidateFilter | None = None,
    ):
        self.root_color = root_color
        self.model = model
        self.cfg = cfg
        self.filter = candidate_filter or CandidateFilter()

        # node id → {EncodedAction → prior probability}
        self._priors: dict[int, dict[EncodedAction, float]] = {}

    def search(self, game: Game) -> tuple[EncodedAction, dict]:
        """Run neural-guided MCTS from the current game state."""
        saved_rng = random.getstate()
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed + game.state.num_turns * 997)
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
        self._compute_priors(root)
        self._order_unexpanded_by_priors(root)

        t0 = time.perf_counter()
        max_depth_seen = 0
        cfg = self.cfg

        for _sim in range(cfg.max_simulations):
            if cfg.move_time_ms is not None:
                if (time.perf_counter() - t0) * 1000 >= cfg.move_time_ms:
                    break

            # --- Selection ---
            node = root
            while (
                not node.is_terminal
                and node.depth < cfg.max_depth
                and node.children
                and (
                    not node.has_unexpanded
                    or len(node.children) >= _expansion_limit(node.visits)
                )
            ):
                node = self._select_child_puct(node)

            # --- Expansion ---
            if (
                not node.is_terminal
                and node.depth < cfg.max_depth
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

                    if child.depth < cfg.max_depth and not child.is_terminal:
                        child.init_unexpanded(self.filter)
                        self._compute_priors(child)
                        self._order_unexpanded_by_priors(child)

                    node = child

            max_depth_seen = max(max_depth_seen, node.depth)

            # --- Evaluation ---
            value = self._evaluate(node)

            # --- Backup ---
            _backup(node, value)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        best_ea = _best_action(root)

        stats = {
            "simulations": root.visits,
            "max_depth": max_depth_seen,
            "elapsed_ms": elapsed_ms,
            "root_children": {
                ActionCodec.decode_to_str(ea): {
                    "visits": ch.visits,
                    "avg_value": round(ch.value_avg, 4),
                    "prior": round(self._priors.get(id(root), {}).get(ea, 0.0), 4),
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
                "NeuralMCTS %s — %d sims, depth=%d, %.0f ms  |  best=%s",
                self.root_color.value,
                root.visits,
                max_depth_seen,
                elapsed_ms,
                ActionCodec.decode_to_str(best_ea),
            )
            priors = self._priors.get(id(root), {})
            for ea, ch in top:
                log.debug(
                    "  %s  visits=%d  avg=%.4f  prior=%.4f",
                    ActionCodec.decode_to_str(ea),
                    ch.visits,
                    ch.value_avg,
                    priors.get(ea, 0.0),
                )

        return best_ea, stats

    # ------------------------------------------------------------------
    # PUCT selection — replaces UCT with neural priors
    # ------------------------------------------------------------------
    def _select_child_puct(self, node: TreeNode) -> TreeNode:
        """PUCT selection: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))."""
        is_root_turn = node.game.state.current_color() == self.root_color
        priors = self._priors.get(id(node), {})
        sqrt_parent = math.sqrt(max(1, node.visits))

        best_child: TreeNode | None = None
        best_score = -math.inf

        for ea, child in sorted(
            node.children.items(), key=lambda p: ActionCodec.sort_key(p[0])
        ):
            prior = priors.get(ea, 1.0 / max(1, len(node.children)))
            q = child.value_avg if is_root_turn else -child.value_avg
            u = self.cfg.puct_c * prior * sqrt_parent / (1 + child.visits)
            score = q + u

            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child

    # ------------------------------------------------------------------
    # Neural prior computation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_priors(self, node: TreeNode) -> None:
        """Run the model to get action priors for *node*'s candidate actions."""
        if not self.cfg.use_model_priors:
            return

        ctx = node.context
        ps = ctx.public_state
        eas = ctx.encoded_actions
        if not eas:
            return

        s_feats = torch.tensor([state_features(ps)], dtype=torch.float32)
        a_feats_list = [action_features(ea, ps) for ea in eas]
        a_feats = torch.tensor([a_feats_list], dtype=torch.float32)
        mask = torch.ones(1, len(eas), dtype=torch.bool)

        logits, _ = self.model(s_feats, a_feats, mask)
        probs = torch.softmax(logits[0, : len(eas)], dim=0)

        prior_map: dict[EncodedAction, float] = {}
        for i, ea in enumerate(eas):
            prior_map[ea] = probs[i].item()

        self._priors[id(node)] = prior_map

    def _order_unexpanded_by_priors(self, node: TreeNode) -> None:
        """Prioritize first expansion by model policy, with stable tie-breaks."""
        if not self.cfg.use_model_priors:
            return

        priors = self._priors.get(id(node))
        if not priors:
            return

        node.sort_unexpanded(
            key=lambda ea: (-priors.get(ea, 0.0), ActionCodec.sort_key(ea))
        )

    # ------------------------------------------------------------------
    # Leaf evaluation — neural and/or heuristic
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, node: TreeNode) -> float:
        """Evaluate a leaf node from root_color's perspective."""
        winner = node.game.winning_color()
        if winner is not None:
            return 1.0 if winner == self.root_color else -1.0

        neural_value = 0.0
        heuristic_value = 0.0

        if self.cfg.use_model_value:
            ps = node.context.public_state
            eas = node.context.encoded_actions
            s_feats = torch.tensor([state_features(ps)], dtype=torch.float32)
            a_feats_list = [action_features(ea, ps) for ea in eas] if eas else [[0.0] * ACTION_DIM]
            a_feats = torch.tensor([a_feats_list], dtype=torch.float32)
            mask = torch.ones(1, len(a_feats_list), dtype=torch.bool)

            _, value = self.model(s_feats, a_feats, mask)

            # Model produces value from acting player's perspective;
            # we need it from root player's perspective.
            acting = node.game.state.current_color()
            neural_value = value.item()
            if acting != self.root_color:
                neural_value = -neural_value

        w = self.cfg.heuristic_value_weight
        if w > 0.0 or not self.cfg.use_model_value:
            heuristic_value = evaluate_leaf(node.game, self.root_color)

        if not self.cfg.use_model_value:
            return heuristic_value

        return (1.0 - w) * neural_value + w * heuristic_value


# -----------------------------------------------------------------------
# Shared helpers (same logic as MCTS v1, extracted to avoid subclassing)
# -----------------------------------------------------------------------

def _backup(node: TreeNode, value: float) -> None:
    while node is not None:
        node.visits += 1
        node.value_sum += value
        node = node.parent  # type: ignore[assignment]


def _best_action(root: TreeNode) -> EncodedAction:
    if not root.children:
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


def _expansion_limit(node_visits: int) -> int:
    return max(1, int(math.sqrt(max(1, node_visits))))


# -----------------------------------------------------------------------
# Player wrapper
# -----------------------------------------------------------------------

class NeuralMCTSPlayer(Player):
    """Catanatron Player that uses neural-guided MCTS for decisions.

    Args:
        color: Player colour.
        model: Trained PolicyValueNet (or None for heuristic-only fallback).
        config: Search and model-usage parameters.
    """

    def __init__(
        self,
        color,
        *,
        model: PolicyValueNet | None = None,
        config: NeuralMCTSConfig | None = None,
        is_bot: bool = True,
    ):
        super().__init__(color, is_bot=is_bot)
        self.model = model
        self.config = config or NeuralMCTSConfig()
        self._calls = 0
        self._total_search_ms = 0.0

    def decide(self, game, playable_actions):
        self._calls += 1

        if len(playable_actions) == 1:
            return playable_actions[0]

        root_ctx = DecisionContext(game, playable_actions, self.color)
        t0 = time.perf_counter()

        cfg = self.config
        candidate_filter = CandidateFilter(
            top_k_roads=cfg.top_k_roads,
            top_k_trades=cfg.top_k_trades,
            top_k_robber=cfg.top_k_robber,
        )

        if self.model is not None:
            engine = NeuralMCTS(
                root_color=self.color,
                model=self.model,
                cfg=cfg,
                candidate_filter=candidate_filter,
            )
        else:
            # Fallback: use plain MCTS with heuristic evaluation
            from catan_ai.search.mcts import MCTS

            engine = MCTS(
                root_color=self.color,
                max_simulations=cfg.max_simulations,
                max_depth=cfg.max_depth,
                exploration_c=cfg.puct_c,
                candidate_filter=candidate_filter,
                move_time_ms=cfg.move_time_ms,
                seed=cfg.seed,
            )

        best_ea, stats = engine.search(game)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_search_ms += elapsed_ms

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "NeuralMCTSPlayer %s call #%d — %s (%.0f ms, %d sims, depth %d)",
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
