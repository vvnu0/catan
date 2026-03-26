"""Tree node for MCTS.

Each node holds a *copy* of the Catanatron Game at that point in the search
tree, along with visit statistics used by UCT selection.
"""

from __future__ import annotations

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.public_state import EncodedAction
from catan_ai.players.decision_context import DecisionContext


class TreeNode:
    """Single node in the MCTS game tree."""

    __slots__ = (
        "parent",
        "incoming_action",
        "game",
        "depth",
        "visits",
        "value_sum",
        "children",
        "_context",
        "_unexpanded",
    )

    def __init__(self, game, *, parent=None, incoming_action=None, depth=0):
        self.game = game
        self.parent: TreeNode | None = parent
        self.incoming_action: EncodedAction | None = incoming_action
        self.depth: int = depth

        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: dict[EncodedAction, TreeNode] = {}

        self._context: DecisionContext | None = None
        self._unexpanded: list[EncodedAction] | None = None

    # ------------------------------------------------------------------
    # Lazy context — built once on first access
    # ------------------------------------------------------------------
    @property
    def context(self) -> DecisionContext:
        if self._context is None:
            acting = self.game.state.current_color()
            self._context = DecisionContext(
                self.game, self.game.playable_actions, acting,
            )
        return self._context

    # ------------------------------------------------------------------
    # Unexpanded action management
    # ------------------------------------------------------------------
    def init_unexpanded(self, filter_fn) -> None:
        """Populate the unexpanded list using *filter_fn(ps, actions)*."""
        ctx = self.context
        filtered = filter_fn(ctx.public_state, ctx.encoded_actions)
        self._unexpanded = list(filtered)

    def pop_unexpanded(self) -> EncodedAction | None:
        """Remove and return the next unexpanded action, or None."""
        if self._unexpanded:
            return self._unexpanded.pop(0)
        return None

    @property
    def has_unexpanded(self) -> bool:
        return bool(self._unexpanded)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        return self.game.winning_color() is not None

    @property
    def is_fully_expanded(self) -> bool:
        return self._unexpanded is not None and len(self._unexpanded) == 0

    @property
    def value_avg(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0
