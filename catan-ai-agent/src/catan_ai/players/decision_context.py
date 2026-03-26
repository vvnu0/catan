"""Thin bridge between a single Catanatron decide() call and PublicState.

This is the *only* place outside the adapter package that is allowed to touch
raw Catanatron Action objects.  It builds a PublicState via the adapter,
establishes a stable mapping from EncodedAction → raw Action, and hands
everything to the downstream heuristic so the heuristic never sees raw
objects.
"""

from __future__ import annotations

from catanatron.game import Game
from catanatron.models.player import Color

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.catanatron_adapter import public_state_from_game
from catan_ai.adapters.public_state import EncodedAction, PublicState


class DecisionContext:
    """One-shot context built at the start of each decide() call.

    Attributes:
        public_state: Masked game snapshot for the acting player.
        encoded_actions: Deterministically sorted legal actions.
    """

    def __init__(self, game: Game, playable_actions, acting_color: Color):
        self.public_state: PublicState = public_state_from_game(
            game, acting_color
        )

        # Build a deterministic mapping: EncodedAction → raw Action.
        # Both lists are sorted the same way so indices correspond.
        paired = sorted(
            ((ActionCodec.encode(a), a) for a in playable_actions),
            key=lambda pair: ActionCodec.sort_key(pair[0]),
        )
        self._encoded: list[EncodedAction] = [ea for ea, _ in paired]
        self._raw_by_encoded: dict[EncodedAction, object] = {
            ea: raw for ea, raw in paired
        }

        self.encoded_actions: tuple[EncodedAction, ...] = tuple(self._encoded)

    def get_raw_action(self, encoded_action: EncodedAction):
        """Return the original Catanatron Action corresponding to *encoded_action*."""
        return self._raw_by_encoded[encoded_action]
