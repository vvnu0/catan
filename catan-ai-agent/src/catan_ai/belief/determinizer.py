"""Determinizer — sample a full hidden-world consistent with public evidence.

Creates a *copy* of the live game with hidden state overwritten to match
a single sampled world.  The live game is never mutated.

What gets overwritten on the copy:
  - Opponent resource hand identities  (from ResourceBelief)
  - Bank resource counts  (recomputed for conservation)
  - Dev-card deck  (always reshuffled for variance)
  - Opponent dev-card hand identities  (optional, from DevCardBelief)

What is preserved exactly:
  - Acting player's own resource hand
  - Acting player's own dev-card hand
  - All board structures and buildings
  - All public player-state fields (VP, longest road, etc.)
  - Action history and turn counters
  - Playable actions at the root

Limitations:
  - This is a practical single-world sampler, not a game-theoretic solution.
  - In 2-player, resource determinization is a no-op because conservation
    fully determines the opponent's hand from bank + own hand.
  - Dev-card sampling is optional and off by default.
"""

from __future__ import annotations

import logging
import random as _random

from catanatron.game import Game
from catanatron.models.enums import RESOURCES
from catanatron.models.player import Color
from catanatron.state_functions import player_key

from catan_ai.belief.devcard_belief import DEV_CARD_TYPES, DevCardBelief
from catan_ai.belief.public_history import (
    RESOURCE_TYPES,
    PublicEvidence,
    extract_public_evidence,
)
from catan_ai.belief.resource_belief import PER_RESOURCE_TOTAL, ResourceBelief

log = logging.getLogger(__name__)


class Determinizer:
    """Sample determinized game copies for multi-world search.

    Args:
        acting_color: The player we are searching for (hands kept exact).
        belief_mode: Resource belief mode ('conservation', 'count_only',
            'frequency_with_constraints').
        enable_devcard_sampling: If True, sample opponent dev-card identities
            and reshuffle the deck.  If False, only shuffle the deck order.
        max_invalid_samples: Retry limit before giving up on one world.
    """

    def __init__(
        self,
        acting_color: Color,
        *,
        belief_mode: str = "conservation",
        enable_devcard_sampling: bool = False,
        max_invalid_samples: int = 20,
    ):
        self.acting_color = acting_color
        self.belief_mode = belief_mode
        self.enable_devcard_sampling = enable_devcard_sampling
        self.max_invalid_samples = max_invalid_samples

    def sample_world(
        self, game: Game, rng: _random.Random
    ) -> Game | None:
        """Create one determinized game copy, or ``None`` on failure.

        Never mutates *game*.
        """
        evidence = extract_public_evidence(game, self.acting_color)
        res_belief = ResourceBelief(evidence, mode=self.belief_mode)

        dev_belief: DevCardBelief | None = None
        if self.enable_devcard_sampling:
            dev_belief = DevCardBelief(evidence)

        for _ in range(self.max_invalid_samples):
            world = self._try_sample(game, evidence, res_belief, dev_belief, rng)
            if world is not None:
                return world

        log.debug("Determinizer: exhausted retry limit (%d)", self.max_invalid_samples)
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _try_sample(
        self,
        game: Game,
        evidence: PublicEvidence,
        res_belief: ResourceBelief,
        dev_belief: DevCardBelief | None,
        rng: _random.Random,
    ) -> Game | None:
        world = game.copy()
        state = world.state

        # --- 1. Sample and inject opponent resources ---
        opp_hands = res_belief.sample_opponent_hands(rng)
        for opp_ev in evidence.opponent_evidence:
            color_enum = _color_from_value(state, opp_ev.color)
            if color_enum is None:
                return None
            key = player_key(state, color_enum)

            hand = opp_hands.get(opp_ev.color)
            if hand is None:
                return None

            total_assigned = sum(hand.values())
            if total_assigned != opp_ev.total_resource_cards:
                return None

            for r in RESOURCE_TYPES:
                if hand[r] < 0:
                    return None
                state.player_state[f"{key}_{r}_IN_HAND"] = hand[r]

        # --- 2. Recompute bank from conservation ---
        for i, r in enumerate(RESOURCE_TYPES):
            player_total = sum(
                state.player_state[f"{player_key(state, c)}_{r}_IN_HAND"]
                for c in state.colors
            )
            bank_r = PER_RESOURCE_TOTAL - player_total
            if bank_r < 0:
                return None
            state.resource_freqdeck[i] = bank_r

        # --- 3. Dev-card handling ---
        if dev_belief is not None:
            result = dev_belief.sample_hand_and_deck(rng)
            if result is None:
                return None

            opp_dev_hands, new_deck = result

            for opp_ev in evidence.opponent_evidence:
                color_enum = _color_from_value(state, opp_ev.color)
                if color_enum is None:
                    return None
                key = player_key(state, color_enum)
                dev_hand = opp_dev_hands.get(opp_ev.color)
                if dev_hand is None:
                    return None
                for d in DEV_CARD_TYPES:
                    state.player_state[f"{key}_{d}_IN_HAND"] = dev_hand.get(d, 0)

            state.development_listdeck = new_deck
        else:
            # Even without full dev-card sampling, shuffle the deck so that
            # different worlds see different draw orders during search.
            rng.shuffle(state.development_listdeck)

        return world


def _color_from_value(state, color_str: str) -> Color | None:
    """Resolve a color string like 'RED' back to the Color enum."""
    for c in state.colors:
        if c.value == color_str:
            return c
    return None
