"""Dev-card belief model — hidden dev-card identity estimation.

Tracks the unseen dev-card pool (cards that are neither played, nor in the
acting player's hand) and can sample how those unseen cards are distributed
between opponents' hands and the remaining deck.

Starting composition (25 total):
    KNIGHT × 14, YEAR_OF_PLENTY × 2, MONOPOLY × 2,
    ROAD_BUILDING × 2, VICTORY_POINT × 5

Limitations:
  - No Bayesian updates from observed play patterns.
  - Off by default in the belief config (resource determinization is the
    primary focus for this version).
  - When disabled, the determinizer still shuffles the deck for variance.
"""

from __future__ import annotations

import random as _random

from catan_ai.belief.public_history import DEV_CARD_TYPES, PublicEvidence

STARTING_DEV_COUNTS: dict[str, int] = {
    "KNIGHT": 14,
    "YEAR_OF_PLENTY": 2,
    "MONOPOLY": 2,
    "ROAD_BUILDING": 2,
    "VICTORY_POINT": 5,
}


class DevCardBelief:
    """Compute or sample feasible opponent dev-card hands + remaining deck.

    The *unseen pool* for each dev-card type is::

        unseen[type] = starting[type]
                       - all_played[type]   (summed across all players)
                       - own_in_hand[type]  (acting player's known hand)

    The unseen pool must be split into:
      - one sub-hand per opponent (matching their known total count)
      - the remaining deck (matching the known deck size)
    """

    def __init__(self, evidence: PublicEvidence):
        self.evidence = evidence

        all_played: dict[str, int] = {d: 0 for d in DEV_CARD_TYPES}
        for d in DEV_CARD_TYPES:
            all_played[d] += evidence.own_played_dev_cards.get(d, 0)
        for opp in evidence.opponent_evidence:
            for d in DEV_CARD_TYPES:
                all_played[d] += opp.played_dev_cards.get(d, 0)

        self.unseen_pool: dict[str, int] = {}
        for d in DEV_CARD_TYPES:
            self.unseen_pool[d] = (
                STARTING_DEV_COUNTS[d]
                - all_played.get(d, 0)
                - evidence.own_dev_cards.get(d, 0)
            )

        self.total_unseen = sum(max(0, v) for v in self.unseen_pool.values())

        expected_unseen = evidence.dev_deck_size + sum(
            o.total_dev_cards for o in evidence.opponent_evidence
        )
        self.consistent = self.total_unseen == expected_unseen

    def sample_hand_and_deck(
        self, rng: _random.Random
    ) -> tuple[dict[str, dict[str, int]], list[str]] | None:
        """Sample dev-card identities for all opponents and the remaining deck.

        Returns ``(opponent_hands, deck)`` where:
          - opponent_hands: ``{color: {dev_type: count}}``
          - deck: list of dev-card type strings (shuffled)

        Returns ``None`` if a consistent sample cannot be produced (caller
        should reject this world).
        """
        if not self.consistent:
            return None

        ev = self.evidence

        # Build flat pool of unseen dev cards.
        pool: list[str] = []
        for d in DEV_CARD_TYPES:
            count = self.unseen_pool[d]
            if count < 0:
                return None
            pool.extend([d] * count)

        rng.shuffle(pool)

        # Deal to each opponent based on their known total dev-card count.
        opponent_hands: dict[str, dict[str, int]] = {}
        idx = 0
        for opp in ev.opponent_evidence:
            hand = {d: 0 for d in DEV_CARD_TYPES}
            for _ in range(opp.total_dev_cards):
                if idx >= len(pool):
                    return None
                hand[pool[idx]] += 1
                idx += 1
            opponent_hands[opp.color] = hand

        deck = pool[idx:]
        return opponent_hands, deck
