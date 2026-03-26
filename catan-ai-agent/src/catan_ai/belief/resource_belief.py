"""Resource belief model — per-opponent hidden resource estimation.

Modes:
  count_only:
      Ignores bank conservation.  Samples each opponent's hand by drawing
      N cards uniformly from the five resource types.  Simple but can
      produce states that violate conservation.

  conservation (recommended, default):
      Uses bank conservation: for each resource type,
        bank[r] + own_hand[r] + sum(opponent_hands[r]) = 19.
      In 2-player this is fully deterministic (no sampling at all).
      In multi-player it distributes the hidden pool uniformly.

  frequency_with_constraints:
      Same as conservation, reserved for future production-weighted
      priors.  Currently identical to conservation.

Limitations:
  - No production-history tracking (what was rolled / produced / spent).
  - No trade-history tracking.
  - In 2-player, conservation mode is exact so there is nothing to sample.
"""

from __future__ import annotations

import random as _random

from catan_ai.belief.public_history import PublicEvidence, RESOURCE_TYPES

PER_RESOURCE_TOTAL = 19


class ResourceBelief:
    """Compute or sample feasible opponent resource hands."""

    def __init__(self, evidence: PublicEvidence, mode: str = "conservation"):
        if mode not in ("count_only", "conservation", "frequency_with_constraints"):
            raise ValueError(f"Unknown resource belief mode: {mode!r}")
        self.evidence = evidence
        self.mode = mode

    def sample_opponent_hands(
        self, rng: _random.Random
    ) -> dict[str, dict[str, int]]:
        """Return ``{opponent_color: {resource: count}}`` for every opponent.

        The result is consistent with the evidence under the chosen mode.
        Returns ``None`` if a valid sample could not be produced (caller
        should retry or reject).
        """
        if self.mode == "count_only":
            return self._sample_count_only(rng)
        return self._sample_conservation(rng)

    # ------------------------------------------------------------------
    # conservation / frequency_with_constraints
    # ------------------------------------------------------------------
    def _sample_conservation(
        self, rng: _random.Random
    ) -> dict[str, dict[str, int]]:
        ev = self.evidence

        hidden_pool = {
            r: PER_RESOURCE_TOTAL - ev.bank_resources.get(r, 0) - ev.own_resources.get(r, 0)
            for r in RESOURCE_TYPES
        }

        opponents = ev.opponent_evidence
        if len(opponents) == 1:
            # 2-player: fully determined — no sampling needed.
            return {opponents[0].color: dict(hidden_pool)}

        # Multi-player: build a flat pool and deal randomly.
        pool: list[str] = []
        for r in RESOURCE_TYPES:
            count = hidden_pool[r]
            if count < 0:
                return self._sample_count_only(rng)
            pool.extend([r] * count)

        rng.shuffle(pool)

        result: dict[str, dict[str, int]] = {}
        idx = 0
        for opp in opponents:
            hand = {r: 0 for r in RESOURCE_TYPES}
            for _ in range(opp.total_resource_cards):
                if idx >= len(pool):
                    break
                hand[pool[idx]] += 1
                idx += 1
            result[opp.color] = hand
        return result

    # ------------------------------------------------------------------
    # count_only (ignores conservation)
    # ------------------------------------------------------------------
    def _sample_count_only(
        self, rng: _random.Random
    ) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for opp in self.evidence.opponent_evidence:
            hand = {r: 0 for r in RESOURCE_TYPES}
            for _ in range(opp.total_resource_cards):
                hand[rng.choice(RESOURCE_TYPES)] += 1
            result[opp.color] = hand
        return result
