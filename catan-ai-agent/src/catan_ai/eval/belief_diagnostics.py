"""Compact diagnostics for opponent-modeling ablations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from catan_ai.players.belief_mcts_player import BeliefMCTSPlayer


@dataclass
class ModeDiagnostics:
    mode: str
    players_created: int = 0
    decisions: int = 0
    worlds_attempted: int = 0
    worlds_used: int = 0
    invalid_samples: int = 0
    fallbacks: int = 0
    expected_worlds_per_decision: int = 0

    @property
    def rejection_rate(self) -> float:
        return self.invalid_samples / self.worlds_attempted if self.worlds_attempted else 0.0

    @property
    def avg_worlds_used_per_decision(self) -> float:
        return self.worlds_used / self.decisions if self.decisions else 0.0

    @property
    def active_belief_decisions_estimate(self) -> float:
        if self.expected_worlds_per_decision <= 0:
            return 0.0
        return self.worlds_attempted / self.expected_worlds_per_decision

    @property
    def avg_belief_updates_per_player_game(self) -> float:
        return self.active_belief_decisions_estimate / self.players_created if self.players_created else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.decisions if self.decisions else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update({
            "rejection_rate": self.rejection_rate,
            "avg_worlds_used_per_decision": self.avg_worlds_used_per_decision,
            "active_belief_decisions_estimate": self.active_belief_decisions_estimate,
            "avg_belief_updates_per_player_game": self.avg_belief_updates_per_player_game,
            "fallback_rate": self.fallback_rate,
            "basic_consistency_ok": self.worlds_used >= 0
            and self.worlds_attempted >= self.worlds_used
            and self.invalid_samples >= 0,
        })
        return data


class BeliefDiagnosticsCollector:
    """Collect diagnostics from players created during arena runs."""

    def __init__(self, modes: list[str]):
        self._players: dict[str, list[object]] = {mode: [] for mode in modes}
        self._expected_worlds: dict[str, int] = {mode: 0 for mode in modes}

    def track(self, mode: str, player: object) -> object:
        self._players.setdefault(mode, []).append(player)
        if isinstance(player, BeliefMCTSPlayer):
            self._expected_worlds[mode] = max(
                self._expected_worlds.get(mode, 0),
                int(player.config.num_worlds),
            )
        return player

    def snapshot(self) -> dict[str, dict]:
        return {
            mode: self._summarize(mode, players).to_dict()
            for mode, players in self._players.items()
        }

    def _summarize(self, mode: str, players: list[object]) -> ModeDiagnostics:
        diag = ModeDiagnostics(
            mode=mode,
            players_created=len(players),
            expected_worlds_per_decision=self._expected_worlds.get(mode, 0),
        )
        for player in players:
            diag.decisions += int(getattr(player, "_calls", 0))
            diag.worlds_attempted += int(getattr(player, "_worlds_attempted_total", 0))
            diag.worlds_used += int(getattr(player, "_worlds_used_total", 0))
            diag.invalid_samples += int(getattr(player, "_invalid_samples_total", 0))
            diag.fallbacks += int(getattr(player, "_fallback_count", 0))
        return diag


def resource_entropy(counts: dict[str, int]) -> float:
    """Return entropy for a resource-count distribution."""
    total = sum(max(0, v) for v in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy
