"""Self-play data generation using an MCTS teacher.

Each game produces a list of decision samples.  After the game finishes
the final outcome is back-filled into every sample so the value target
reflects the actual game result from the acting player's perspective.

Samples are written to disk as shards via ``torch.save``.
"""

from __future__ import annotations

import io
import contextlib
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from catanatron import Color, Game
from catanatron.models.player import Player

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.adapters.catanatron_adapter import public_state_from_game
from catan_ai.adapters.public_state import EncodedAction
from catan_ai.models.action_features import action_features, state_features
from catan_ai.players.decision_context import DecisionContext
from catan_ai.search.candidate_filter import CandidateFilter
from catan_ai.search.mcts import MCTS

log = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Parameters for self-play data generation."""

    num_games: int = 10
    output_dir: str = "data/self_play"
    seed: int = 42
    teacher_type: str = "mcts"

    # MCTS teacher budget
    max_simulations: int = 50
    max_depth: int = 10
    exploration_c: float = 1.41
    top_k_roads: int = 3
    top_k_trades: int = 2
    top_k_robber: int = 4

    shard_size: int = 512


@dataclass
class DecisionSample:
    """One training sample from a single decision point."""

    state_feats: list[float]
    action_feats: list[list[float]]
    encoded_actions: list[str]
    target_policy: list[float]
    chosen_action: str
    target_value: float  # filled after game ends
    meta: dict = field(default_factory=dict)


class _RecordingMCTSPlayer(Player):
    """MCTS player that records decision samples during self-play.

    Not meant for direct use — instantiated internally by ``run_self_play``.
    """

    def __init__(
        self,
        color,
        *,
        cfg: SelfPlayConfig,
        game_id: int,
    ):
        super().__init__(color, is_bot=True)
        self.cfg = cfg
        self.game_id = game_id
        self.samples: list[DecisionSample] = []
        self._ply = 0

    def decide(self, game, playable_actions):
        self._ply += 1

        if len(playable_actions) == 1:
            return playable_actions[0]

        cfg = self.cfg
        root_ctx = DecisionContext(game, playable_actions, self.color)
        ps = root_ctx.public_state

        mcts = MCTS(
            root_color=self.color,
            max_simulations=cfg.max_simulations,
            max_depth=cfg.max_depth,
            exploration_c=cfg.exploration_c,
            candidate_filter=CandidateFilter(
                top_k_roads=cfg.top_k_roads,
                top_k_trades=cfg.top_k_trades,
                top_k_robber=cfg.top_k_robber,
            ),
            seed=cfg.seed,
        )
        best_ea, stats = mcts.search(game)

        # Build normalised visit distribution over *all* root legal actions.
        visit_counts: list[int] = []
        for ea in root_ctx.encoded_actions:
            ea_str = ActionCodec.decode_to_str(ea)
            child_info = stats["root_children"].get(ea_str)
            visit_counts.append(child_info["visits"] if child_info else 0)

        total_visits = sum(visit_counts)
        if total_visits > 0:
            target_policy = [v / total_visits for v in visit_counts]
        else:
            n = len(visit_counts)
            target_policy = [1.0 / n] * n

        s_feats = state_features(ps)
        a_feats = [action_features(ea) for ea in root_ctx.encoded_actions]
        ea_strs = [ActionCodec.decode_to_str(ea) for ea in root_ctx.encoded_actions]

        sample = DecisionSample(
            state_feats=s_feats,
            action_feats=a_feats,
            encoded_actions=ea_strs,
            target_policy=target_policy,
            chosen_action=ActionCodec.decode_to_str(best_ea),
            target_value=0.0,
            meta={
                "seed": cfg.seed,
                "game_id": self.game_id,
                "ply": self._ply,
                "color": self.color.value,
                "teacher": cfg.teacher_type,
                "config_hash": _config_hash(cfg),
            },
        )
        self.samples.append(sample)

        return root_ctx.get_raw_action(best_ea)


def run_self_play(cfg: SelfPlayConfig) -> Path:
    """Run self-play games and write training shards to ``cfg.output_dir``.

    Returns the output directory path.
    """
    if cfg.teacher_type != "mcts":
        raise ValueError(
            "Unsupported teacher_type. This phase currently supports only "
            "teacher_type='mcts' for visit-distribution targets."
        )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict] = []
    t0 = time.perf_counter()

    for game_idx in range(cfg.num_games):
        game_seed = cfg.seed + game_idx

        red = _RecordingMCTSPlayer(Color.RED, cfg=cfg, game_id=game_idx)
        blue = _RecordingMCTSPlayer(Color.BLUE, cfg=cfg, game_id=game_idx)

        game = Game([red, blue], seed=game_seed)
        with contextlib.redirect_stdout(io.StringIO()):
            winner = game.play()

        # Back-fill value targets with the actual game outcome.
        for player in (red, blue):
            if winner is None:
                outcome = 0.0
            elif winner == player.color:
                outcome = 1.0
            else:
                outcome = -1.0

            for sample in player.samples:
                sample.target_value = outcome
                all_samples.append(_sample_to_dict(sample))

        log.info(
            "Self-play game %d/%d — winner=%s turns=%d samples=%d",
            game_idx + 1,
            cfg.num_games,
            winner.value if winner else "draw",
            game.state.num_turns,
            len(red.samples) + len(blue.samples),
        )

    elapsed = time.perf_counter() - t0
    log.info(
        "Self-play complete: %d games, %d samples, %.1f s",
        cfg.num_games,
        len(all_samples),
        elapsed,
    )

    _write_shards(all_samples, out, cfg.shard_size)
    return out


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _sample_to_dict(sample: DecisionSample) -> dict:
    return {
        "state_feats": torch.tensor(sample.state_feats, dtype=torch.float32),
        "action_feats": torch.tensor(sample.action_feats, dtype=torch.float32),
        "target_policy": torch.tensor(sample.target_policy, dtype=torch.float32),
        "target_value": torch.tensor(sample.target_value, dtype=torch.float32),
        "encoded_actions": sample.encoded_actions,
        "chosen_action": sample.chosen_action,
        "meta": sample.meta,
    }


def _write_shards(samples: list[dict], out_dir: Path, shard_size: int) -> None:
    for i in range(0, len(samples), shard_size):
        shard = samples[i : i + shard_size]
        path = out_dir / f"shard_{i // shard_size:04d}.pt"
        torch.save(shard, path)
        log.info("Wrote shard %s (%d samples)", path.name, len(shard))


def _config_hash(cfg: SelfPlayConfig) -> str:
    key = f"{cfg.max_simulations}_{cfg.max_depth}_{cfg.exploration_c}"
    return hashlib.md5(key.encode()).hexdigest()[:8]
