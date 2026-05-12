"""Ablation study: priors vs value in Neural MCTS.

Loads the existing v3_combined checkpoint and runs four matchups against
frequency MCTS, toggling use_model_priors and use_model_value independently.

Usage:
    python scripts/run_ablation_study.py
    python scripts/run_ablation_study.py --eval-games 50  # 100 total with seat swap
"""

from __future__ import annotations

import argparse
import json
import logging
import random as _random
import time
from pathlib import Path
from typing import Any

from catanatron.models.player import Color, Player

from catan_ai.adapters.action_codec import ActionCodec
from catan_ai.belief.determinizer import Determinizer
from catan_ai.eval.arena import Arena
from catan_ai.players.belief_mcts_player import BeliefMCTSConfig, BeliefMCTSPlayer
from catan_ai.players.decision_context import DecisionContext
from catan_ai.players.neural_mcts_player import NeuralMCTS, NeuralMCTSConfig
from catan_ai.search.candidate_filter import CandidateFilter
from catan_ai.training.checkpoints import load_checkpoint

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO / "experiments" / "neural_scaling_study" / "v3_combined"
    / "checkpoints" / "best.pt"
)
OUTPUT_DIR = REPO / "experiments" / "ablation_study"


# ── NeuralFrequencyMCTSPlayer (reused from run_v3_combined) ──────────

class _NeuralFrequencyMCTSPlayer(Player):
    """NeuralMCTS over one count-only determinized frequency-belief world."""

    def __init__(self, color, *, model, config: NeuralMCTSConfig,
                 belief_seed: int, is_bot: bool = True):
        super().__init__(color, is_bot=is_bot)
        self.model = model
        self.config = config
        self.belief_seed = belief_seed
        self._calls = 0
        self._total_search_ms = 0.0

    def decide(self, game, playable_actions):
        self._calls += 1
        if len(playable_actions) == 1:
            return playable_actions[0]

        root_ctx = DecisionContext(game, playable_actions, self.color)
        rng = _random.Random(self.belief_seed + game.state.num_turns * 997 + id(self))
        world = Determinizer(
            acting_color=self.color,
            belief_mode="count_only",
            enable_devcard_sampling=False,
        ).sample_world(game, rng)
        if world is None:
            world = game

        t0 = time.perf_counter()
        engine = NeuralMCTS(
            root_color=self.color,
            model=self.model,
            cfg=self.config,
            candidate_filter=CandidateFilter(
                top_k_roads=self.config.top_k_roads,
                top_k_trades=self.config.top_k_trades,
                top_k_robber=self.config.top_k_robber,
            ),
        )
        best_ea, stats = engine.search(world)
        self._total_search_ms += (time.perf_counter() - t0) * 1000

        # Guard against the determinized world generating impossible bank trades
        valid_eas = set(root_ctx.encoded_actions)
        if best_ea not in valid_eas:
            # Fall back to the most visited action that is actually legal
            valid_candidates = []
            str_to_ea = {ActionCodec.decode_to_str(ea): ea for ea in valid_eas}
            for action_str, child_stats in stats["root_children"].items():
                ea = str_to_ea.get(action_str)
                if ea is not None:
                    valid_candidates.append((child_stats["visits"], ea))

            if valid_candidates:
                valid_candidates.sort(key=lambda t: (-t[0], ActionCodec.sort_key(t[1])))
                best_ea = valid_candidates[0][1]
            else:
                best_ea = root_ctx.encoded_actions[0]

        return root_ctx.get_raw_action(best_ea)

    @property
    def avg_move_ms(self) -> float:
        return self._total_search_ms / self._calls if self._calls > 0 else 0.0


# ── Ablation configs ─────────────────────────────────────────────────

ABLATION_CONFIGS: dict[str, dict[str, bool]] = {
    "neither":     {"use_model_priors": False, "use_model_value": False},
    "priors_only": {"use_model_priors": True,  "use_model_value": False},
    "value_only":  {"use_model_priors": False, "use_model_value": True},
    "full_neural": {"use_model_priors": True,  "use_model_value": True},
}


# ── Main ─────────────────────────────────────────────────────────────

def run_ablation(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    eval_games: int = 50,
    eval_sims: int = 30,
    eval_depth: int = 8,
    search_seed: int = 2026,
    top_k_roads: int = 3,
    top_k_trades: int = 2,
    top_k_robber: int = 4,
) -> dict[str, Any]:
    """Run four ablation matchups and write results."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Load the trained model
    log.info("Loading checkpoint: %s", checkpoint_path)
    model, checkpoint = load_checkpoint(checkpoint_path)
    log.info("Checkpoint epoch: %s", checkpoint.get("epoch"))

    # Frequency MCTS baseline opponent (same for all matchups)
    freq_cfg = BeliefMCTSConfig(
        num_worlds=1,
        sims_per_world=eval_sims,
        belief_seed=search_seed,
        belief_mode="count_only",
        enable_devcard_sampling=False,
        max_depth=eval_depth,
        top_k_roads=top_k_roads,
        top_k_trades=top_k_trades,
        top_k_robber=top_k_robber,
    )

    def make_frequency(color: Color) -> BeliefMCTSPlayer:
        return BeliefMCTSPlayer(color, config=freq_cfg)

    arena = Arena(num_games=eval_games, base_seed=6000, swap_seats=True)

    results: dict[str, Any] = {}

    for name, toggles in ABLATION_CONFIGS.items():
        neural_cfg = NeuralMCTSConfig(
            max_simulations=eval_sims,
            max_depth=eval_depth,
            puct_c=1.5,
            top_k_roads=top_k_roads,
            top_k_trades=top_k_trades,
            top_k_robber=top_k_robber,
            seed=search_seed,
            use_model_priors=toggles["use_model_priors"],
            use_model_value=toggles["use_model_value"],
            heuristic_value_weight=0.0 if toggles["use_model_value"] else 0.1,
        )

        def make_candidate(color: Color, _cfg=neural_cfg) -> _NeuralFrequencyMCTSPlayer:
            return _NeuralFrequencyMCTSPlayer(
                color, model=model, config=_cfg, belief_seed=search_seed,
            )

        label = f"{name} vs frequency_mcts"
        log.info("Running ablation: %s (%d games × 2 seats) ...", name, eval_games)
        match_result = arena.compare(make_candidate, make_frequency, label)

        results[name] = {
            "label": label,
            "use_model_priors": toggles["use_model_priors"],
            "use_model_value": toggles["use_model_value"],
            "games": match_result.games,
            "wins": match_result.wins,
            "losses": match_result.losses,
            "draws": match_result.draws,
            "win_rate": match_result.win_rate,
            "avg_final_vp": match_result.avg_final_vp,
            "avg_turns": match_result.avg_turns,
            "mean_latency_ms": match_result.avg_move_ms,
        }

        log.info("  → %s", match_result.summary())

    elapsed = time.perf_counter() - t0

    summary = {
        "experiment": "ablation_study",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "eval_games_per_seat": eval_games,
        "total_games_per_config": eval_games * 2,
        "eval_sims": eval_sims,
        "eval_depth": eval_depth,
        "elapsed_sec": elapsed,
        "results": results,
    }

    # Write JSON summary
    summary_path = OUTPUT_DIR / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info("Summary written to %s", summary_path)

    # Print results table
    print()
    print("=" * 72)
    print("ABLATION STUDY RESULTS")
    print("=" * 72)
    print(f"{'Config':<16} {'Priors':<8} {'Value':<8} {'Win Rate':>9} "
          f"{'W/L/D':>10} {'Avg VP':>7} {'Avg Turns':>10} {'Latency':>10}")
    print("-" * 72)
    for name, r in results.items():
        p = "✓" if r["use_model_priors"] else "✗"
        v = "✓" if r["use_model_value"] else "✗"
        wld = f"{r['wins']}/{r['losses']}/{r['draws']}"
        print(f"{name:<16} {p:<8} {v:<8} {r['win_rate']:>8.1%} "
              f"{wld:>10} {r['avg_final_vp']:>7.1f} {r['avg_turns']:>10.1f} "
              f"{r['mean_latency_ms']:>8.1f}ms")
    print("=" * 72)
    print(f"Total time: {elapsed:.0f}s")
    print()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural MCTS ablation study")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to trained checkpoint")
    parser.add_argument("--eval-games", type=int, default=50,
                        help="Games per seat per config (total = 2× this)")
    parser.add_argument("--eval-sims", type=int, default=30)
    parser.add_argument("--eval-depth", type=int, default=8)
    parser.add_argument("--search-seed", type=int, default=2026)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_ablation(
        checkpoint_path=args.checkpoint,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        eval_depth=args.eval_depth,
        search_seed=args.search_seed,
    )


if __name__ == "__main__":
    main()
