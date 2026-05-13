"""Matchup: Full Neural MCTS vs Priors Only Neural MCTS.

Usage:
    python scripts/run_priors_vs_full.py
    python scripts/run_priors_vs_full.py --eval-games 50
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


def run_matchup(
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
    """Run matchup and write results."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    log.info("Loading checkpoint: %s", checkpoint_path)
    model, checkpoint = load_checkpoint(checkpoint_path)
    log.info("Checkpoint epoch: %s", checkpoint.get("epoch"))

    # We use a different base seed so the games are distinct from the ablation study
    arena = Arena(num_games=eval_games, base_seed=8888, swap_seats=True)

    # 1. Priors Only Config
    priors_cfg = NeuralMCTSConfig(
        max_simulations=eval_sims,
        max_depth=eval_depth,
        puct_c=1.5,
        top_k_roads=top_k_roads,
        top_k_trades=top_k_trades,
        top_k_robber=top_k_robber,
        seed=search_seed,
        use_model_priors=True,
        use_model_value=False,
        heuristic_value_weight=0.1,
    )

    def make_priors_only(color: Color) -> _NeuralFrequencyMCTSPlayer:
        return _NeuralFrequencyMCTSPlayer(
            color, model=model, config=priors_cfg, belief_seed=search_seed,
        )

    # 2. Full Neural Config
    full_cfg = NeuralMCTSConfig(
        max_simulations=eval_sims,
        max_depth=eval_depth,
        puct_c=1.5,
        top_k_roads=top_k_roads,
        top_k_trades=top_k_trades,
        top_k_robber=top_k_robber,
        seed=search_seed,
        use_model_priors=True,
        use_model_value=True,
        heuristic_value_weight=0.0,
    )

    def make_full_neural(color: Color) -> _NeuralFrequencyMCTSPlayer:
        return _NeuralFrequencyMCTSPlayer(
            color, model=model, config=full_cfg, belief_seed=search_seed,
        )

    label = "full_neural vs priors_only"
    log.info("Running matchup: %s (%d games × 2 seats) ...", label, eval_games)
    
    # Candidate 1 = Full Neural, Candidate 2 = Priors Only
    match_result = arena.compare(make_full_neural, make_priors_only, label)

    log.info("  → %s", match_result.summary())

    elapsed = time.perf_counter() - t0

    summary = {
        "experiment": "full_vs_priors_only",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "eval_games_per_seat": eval_games,
        "total_games": eval_games * 2,
        "eval_sims": eval_sims,
        "eval_depth": eval_depth,
        "elapsed_sec": elapsed,
        "results": {
            "label": label,
            "games": match_result.games,
            "wins_for_full_neural": match_result.wins,
            "losses_for_full_neural": match_result.losses,
            "draws": match_result.draws,
            "win_rate_for_full_neural": match_result.win_rate,
            "avg_final_vp": match_result.avg_final_vp,
            "avg_turns": match_result.avg_turns,
            "mean_latency_ms": match_result.avg_move_ms,
        }
    }

    summary_path = OUTPUT_DIR / "full_vs_priors_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info("Summary written to %s", summary_path)

    # Print results table
    print()
    print("=" * 72)
    print("MATCHUP RESULTS: Full Neural vs Priors Only")
    print("=" * 72)
    print(f"Win Rate (Full Neural): {match_result.win_rate:>8.1%}")
    print(f"W/L/D (Full Neural):    {match_result.wins}/{match_result.losses}/{match_result.draws}")
    print(f"Avg Final VP:           {match_result.avg_final_vp:>7.1f}")
    print(f"Avg Game Length:        {match_result.avg_turns:>7.1f} turns")
    print(f"Avg Search Latency:     {match_result.avg_move_ms:>7.1f} ms")
    print("=" * 72)
    print(f"Total time: {elapsed:.0f}s")
    print()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Matchup: Full Neural vs Priors Only")
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

    run_matchup(
        checkpoint_path=args.checkpoint,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        eval_depth=args.eval_depth,
        search_seed=args.search_seed,
    )


if __name__ == "__main__":
    main()
