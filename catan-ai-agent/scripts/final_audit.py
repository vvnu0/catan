#!/usr/bin/env python3
"""Final Audit — complete performance benchmark for the Catan AI project.

Runs a seeded tournament between NeuralMCTS (v3), FrequencyBeliefMCTS,
and HeuristicBot, then generates metrics, diagnostics, and figures.

Usage:
    python scripts/final_audit.py [--games 50] [--seed 9000]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Root imports in src/catan_ai
from catan_ai.eval.model_diagnostics import compute_model_diagnostics
from catan_ai.training.checkpoints import load_checkpoint

# Local script modules
from final_audit_core import (
    AggregateStats,
    InstrumentedArena,
    aggregate,
    make_player_factories,
)
from final_audit_figures import (
    figure1_breakthrough,
    figure2_efficiency_frontier,
    figure3_strategic_decisiveness,
    figure4_policy_confidence,
)

REPO = Path(__file__).resolve().parents[1]
CKPT_PATH = REPO / "experiments" / "neural_scaling_study" / "v3_combined" / "checkpoints" / "best.pt"
DATA_DIR = REPO / "experiments" / "neural_scaling_study" / "v3_combined" / "combined_data"
OUT_DIR = REPO / "reports" / "final_audit"

log = logging.getLogger(__name__)


def run_audit(num_games: int = 25, base_seed: int = 9000) -> None:
    """Main audit pipeline."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # ── 1. Load V3 model ─────────────────────────────────────────────
    print("=" * 60)
    print("FINAL AUDIT — Catan AI Performance Benchmark")
    print("=" * 60)
    print(f"\nLoading V3 checkpoint from {CKPT_PATH} ...")
    model, checkpoint = load_checkpoint(CKPT_PATH)
    print(f"  ✓ Loaded (epoch {checkpoint.get('epoch')})")

    # ── 2. Re-run diagnostics ────────────────────────────────────────
    print("\nRunning model diagnostics ...")
    diagnostics = compute_model_diagnostics(
        model=model,
        data_dir=str(DATA_DIR),
        checkpoint=checkpoint,
        checkpoint_path=str(CKPT_PATH),
        max_samples=512,
    )
    print(f"  flat_policy_fraction: {diagnostics['flat_policy_fraction']:.4f}")
    print(f"  value_mae:            {diagnostics['value_mae']:.4f}")
    print(f"  top1_match_rate:      {diagnostics['top1_match_rate']:.4f}")

    # ── 3. Create player factories ───────────────────────────────────
    from catan_ai.players.neural_mcts_player import NeuralMCTSConfig

    neural_cfg = NeuralMCTSConfig(
        max_simulations=30, max_depth=8, puct_c=1.5,
        top_k_roads=3, top_k_trades=2, top_k_robber=4,
        seed=2026, use_model_priors=True, use_model_value=True,
    )
    factories = make_player_factories(model, neural_cfg, search_seed=2026)

    # ── 4. Run tournament ────────────────────────────────────────────
    arena = InstrumentedArena(num_games=num_games, base_seed=base_seed,
                              swap_seats=True)
    effective_games = num_games * 2  # swap_seats doubles it

    print(f"\n▸ Running tournament ({effective_games} games per matchup) ...")

    print("\n  [1/3] NeuralMCTS vs FrequencyMCTS ...")
    neural_vs_freq = arena.run_matchup(
        factories["neural"], factories["frequency"],
        "NeuralMCTS vs FrequencyMCTS")
    neural_vs_freq_stats = aggregate(neural_vs_freq, "NeuralMCTS vs FrequencyMCTS")

    print("\n  [2/3] NeuralMCTS vs HeuristicBot ...")
    neural_vs_heur = arena.run_matchup(
        factories["neural"], factories["heuristic"],
        "NeuralMCTS vs HeuristicBot")
    neural_vs_heur_stats = aggregate(neural_vs_heur, "NeuralMCTS vs HeuristicBot")

    print("\n  [3/3] FrequencyMCTS vs HeuristicBot ...")
    freq_vs_heur = arena.run_matchup(
        factories["frequency"], factories["heuristic"],
        "FrequencyMCTS vs HeuristicBot")
    freq_vs_heur_stats = aggregate(freq_vs_heur, "FrequencyMCTS vs HeuristicBot")

    all_stats = {
        "neural_vs_freq": neural_vs_freq_stats,
        "neural_vs_heur": neural_vs_heur_stats,
        "freq_vs_heur": freq_vs_heur_stats,
    }

    # ── 5. Generate figures ──────────────────────────────────────────
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("\n▸ Generating figures ...")

    # Figure 1: v1 vs v3 breakthrough
    figure1_breakthrough(neural_vs_freq_stats.as_dict(), fig_dir)

    # Figure 2: Efficiency frontier (all agents vs frequency baseline)
    # For heuristic, estimate from freq_vs_heur (inverted perspective)
    # Get measured HeuristicBot latency from opponent_move_ms in neural_vs_heur
    heur_latency_values = [r.opponent_move_ms for r in neural_vs_heur if r.opponent_move_ms > 0]
    heur_avg_ms = (sum(heur_latency_values) / len(heur_latency_values)
                   if heur_latency_values else 0.1)
    agent_scatter = {
        "NeuralMCTS (v3)": {
            "avg_move_ms": neural_vs_freq_stats.avg_move_ms,
            "win_rate": neural_vs_freq_stats.win_rate,
        },
        "FrequencyMCTS": {
            "avg_move_ms": freq_vs_heur_stats.avg_move_ms,
            "win_rate": 0.50,  # baseline by definition (same agent vs itself)
        },
        "HeuristicBot": {
            "avg_move_ms": heur_avg_ms,
            "win_rate": 1.0 - freq_vs_heur_stats.win_rate,
        },
    }
    figure2_efficiency_frontier(agent_scatter, fig_dir)

    # Figure 3: Strategic decisiveness
    figure3_strategic_decisiveness(neural_vs_heur, fig_dir)

    # Figure 4: Policy confidence
    figure4_policy_confidence(model, str(DATA_DIR), fig_dir)

    # ── 6. Output summary ────────────────────────────────────────────
    elapsed = time.perf_counter() - t0

    # Save JSON
    summary = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "games_per_matchup": effective_games,
        "base_seed": base_seed,
        "elapsed_sec": round(elapsed, 1),
        "diagnostics": {
            "flat_policy_fraction": diagnostics["flat_policy_fraction"],
            "value_mae": diagnostics["value_mae"],
            "top1_match_rate": diagnostics["top1_match_rate"],
        },
        "matchups": {k: v.as_dict() for k, v in all_stats.items()},
    }
    json_path = OUT_DIR / "final_audit_results.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    # ── 7. Print Markdown table ──────────────────────────────────────
    _print_markdown_table(all_stats, diagnostics, elapsed, effective_games)


def _print_markdown_table(
    stats: dict[str, AggregateStats],
    diagnostics: dict,
    elapsed: float,
    games_per: int,
) -> None:
    """Print the final results as a Markdown table."""
    print("\n" + "=" * 70)
    print("FINAL AUDIT RESULTS")
    print("=" * 70)

    # Primary metrics table
    print("\n## Primary Metrics\n")
    print("| Matchup | Games | Win Rate | Avg VP | Latency (ms) | Avg Turns to Win |")
    print("|---------|-------|----------|--------|-------------|-----------------|")
    for key, s in stats.items():
        print(f"| {s.label} | {s.games} | {s.win_rate:.1%} | "
              f"{s.avg_vp:.1f} | {s.avg_move_ms:.1f} | {s.avg_turns_to_win:.0f} |")

    # Strategic intelligence table
    print("\n## Strategic Intelligence\n")
    print("| Matchup | Econ Efficiency | Spatial Mastery | Largest Army % | Longest Road % |")
    print("|---------|----------------|-----------------|---------------|---------------|")
    for key, s in stats.items():
        print(f"| {s.label} | {s.econ_efficiency:.2f} | "
              f"{s.spatial_mastery:.1f} pips | {s.largest_army_rate:.0%} | "
              f"{s.longest_road_rate:.0%} |")

    # Diagnostics
    print("\n## V3 Model Diagnostics\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Flat Policy Fraction | {diagnostics['flat_policy_fraction']:.4f} |")
    print(f"| Value MAE | {diagnostics['value_mae']:.4f} |")
    print(f"| Top-1 Match Rate | {diagnostics['top1_match_rate']:.4f} |")

    print(f"\n⏱  Total elapsed: {elapsed:.0f}s")
    print(f"📊 Results saved to: reports/final_audit/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Final audit for Catan AI")
    parser.add_argument("--games", type=int, default=25,
                        help="Games per matchup (doubled with seat-swap)")
    parser.add_argument("--seed", type=int, default=9000,
                        help="Base seed for reproducibility")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_audit(num_games=args.games, base_seed=args.seed)


if __name__ == "__main__":
    main()
