#!/usr/bin/env python3
"""Regenerate the three LaTeX evaluation tables with fresh numbers.

Table 1 — Baseline Search Sanity Check
       MCTSPlayer vs DebugPlayer / HeuristicBot / RandomPlayer (20 games each)

Table 2 — Opponent Modeling Results
       freq/part vs none/freq across worlds={2,4} × regimes={compute_matched,world_scaled}

Table 3 — Neural Guidance Results
       Self-play sample count, diagnostics, and NeuralMCTS evaluation matchups.

Usage:
    # Run everything (slow — plays hundreds of games):
    python scripts/regenerate_latex_tables.py

    # Run only Table 1:
    python scripts/regenerate_latex_tables.py --table 1

    # Run only Table 2:
    python scripts/regenerate_latex_tables.py --table 2

    # Run only Table 3 (reuses existing shards/checkpoint if present):
    python scripts/regenerate_latex_tables.py --table 3

    # Adjust game counts:
    python scripts/regenerate_latex_tables.py --t1-games 10 --t2-games 4 --t3-games 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from catanatron import Color, Game, RandomPlayer
from catanatron.models.player import Player

from catan_ai.eval.arena import Arena, MatchResult
from catan_ai.players import DebugPlayer, HeuristicBot, MCTSPlayer

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]

# Ensure sibling scripts are importable when run from the repo root.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


# ═════════════════════════════════════════════════════════════════════
# Table 1 — Baseline Search Sanity Check
# ═════════════════════════════════════════════════════════════════════

MCTS_SIMS = 50
MCTS_DEPTH = 10
TOP_K_ROADS = 3
TOP_K_TRADES = 2
TOP_K_ROBBER = 4


def _make_mcts(color: Color) -> MCTSPlayer:
    return MCTSPlayer(
        color,
        max_simulations=MCTS_SIMS,
        max_depth=MCTS_DEPTH,
        top_k_roads=TOP_K_ROADS,
        top_k_trades=TOP_K_TRADES,
        top_k_robber=TOP_K_ROBBER,
        seed=0,
    )


def run_table1(num_games: int = 20, base_seed: int = 5000) -> list[MatchResult]:
    """Play MCTSPlayer vs three baselines and return MatchResults."""
    arena = Arena(num_games=num_games, base_seed=base_seed, swap_seats=False)

    matchups = [
        ("MCTSPlayer vs DebugPlayer", lambda c: DebugPlayer(c)),
        ("MCTSPlayer vs HeuristicBot", lambda c: HeuristicBot(c)),
        ("MCTSPlayer vs RandomPlayer", lambda c: RandomPlayer(c)),
    ]
    results: list[MatchResult] = []
    for label, make_baseline in matchups:
        print(f"  Running {label} ({num_games} games) ...")
        r = arena.compare(_make_mcts, make_baseline, label)
        results.append(r)
        print(f"    {r.summary()}")
    return results


def format_table1_latex(results: list[MatchResult]) -> str:
    """Format Table 1 results as LaTeX tabular rows."""
    lines = []
    for r in results:
        record = f"{r.wins}W / {r.losses}L / {r.draws}D"
        wr = f"{r.win_rate * 100:.0f}\\%"
        avg_turns = f"{r.avg_turns:.1f}"
        latency = f"{r.avg_move_ms:.1f} ms"
        lines.append(
            f"{r.label} & {r.games} & {record} & {wr} & {avg_turns} & {latency} \\\\"
        )
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Table 2 — Opponent Modeling Results
# ═════════════════════════════════════════════════════════════════════

def run_table2(
    num_games: int = 8,
    base_seed: int = 7000,
    sims: int = 16,
    world_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run the opponent-modeling sweep and return per-setting rows."""
    from catan_ai.eval.opponent_modeling import (
        OpponentModelEvalConfig,
        OpponentModelMode,
        make_opponent_model_player,
    )

    if world_counts is None:
        world_counts = [2, 4]

    seeds = [base_seed + i for i in range(num_games)]
    regimes = ("compute_matched", "world_scaled")

    specs = [
        (OpponentModelMode.FREQUENCY, OpponentModelMode.NONE, "freq vs none"),
        (OpponentModelMode.PARTICLE, OpponentModelMode.NONE, "part vs none"),
        (OpponentModelMode.PARTICLE, OpponentModelMode.FREQUENCY, "part vs freq"),
    ]

    rows: list[dict[str, Any]] = []

    for worlds in world_counts:
        for regime in regimes:
            for cand_mode, base_mode, label in specs:
                # Determine sim budget per the regime
                base_cfg = OpponentModelEvalConfig(
                    total_simulations=sims,
                    particle_worlds=worlds,
                )
                if regime == "compute_matched":
                    cand_cfg = OpponentModelEvalConfig(
                        total_simulations=sims,
                        particle_worlds=worlds,
                    )
                else:  # world_scaled
                    scaled_sims = sims * worlds if cand_mode == OpponentModelMode.PARTICLE else sims
                    cand_cfg = OpponentModelEvalConfig(
                        total_simulations=scaled_sims,
                        particle_worlds=worlds,
                    )

                setting = f"{label} / {regime} / w={worlds}"
                print(f"  Running {setting} ({len(seeds)*2} games) ...")

                aggregate = MatchResult(label=label)
                for seed in seeds:
                    arena = Arena(num_games=1, base_seed=seed, swap_seats=True)

                    def make_cand(color: Color, _cfg=cand_cfg, _mode=cand_mode):
                        return make_opponent_model_player(_mode, color, _cfg)

                    def make_base(color: Color, _cfg=base_cfg, _mode=base_mode):
                        return make_opponent_model_player(_mode, color, _cfg)

                    r = arena.compare(make_cand, make_base, label)
                    # Merge into aggregate
                    aggregate.games += r.games
                    aggregate.wins += r.wins
                    aggregate.losses += r.losses
                    aggregate.draws += r.draws
                    aggregate.turn_counts.extend(r.turn_counts)
                    aggregate.move_times.extend(r.move_times)
                    aggregate.final_vps.extend(r.final_vps)

                actual_sims = cand_cfg.total_simulations
                row = {
                    "label": label,
                    "regime": regime,
                    "worlds": worlds,
                    "sims": actual_sims,
                    "games": aggregate.games,
                    "wins": aggregate.wins,
                    "losses": aggregate.losses,
                    "draws": aggregate.draws,
                    "win_rate": aggregate.win_rate,
                    "avg_vp": aggregate.avg_final_vp,
                    "avg_turns": aggregate.avg_turns,
                    "latency_ms": aggregate.avg_move_ms,
                }
                rows.append(row)
                print(f"    {aggregate.summary()}")

    return rows


def format_table2_latex(rows: list[dict[str, Any]]) -> str:
    """Format Table 2 results as LaTeX tabular rows."""
    lines = []
    prev_worlds = None
    for r in rows:
        # Insert \addlinespace between world count groups
        group_key = (r["worlds"], r["regime"])
        if prev_worlds is not None and r["worlds"] != prev_worlds:
            lines.append("\\addlinespace")
        # Also add spacing between regime groups within same worlds
        if prev_worlds == r["worlds"] and r["label"] == "freq vs none" and lines and not lines[-1].startswith("\\addlinespace"):
            lines.append("\\addlinespace")
        prev_worlds = r["worlds"]

        regime_esc = r["regime"].replace("_", "\\_")
        record = f"{r['wins']}W / {r['losses']}L / {r['draws']}D"
        lines.append(
            f"{r['label']} & {regime_esc} & {r['worlds']} & {r['sims']} & "
            f"{r['games']} & {record} & {r['win_rate']:.3f} & "
            f"{r['avg_vp']:.2f} & {r['avg_turns']:.1f} & "
            f"{r['latency_ms']:.2f} ms \\\\"
        )
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Table 3 — Neural Guidance Results
# ═════════════════════════════════════════════════════════════════════

def run_table3(
    num_eval_games: int = 2,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """Run the mini neural experiment and collect diagnostics.

    Returns a dict with all metrics needed for Table 3.
    """
    # ── Try to load existing scaling study results first ──
    scaling_path = REPO / "experiments" / "neural_scaling_study" / "medium" / "aggregate_summary.json"
    scaling_data = None
    if scaling_path.exists():
        scaling_data = json.loads(scaling_path.read_text(encoding="utf-8"))
        print(f"  Loaded neural scaling study from {scaling_path}")

    # ── Try to load existing mini neural experiment summary ──
    mini_summary_candidates = [
        REPO / "experiments" / "mini_neural_exp" / "summary.json",
        REPO / "reports" / "mini_neural_experiment_summary.json",
    ]
    summary = None
    for p in mini_summary_candidates:
        if p.exists():
            summary = json.loads(p.read_text(encoding="utf-8"))
            print(f"  Loaded existing mini neural summary from {p}")
            break

    # ── Re-run only if no prior results exist ──
    if summary is None:
        try:
            from run_mini_neural_experiment import run_experiment

            print(f"  Running mini neural experiment (eval games={num_eval_games}) ...")
            summary = run_experiment(reuse_existing=reuse_existing)
            print(f"  Done. Elapsed: {summary['elapsed_sec']:.1f}s")
        except Exception as exc:
            print(f"  ⚠ Mini neural experiment failed: {exc}")
            print("    Continuing with scaling-study data only.")
            summary = {}

    # ── Collect diagnostics from scaling study (preferred) or fallback ──
    if scaling_data:
        agg = scaling_data.get("aggregate_metrics", {})
        self_play_samples = scaling_data.get("total_samples", 0)
        neural_freq_wr = agg.get("win_rate_vs_frequency_mcts", {}).get("mean", 0.0)
        neural_freq_vp = agg.get("avg_final_vp_vs_frequency_mcts", {}).get("mean", 0.0)
        top1_match = agg.get("top1_match_rate", {}).get("mean", 0.0)
        flat_frac = agg.get("flat_policy_fraction", {}).get("mean", 0.0)
        value_mae = agg.get("value_mae", {}).get("mean", 0.0)
    else:
        self_play_samples = summary.get("self_play_samples", 0)
        neural_freq_wr = 0.0
        neural_freq_vp = 0.0
        top1_match = 0.0
        flat_frac = 0.0
        value_mae = 0.0

    # ── Pull mini NeuralMCTS eval matchup results ──
    mini_vs_mcts_wr = 0.0
    mini_vs_heur_wr = 0.0
    for m in summary.get("eval_matchups", []):
        if "vs MCTS" in m["label"] and "NeuralMCTS" in m["label"]:
            mini_vs_mcts_wr = m["win_rate"]
        if "vs HeuristicBot" in m["label"] and "NeuralMCTS" in m["label"]:
            mini_vs_heur_wr = m["win_rate"]

    return {
        "self_play_samples": self_play_samples,
        "neural_freq_mcts_wr": neural_freq_wr,
        "neural_freq_mcts_vp": neural_freq_vp,
        "top1_match_rate": top1_match,
        "flat_policy_fraction": flat_frac,
        "value_mae": value_mae,
        "mini_vs_mcts_wr": mini_vs_mcts_wr,
        "mini_vs_heur_wr": mini_vs_heur_wr,
    }


def format_table3_latex(metrics: dict[str, Any]) -> str:
    """Format Table 3 results as LaTeX tabular rows."""
    rows = [
        ("Self-play samples", f"{metrics['self_play_samples']}"),
        ("Neural frequency MCTS win rate vs frequency MCTS",
         f"{metrics['neural_freq_mcts_wr']:.3f}"),
        ("Neural frequency MCTS avg final VP",
         f"{metrics['neural_freq_mcts_vp']:.2f}"),
        ("Top-1 policy target match rate",
         f"{metrics['top1_match_rate']:.3f}"),
        ("Flat policy fraction",
         f"{metrics['flat_policy_fraction']:.3f}"),
        ("Value MAE",
         f"{metrics['value_mae']:.3f}"),
        ("Mini NeuralMCTS checkpoint vs MCTS win rate",
         f"{metrics['mini_vs_mcts_wr']:.3f}"),
        ("Mini NeuralMCTS checkpoint vs HeuristicBot win rate",
         f"{metrics['mini_vs_heur_wr']:.3f}"),
    ]
    return "\n".join(f"{name} & {val} \\\\" for name, val in rows)


# ═════════════════════════════════════════════════════════════════════
# JSON output
# ═════════════════════════════════════════════════════════════════════

def save_results_json(
    output_path: Path,
    t1_results: list[MatchResult] | None,
    t2_rows: list[dict[str, Any]] | None,
    t3_metrics: dict[str, Any] | None,
) -> None:
    """Save all table data to a single JSON file."""
    payload: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    if t1_results is not None:
        payload["table1"] = [
            {
                "label": r.label,
                "games": r.games,
                "wins": r.wins,
                "losses": r.losses,
                "draws": r.draws,
                "win_rate": r.win_rate,
                "avg_turns": r.avg_turns,
                "avg_move_ms": r.avg_move_ms,
            }
            for r in t1_results
        ]
    if t2_rows is not None:
        payload["table2"] = t2_rows
    if t3_metrics is not None:
        payload["table3"] = t3_metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n📄 JSON saved to {output_path}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate LaTeX evaluation tables with fresh experiment data."
    )
    parser.add_argument(
        "--table", type=int, choices=[1, 2, 3], default=None,
        help="Run only the specified table (1, 2, or 3). Default: run all.",
    )
    parser.add_argument("--t1-games", type=int, default=20,
                        help="Games per matchup for Table 1 (default: 20)")
    parser.add_argument("--t2-games", type=int, default=8,
                        help="Seeds (games) for Table 2; actual games = 2× this (default: 8)")
    parser.add_argument("--t3-games", type=int, default=2,
                        help="Eval games per matchup for Table 3 (default: 2)")
    parser.add_argument("--output", type=str,
                        default=str(REPO / "reports" / "latex_table_data.json"),
                        help="Path to write JSON results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    run_all = args.table is None
    t0 = time.perf_counter()
    t1_results = None
    t2_rows = None
    t3_metrics = None

    # ── Table 1 ──────────────────────────────────────────────────────
    if run_all or args.table == 1:
        print("\n" + "=" * 70)
        print("TABLE 1 — Baseline Search Sanity Check")
        print("=" * 70)
        t1_results = run_table1(num_games=args.t1_games)
        print("\n--- LaTeX rows ---")
        print(format_table1_latex(t1_results))

    # ── Table 2 ──────────────────────────────────────────────────────
    if run_all or args.table == 2:
        print("\n" + "=" * 70)
        print("TABLE 2 — Opponent Modeling Results")
        print("=" * 70)
        t2_rows = run_table2(num_games=args.t2_games)
        print("\n--- LaTeX rows ---")
        print(format_table2_latex(t2_rows))

    # ── Table 3 ──────────────────────────────────────────────────────
    if run_all or args.table == 3:
        print("\n" + "=" * 70)
        print("TABLE 3 — Neural Guidance Results")
        print("=" * 70)
        t3_metrics = run_table3(num_eval_games=args.t3_games)
        print("\n--- LaTeX rows ---")
        print(format_table3_latex(t3_metrics))

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    save_results_json(Path(args.output), t1_results, t2_rows, t3_metrics)
    print(f"\n⏱  Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
