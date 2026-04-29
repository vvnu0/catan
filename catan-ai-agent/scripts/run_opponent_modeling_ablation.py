"""Run proposal-aligned opponent-modeling ablations.

Compares:
  - none: plain single-world MCTS
  - frequency: one count-only determinized world
  - particle: multi-world conservation/dev-card determinization
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from catanatron.models.player import Color

from catan_ai.eval.arena import Arena, MatchResult
from catan_ai.eval.belief_diagnostics import BeliefDiagnosticsCollector
from catan_ai.eval.opponent_modeling import (
    OpponentModelEvalConfig,
    OpponentModelMode,
    make_opponent_model_player,
)

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO / "experiments" / "opponent_modeling_ablation"


def run_ablation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    games: int = 2,
    seeds: list[int] | None = None,
    base_seed: int = 7000,
    swap_seats: bool = True,
    search_cfg: OpponentModelEvalConfig | None = None,
) -> dict[str, Any]:
    """Run the ablation and write summary, match CSV, and diagnostics JSON."""
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_cfg = search_cfg or OpponentModelEvalConfig()
    eval_seeds = seeds if seeds is not None else [base_seed + i for i in range(games)]
    modes = [mode.value for mode in OpponentModelMode]
    diagnostics = BeliefDiagnosticsCollector(modes)

    match_specs = [
        (OpponentModelMode.FREQUENCY, OpponentModelMode.NONE),
        (OpponentModelMode.PARTICLE, OpponentModelMode.NONE),
        (OpponentModelMode.PARTICLE, OpponentModelMode.FREQUENCY),
    ]

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    for candidate_mode, baseline_mode in match_specs:
        result = _run_matchup(
            candidate_mode=candidate_mode,
            baseline_mode=baseline_mode,
            seeds=eval_seeds,
            swap_seats=swap_seats,
            search_cfg=search_cfg,
            diagnostics=diagnostics,
        )
        results.append(_matchup_summary(result, candidate_mode, baseline_mode, eval_seeds, swap_seats, search_cfg))

    diag_payload = {
        "experiment_name": "opponent_modeling_ablation",
        "modes": _mode_descriptions(),
        "diagnostics": diagnostics.snapshot(),
    }
    summary = {
        "experiment_name": "opponent_modeling_ablation",
        "modes_compared": modes,
        "search_budget": asdict(search_cfg),
        "seeds": eval_seeds,
        "games_per_matchup": len(eval_seeds) * (2 if swap_seats else 1),
        "swap_seats": swap_seats,
        "elapsed_sec": time.perf_counter() - t0,
        "eval_matchups": results,
        "artifact_paths": {
            "summary_path": _display_path(output_dir / "summary.json"),
            "matchups_csv": _display_path(output_dir / "matchups.csv"),
            "belief_diagnostics": _display_path(output_dir / "belief_diagnostics.json"),
        },
    }

    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "belief_diagnostics.json", diag_payload)
    _write_matchups_csv(output_dir / "matchups.csv", results)

    return summary


def _run_matchup(
    *,
    candidate_mode: OpponentModelMode,
    baseline_mode: OpponentModelMode,
    seeds: list[int],
    swap_seats: bool,
    search_cfg: OpponentModelEvalConfig,
    diagnostics: BeliefDiagnosticsCollector,
) -> MatchResult:
    aggregate = MatchResult(label=f"{candidate_mode.value} vs {baseline_mode.value}")

    def make_candidate(color: Color):
        player = make_opponent_model_player(candidate_mode, color, search_cfg)
        return diagnostics.track(candidate_mode.value, player)

    def make_baseline(color: Color):
        player = make_opponent_model_player(baseline_mode, color, search_cfg)
        return diagnostics.track(baseline_mode.value, player)

    for seed in seeds:
        arena = Arena(num_games=1, base_seed=seed, swap_seats=swap_seats)
        result = arena.compare(make_candidate, make_baseline, aggregate.label)
        _merge_result(aggregate, result)

    log.info(aggregate.summary())
    return aggregate


def _merge_result(target: MatchResult, source: MatchResult) -> None:
    target.games += source.games
    target.wins += source.wins
    target.losses += source.losses
    target.draws += source.draws
    target.turn_counts.extend(source.turn_counts)
    target.move_times.extend(source.move_times)
    target.final_vps.extend(source.final_vps)


def _matchup_summary(
    result: MatchResult,
    candidate_mode: OpponentModelMode,
    baseline_mode: OpponentModelMode,
    seeds: list[int],
    swap_seats: bool,
    search_cfg: OpponentModelEvalConfig,
) -> dict[str, Any]:
    return {
        "label": result.label,
        "candidate_mode": candidate_mode.value,
        "baseline_mode": baseline_mode.value,
        "games": result.games,
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "win_rate": result.win_rate,
        "avg_final_vp": result.avg_final_vp,
        "avg_turns": result.avg_turns,
        "mean_latency_ms": result.avg_move_ms,
        "seed_info": {
            "seeds": seeds,
            "swap_seats": swap_seats,
        },
        "search_budget": asdict(search_cfg),
    }


def _mode_descriptions() -> dict[str, str]:
    return {
        "none": "Plain single-world MCTS with no opponent belief intervention.",
        "frequency": "Single count-only determinized world from simple frequency/resource-count beliefs.",
        "particle": "Multi-world determinized belief sampling with conservation resources and dev-card sampling.",
    }


def _write_matchups_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "candidate_mode",
        "baseline_mode",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "avg_final_vp",
        "avg_turns",
        "mean_latency_ms",
        "seeds",
        "swap_seats",
        "total_simulations",
        "max_depth",
        "particle_worlds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "label": row["label"],
                "candidate_mode": row["candidate_mode"],
                "baseline_mode": row["baseline_mode"],
                "games": row["games"],
                "wins": row["wins"],
                "losses": row["losses"],
                "draws": row["draws"],
                "win_rate": row["win_rate"],
                "avg_final_vp": row["avg_final_vp"],
                "avg_turns": row["avg_turns"],
                "mean_latency_ms": row["mean_latency_ms"],
                "seeds": " ".join(str(s) for s in row["seed_info"]["seeds"]),
                "swap_seats": row["seed_info"]["swap_seats"],
                "total_simulations": row["search_budget"]["total_simulations"],
                "max_depth": row["search_budget"]["max_depth"],
                "particle_worlds": row["search_budget"]["particle_worlds"],
            })


def _parse_seeds(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run opponent-modeling ablation")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--games", type=int, default=2, help="Seeds to run if --seeds is omitted")
    parser.add_argument("--seeds", default=None, help="Comma-separated explicit game seeds")
    parser.add_argument("--base-seed", type=int, default=7000)
    parser.add_argument("--no-swap-seats", action="store_true")
    parser.add_argument("--sims", type=int, default=24, help="Total simulations per decision")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--particle-worlds", type=int, default=4)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--top-k-roads", type=int, default=3)
    parser.add_argument("--top-k-trades", type=int, default=2)
    parser.add_argument("--top-k-robber", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run_ablation(
        output_dir=args.output_dir,
        games=args.games,
        seeds=_parse_seeds(args.seeds),
        base_seed=args.base_seed,
        swap_seats=not args.no_swap_seats,
        search_cfg=OpponentModelEvalConfig(
            total_simulations=args.sims,
            max_depth=args.depth,
            search_seed=args.search_seed,
            particle_worlds=args.particle_worlds,
            top_k_roads=args.top_k_roads,
            top_k_trades=args.top_k_trades,
            top_k_robber=args.top_k_robber,
        ),
    )
    print(json.dumps(summary["artifact_paths"], indent=2))


if __name__ == "__main__":
    main()
