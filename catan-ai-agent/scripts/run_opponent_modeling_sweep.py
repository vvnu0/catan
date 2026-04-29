"""Run an opponent-modeling sweep across seeds, budgets, and particle regimes."""

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
CONFIG_DIR = REPO / "configs" / "opponent_modeling_sweep"
REGIMES = ("compute_matched", "world_scaled")


def run_sweep(
    *,
    preset: str = "small",
    output_dir: str | Path | None = None,
    seeds: list[int] | None = None,
    sims_per_move: list[int] | None = None,
    particle_world_counts: list[int] | None = None,
    games: int | None = None,
) -> dict[str, Any]:
    """Run a sweep preset and write per-setting plus aggregate artifacts."""
    cfg = _load_preset(preset)
    if seeds is not None:
        cfg["seeds"] = seeds
    elif games is not None:
        first = int(cfg.get("seeds", [7000])[0])
        cfg["seeds"] = [first + i for i in range(games)]
    if sims_per_move is not None:
        cfg["sims_per_move"] = sims_per_move
    if particle_world_counts is not None:
        cfg["particle_world_counts"] = particle_world_counts
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)

    out = _resolve_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    setting_summaries: list[dict[str, Any]] = []
    all_diagnostics: dict[str, Any] = {}
    t0 = time.perf_counter()

    for sims in cfg["sims_per_move"]:
        for worlds in cfg["particle_world_counts"]:
            for regime in REGIMES:
                setting_dir = out / f"sims_{sims}" / f"worlds_{worlds}" / regime
                setting = _run_setting(
                    cfg=cfg,
                    sims=int(sims),
                    particle_worlds=int(worlds),
                    regime=regime,
                    output_dir=setting_dir,
                )
                rows.extend(setting["matchups"])
                setting_summaries.append(setting["summary"])
                all_diagnostics[setting["setting_id"]] = setting["diagnostics"]

    aggregate = {
        "experiment_name": "opponent_modeling_sweep",
        "preset": cfg["preset"],
        "seeds": cfg["seeds"],
        "games_per_matchup": len(cfg["seeds"]) * 2,
        "sims_per_move": cfg["sims_per_move"],
        "particle_world_counts": cfg["particle_world_counts"],
        "regimes": list(REGIMES),
        "elapsed_sec": time.perf_counter() - t0,
        "matchups": rows,
        "setting_summaries": setting_summaries,
        "belief_diagnostics": all_diagnostics,
        "recommendation": _recommend(rows),
        "artifact_paths": {
            "aggregate_summary": _display_path(out / "aggregate_summary.json"),
            "matchups_csv": _display_path(out / "matchups.csv"),
            "belief_diagnostics": _display_path(out / "belief_diagnostics.json"),
        },
    }
    _write_json(out / "aggregate_summary.json", aggregate)
    _write_json(out / "belief_diagnostics.json", all_diagnostics)
    _write_matchups_csv(out / "matchups.csv", rows)
    return aggregate


def _run_setting(
    *,
    cfg: dict[str, Any],
    sims: int,
    particle_worlds: int,
    regime: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    setting_id = f"sims_{sims}_worlds_{particle_worlds}_{regime}"
    diagnostics = BeliefDiagnosticsCollector([mode.value for mode in OpponentModelMode])
    seeds = list(cfg["seeds"])
    base_cfg = _eval_config(cfg, sims=sims, particle_worlds=particle_worlds)
    particle_sims = sims if regime == "compute_matched" else sims * particle_worlds
    particle_cfg = _eval_config(cfg, sims=particle_sims, particle_worlds=particle_worlds)

    specs = [
        (OpponentModelMode.FREQUENCY, OpponentModelMode.NONE, base_cfg, base_cfg),
        (OpponentModelMode.PARTICLE, OpponentModelMode.NONE, particle_cfg, base_cfg),
        (OpponentModelMode.PARTICLE, OpponentModelMode.FREQUENCY, particle_cfg, base_cfg),
    ]

    matchups = []
    for candidate_mode, baseline_mode, candidate_cfg, baseline_cfg in specs:
        result = _run_matchup(
            candidate_mode=candidate_mode,
            baseline_mode=baseline_mode,
            candidate_cfg=candidate_cfg,
            baseline_cfg=baseline_cfg,
            seeds=seeds,
            diagnostics=diagnostics,
        )
        matchups.append(
            _matchup_summary(
                result=result,
                candidate_mode=candidate_mode,
                baseline_mode=baseline_mode,
                candidate_cfg=candidate_cfg,
                baseline_cfg=baseline_cfg,
                seeds=seeds,
                regime=regime,
                sims=sims,
                particle_worlds=particle_worlds,
                setting_id=setting_id,
            )
        )

    diag_payload = diagnostics.snapshot()
    summary = {
        "setting_id": setting_id,
        "regime": regime,
        "sims_per_move": sims,
        "particle_worlds": particle_worlds,
        "seeds": seeds,
        "games_per_matchup": len(seeds) * 2,
        "matchups": matchups,
        "diagnostics": diag_payload,
        "artifact_paths": {
            "summary": _display_path(output_dir / "summary.json"),
            "matchups_csv": _display_path(output_dir / "matchups.csv"),
            "belief_diagnostics": _display_path(output_dir / "belief_diagnostics.json"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "belief_diagnostics.json", diag_payload)
    _write_matchups_csv(output_dir / "matchups.csv", matchups)
    return {
        "setting_id": setting_id,
        "summary": summary,
        "matchups": matchups,
        "diagnostics": diag_payload,
    }


def _run_matchup(
    *,
    candidate_mode: OpponentModelMode,
    baseline_mode: OpponentModelMode,
    candidate_cfg: OpponentModelEvalConfig,
    baseline_cfg: OpponentModelEvalConfig,
    seeds: list[int],
    diagnostics: BeliefDiagnosticsCollector,
) -> MatchResult:
    aggregate = MatchResult(label=f"{candidate_mode.value} vs {baseline_mode.value}")

    def make_candidate(color: Color):
        player = make_opponent_model_player(candidate_mode, color, candidate_cfg)
        return diagnostics.track(candidate_mode.value, player)

    def make_baseline(color: Color):
        player = make_opponent_model_player(baseline_mode, color, baseline_cfg)
        return diagnostics.track(baseline_mode.value, player)

    for seed in seeds:
        arena = Arena(num_games=1, base_seed=seed, swap_seats=True)
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


def _eval_config(
    cfg: dict[str, Any],
    *,
    sims: int,
    particle_worlds: int,
) -> OpponentModelEvalConfig:
    return OpponentModelEvalConfig(
        total_simulations=sims,
        max_depth=int(cfg.get("max_depth", 8)),
        search_seed=int(cfg.get("search_seed", 2026)),
        particle_worlds=particle_worlds,
        top_k_roads=int(cfg.get("top_k_roads", 3)),
        top_k_trades=int(cfg.get("top_k_trades", 2)),
        top_k_robber=int(cfg.get("top_k_robber", 4)),
        enable_particle_devcards=bool(cfg.get("enable_particle_devcards", True)),
    )


def _matchup_summary(
    *,
    result: MatchResult,
    candidate_mode: OpponentModelMode,
    baseline_mode: OpponentModelMode,
    candidate_cfg: OpponentModelEvalConfig,
    baseline_cfg: OpponentModelEvalConfig,
    seeds: list[int],
    regime: str,
    sims: int,
    particle_worlds: int,
    setting_id: str,
) -> dict[str, Any]:
    return {
        "setting_id": setting_id,
        "regime": regime,
        "sims_per_move": sims,
        "particle_worlds": particle_worlds,
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
            "swap_seats": True,
        },
        "candidate_search_budget": asdict(candidate_cfg),
        "baseline_search_budget": asdict(baseline_cfg),
    }


def _recommend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    freq = _avg_win(rows, "frequency", "none")
    particle_compute = _avg_win(rows, "particle", "none", "compute_matched")
    particle_scaled = _avg_win(rows, "particle", "none", "world_scaled")
    particle_vs_freq_compute = _avg_win(rows, "particle", "frequency", "compute_matched")
    particle_vs_freq_scaled = _avg_win(rows, "particle", "frequency", "world_scaled")

    flags: list[str] = []
    if freq >= 0.55:
        flags.append("keep_frequency")
    elif freq >= 0.45:
        flags.append("keep_frequency_as_neutral")
    else:
        flags.append("keep_none")

    if particle_scaled >= 0.55 and particle_vs_freq_scaled >= 0.5:
        flags.append("keep_particle_world_scaled")
    elif particle_scaled - particle_compute >= 0.15:
        flags.append("particle_needs_more_budget")
    elif particle_compute < 0.45 and particle_scaled < 0.45:
        flags.append("drop_particle")
    else:
        flags.append("particle_inconclusive")

    primary = "keep_frequency" if "keep_frequency" in flags else flags[0]
    if "drop_particle" in flags and primary == "keep_none":
        primary = "keep_none_drop_particle"

    return {
        "primary": primary,
        "flags": flags,
        "metrics": {
            "frequency_vs_none_avg_win_rate": freq,
            "particle_vs_none_compute_matched_avg_win_rate": particle_compute,
            "particle_vs_none_world_scaled_avg_win_rate": particle_scaled,
            "particle_vs_frequency_compute_matched_avg_win_rate": particle_vs_freq_compute,
            "particle_vs_frequency_world_scaled_avg_win_rate": particle_vs_freq_scaled,
        },
    }


def _avg_win(
    rows: list[dict[str, Any]],
    candidate: str,
    baseline: str,
    regime: str | None = None,
) -> float:
    filtered = [
        r for r in rows
        if r["candidate_mode"] == candidate
        and r["baseline_mode"] == baseline
        and (regime is None or r["regime"] == regime)
    ]
    if not filtered:
        return 0.0
    return sum(float(r["win_rate"]) for r in filtered) / len(filtered)


def _write_matchups_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "setting_id",
        "regime",
        "sims_per_move",
        "particle_worlds",
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
        "candidate_total_simulations",
        "baseline_total_simulations",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "setting_id": row["setting_id"],
                "regime": row["regime"],
                "sims_per_move": row["sims_per_move"],
                "particle_worlds": row["particle_worlds"],
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
                "candidate_total_simulations": row["candidate_search_budget"]["total_simulations"],
                "baseline_total_simulations": row["baseline_search_budget"]["total_simulations"],
            })


def _load_preset(preset: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{preset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown sweep preset {preset!r}: {path}")
    return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


def _parse_csv_ints(value: str | None) -> list[int] | None:
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
    parser = argparse.ArgumentParser(description="Run opponent-modeling sweep")
    parser.add_argument("--preset", choices=("small", "medium"), default="small")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated explicit seeds")
    parser.add_argument("--games", type=int, default=None, help="Generate this many seeds")
    parser.add_argument("--sims", default=None, help="Comma-separated sims per move")
    parser.add_argument("--particle-worlds", default=None, help="Comma-separated world counts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    summary = run_sweep(
        preset=args.preset,
        output_dir=args.output_dir,
        seeds=_parse_csv_ints(args.seeds),
        games=args.games,
        sims_per_move=_parse_csv_ints(args.sims),
        particle_world_counts=_parse_csv_ints(args.particle_worlds),
    )
    print(json.dumps(summary["artifact_paths"], indent=2))
    print(json.dumps(summary["recommendation"], indent=2))


if __name__ == "__main__":
    main()
