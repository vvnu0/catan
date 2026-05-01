"""Run or dry-run the canonical final demo match."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from catan_ai.eval.arena import Arena, MatchResult
from catan_ai.eval.opponent_modeling import (
    OpponentModelEvalConfig,
    OpponentModelMode,
    make_opponent_model_player,
)
from catan_ai.players.heuristic_player import HeuristicBot

DEMO_SEED = 9900


def run_demo(
    *,
    matchup: str = "main_vs_reference",
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    games: int = 1,
) -> dict[str, Any]:
    run_name = datetime.now().strftime("demo_%Y%m%d_%H%M%S")
    out = Path(output_dir) if output_dir is not None else REPO / "experiments" / "demo_runs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    manifest = _demo_manifest(matchup=matchup, output_dir=out, dry_run=dry_run, games=games)
    _write_json(out / "demo_manifest.json", manifest)

    if dry_run:
        summary = {
            "status": "dry_run",
            "manifest_path": _display(out / "demo_manifest.json"),
            "summary_path": _display(out / "demo_summary.md"),
            "planned_matchup": matchup,
            "planned_games": games * 2,
        }
        (out / "demo_summary.md").write_text(_demo_markdown(manifest, None, summary), encoding="utf-8")
        _write_json(out / "demo_result.json", summary)
        return summary

    result = _run_matchup(matchup=matchup, games=games)
    row = _match_result_row(result, matchup)
    _write_json(out / "demo_result.json", {"status": "completed", "result": row})
    _write_csv(out / "demo_result.csv", [row])
    summary = {
        "status": "completed",
        "manifest_path": _display(out / "demo_manifest.json"),
        "result_json": _display(out / "demo_result.json"),
        "result_csv": _display(out / "demo_result.csv"),
        "summary_path": _display(out / "demo_summary.md"),
        "result": row,
    }
    (out / "demo_summary.md").write_text(_demo_markdown(manifest, row, summary), encoding="utf-8")
    return summary


def _run_matchup(*, matchup: str, games: int) -> MatchResult:
    cfg = OpponentModelEvalConfig(total_simulations=16, max_depth=8, search_seed=2026)
    arena = Arena(num_games=games, base_seed=DEMO_SEED, swap_seats=True)

    def make_main(color):
        return make_opponent_model_player(OpponentModelMode.FREQUENCY, color, cfg)

    if matchup == "main_vs_reference":
        def make_reference(color):
            return make_opponent_model_player(OpponentModelMode.NONE, color, cfg)
        return arena.compare(make_main, make_reference, "frequency_mcts vs plain_mcts demo")

    if matchup == "main_vs_heuristic":
        return arena.compare(make_main, lambda color: HeuristicBot(color), "frequency_mcts vs heuristic demo")

    raise ValueError(f"Unsupported demo matchup: {matchup}")


def _demo_manifest(*, matchup: str, output_dir: Path, dry_run: bool, games: int) -> dict[str, Any]:
    if matchup == "main_vs_reference":
        baseline = "plain_mcts"
    elif matchup == "main_vs_heuristic":
        baseline = "heuristic"
    else:
        raise ValueError(f"Unsupported demo matchup: {matchup}")
    return {
        "matchup": matchup,
        "main_bot": "frequency belief MCTS",
        "baseline_bot": baseline,
        "seed": DEMO_SEED,
        "games_per_seat": games,
        "swap_seats": True,
        "search_budget": {
            "total_simulations": 16,
            "max_depth": 8,
            "belief_mode": "frequency/count_only",
        },
        "dry_run": dry_run,
        "output_dir": str(output_dir),
        "expected_artifacts": {
            "manifest": str(output_dir / "demo_manifest.json"),
            "result_json": str(output_dir / "demo_result.json"),
            "result_csv": str(output_dir / "demo_result.csv"),
            "summary": str(output_dir / "demo_summary.md"),
        },
    }


def _match_result_row(result: MatchResult, matchup: str) -> dict[str, Any]:
    return {
        "matchup": matchup,
        "label": result.label,
        "games": result.games,
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "win_rate": result.win_rate,
        "avg_final_vp": result.avg_final_vp,
        "avg_turns": result.avg_turns,
        "mean_latency_ms": result.avg_move_ms,
    }


def _demo_markdown(manifest: dict[str, Any], row: dict[str, Any] | None, summary: dict[str, Any]) -> str:
    lines = [
        "# Final Demo Run",
        "",
        f"- Status: `{summary['status']}`",
        f"- Matchup: `{manifest['matchup']}`",
        f"- Main bot: `{manifest['main_bot']}`",
        f"- Baseline bot: `{manifest['baseline_bot']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Swap seats: `{manifest['swap_seats']}`",
        "",
    ]
    if row:
        lines.extend([
            "## Result",
            "",
            f"- Games: `{row['games']}`",
            f"- Main bot record: `{row['wins']}W / {row['losses']}L / {row['draws']}D`",
            f"- Win rate: `{row['win_rate']:.3f}`",
            f"- Avg final VP: `{row['avg_final_vp']:.2f}`",
            f"- Avg turns: `{row['avg_turns']:.2f}`",
            "",
        ])
    else:
        lines.extend([
            "## Dry Run",
            "",
            "No game was played. This file records the exact demo plan and output locations.",
            "",
        ])
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical final demo")
    parser.add_argument("--matchup", choices=("main_vs_reference", "main_vs_heuristic"), default="main_vs_reference")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()
    summary = run_demo(
        matchup=args.matchup,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        games=args.games,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
