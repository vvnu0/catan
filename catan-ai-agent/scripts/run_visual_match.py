"""Launch/save browser-inspectable custom-bot matches via Catanatron DB."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CATANATRON_REPO = REPO.parent / "catanatron" / "catanatron"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(CATANATRON_REPO) not in sys.path:
    sys.path.insert(0, str(CATANATRON_REPO))

from catanatron import Color, Game
from catanatron.web.database_accumulator import StepDatabaseAccumulator

from catan_ai.eval.opponent_modeling import (
    OpponentModelEvalConfig,
    OpponentModelMode,
    make_opponent_model_player,
)
from catan_ai.players.heuristic_player import HeuristicBot

DEMO_SEED = 9900
UI_URL = "http://localhost:3000"


def run_visual_match(
    *,
    matchup: str = "main_vs_reference",
    seed: int = DEMO_SEED,
    dry_run: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir) if output_dir else REPO / "experiments" / "visual_matches" / _timestamp()
    out.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(matchup=matchup, seed=seed, output_dir=out, dry_run=dry_run)
    _write_json(out / "visual_match_manifest.json", manifest)

    if dry_run:
        summary = {
            "status": "dry_run",
            "manifest_path": _display(out / "visual_match_manifest.json"),
            "instructions": "Start Catanatron web DB/server/UI, then rerun without --dry-run.",
        }
        _write_json(out / "visual_match_summary.json", summary)
        return summary

    accumulator = StepDatabaseAccumulator()
    game = Game(_players_for_matchup(matchup), seed=seed)
    winner = game.play(accumulators=[accumulator])
    replay_url = getattr(accumulator, "link", f"{UI_URL}/replays/{game.id}")
    summary = {
        "status": "completed",
        "matchup": matchup,
        "seed": seed,
        "winner": winner.value if winner else None,
        "turns": game.state.num_turns,
        "game_id": game.id,
        "browser_url": replay_url,
        "manifest_path": _display(out / "visual_match_manifest.json"),
    }
    _write_json(out / "visual_match_summary.json", summary)
    (out / "visual_match_summary.md").write_text(_summary_md(summary), encoding="utf-8")
    return summary


def _players_for_matchup(matchup: str):
    cfg = OpponentModelEvalConfig(total_simulations=16, max_depth=8, search_seed=2026)
    main = make_opponent_model_player(OpponentModelMode.FREQUENCY, Color.RED, cfg)
    if matchup == "main_vs_reference":
        opponent = make_opponent_model_player(OpponentModelMode.NONE, Color.BLUE, cfg)
    elif matchup == "main_vs_heuristic":
        opponent = HeuristicBot(Color.BLUE)
    else:
        raise ValueError(f"Unsupported visual matchup: {matchup}")
    return [main, opponent]


def _manifest(*, matchup: str, seed: int, output_dir: Path, dry_run: bool) -> dict[str, Any]:
    return {
        "matchup": matchup,
        "seed": seed,
        "dry_run": dry_run,
        "players": ["MAIN_BOT", "REFERENCE_BOT" if matchup == "main_vs_reference" else "HEURISTIC"],
        "database_url": os.environ.get("DATABASE_URL", "postgresql://catanatron:victorypoint@127.0.0.1:5432/catanatron_db"),
        "requires": [
            "Catanatron postgres/web stack running",
            "catan-ai-agent importable by the Python process",
        ],
        "output_dir": str(output_dir),
    }


def _summary_md(summary: dict[str, Any]) -> str:
    return "\n".join([
        "# Visual Match",
        "",
        f"- Status: `{summary['status']}`",
        f"- Matchup: `{summary['matchup']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Winner: `{summary['winner']}`",
        f"- Turns: `{summary['turns']}`",
        f"- Browser URL: {summary['browser_url']}",
        "",
    ])


def _timestamp() -> str:
    return datetime.now().strftime("visual_%Y%m%d_%H%M%S")


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a browser-inspectable custom bot match")
    parser.add_argument("--matchup", choices=("main_vs_reference", "main_vs_heuristic"), default="main_vs_reference")
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    print(json.dumps(run_visual_match(
        matchup=args.matchup,
        seed=args.seed,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
