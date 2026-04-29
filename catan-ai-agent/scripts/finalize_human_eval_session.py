"""Finalize a human-evaluation session with game results and survey responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catan_ai.eval.human_eval import finalize_session, read_json


def run_finalize(
    *,
    manifest_path: str | Path,
    results_path: str | Path | None = None,
    surveys_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict:
    results = None
    surveys = None
    if results_path is not None:
        payload = read_json(results_path)
        results = payload.get("results", [])
    if surveys_path is not None:
        payload = read_json(surveys_path)
        surveys = payload.get("surveys", [])
    return finalize_session(
        manifest_path=manifest_path,
        results=results,
        surveys=surveys,
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize a human-eval session")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--surveys-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    completed = run_finalize(
        manifest_path=args.manifest,
        results_path=args.results_json,
        surveys_path=args.surveys_json,
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "completed_session_path": str(
            Path(read_json(args.manifest)["session_dir"]) / "completed_session.json"
        ),
        "participant_id": completed["participant_id"],
        "games_completed": len(completed["results"]),
        "surveys_completed": len(completed["surveys"]),
    }, indent=2))


if __name__ == "__main__":
    main()
