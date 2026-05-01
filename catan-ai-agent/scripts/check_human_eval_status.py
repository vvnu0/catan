"""Report prepared/started/completed human-eval session status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO / "experiments" / "human_eval"


def check_status(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    out = Path(output_dir)
    sessions_dir = out / "sessions"
    rows = []
    if sessions_dir.exists():
        for manifest_path in sorted(sessions_dir.glob("*/manifest.json")):
            rows.append(_session_status(manifest_path))
    counts = {
        "prepared_only": sum(1 for row in rows if row["status"] == "prepared_only"),
        "started": sum(1 for row in rows if row["status"] == "started"),
        "completed": sum(1 for row in rows if row["status"] == "completed"),
        "missing_results": sum(1 for row in rows if row["missing_results"]),
        "missing_surveys": sum(1 for row in rows if row["missing_surveys"]),
    }
    summary = {
        "sessions_dir": str(sessions_dir),
        "session_count": len(rows),
        "counts": counts,
        "sessions": rows,
        "artifact_path": str(out / "session_status.json"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "session_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _session_status(manifest_path: Path) -> dict[str, Any]:
    session_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    started = (session_dir / "start_checklist.json").exists() or (session_dir / "session_commands.json").exists()
    completed = (session_dir / "completed_session.json").exists()
    results = (session_dir / "results.json").exists()
    surveys = (session_dir / "surveys.json").exists()
    if completed:
        status = "completed"
    elif started:
        status = "started"
    else:
        status = "prepared_only"
    return {
        "participant_id": manifest.get("participant_id"),
        "skill_group": manifest.get("skill_group"),
        "status": status,
        "session_dir": str(session_dir),
        "manifest_path": str(manifest_path),
        "start_checklist_exists": (session_dir / "start_checklist.json").exists(),
        "session_commands_exists": (session_dir / "session_commands.json").exists(),
        "completed_session_exists": completed,
        "missing_results": not results,
        "missing_surveys": not surveys,
        "planned_games": len(manifest.get("games", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check human-eval session status")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = check_status(output_dir=args.output_dir)
    print(json.dumps({
        "session_count": summary["session_count"],
        "counts": summary["counts"],
        "artifact_path": summary["artifact_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
