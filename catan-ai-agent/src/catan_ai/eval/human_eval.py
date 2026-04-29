"""File-based helpers for human evaluation sessions."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catan_ai.eval.survey import (
    DEFAULT_SURVEY_FIELDS,
    RATING_FIELDS,
    validate_result,
    validate_survey,
)


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line in {path}: {raw!r}")
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def prepare_session(
    *,
    config: dict[str, Any],
    participant_id: str,
    skill_group: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create and write a participant session manifest."""
    skill_group = skill_group.upper()
    if skill_group not in {"A", "B", "C"}:
        raise ValueError("skill_group must be A, B, or C")

    out = Path(output_dir or config.get("output_dir", "experiments/human_eval"))
    session_dir = out / "sessions" / participant_id
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "participant_id": participant_id,
        "skill_group": skill_group,
        "created_at": _now(),
        "status": "planned",
        "main_bot": config.get("main_bot", "frequency"),
        "reference_bot": config.get("reference_bot", "mcts"),
        "swap_order": bool(config.get("swap_order", True)),
        "output_dir": str(out),
        "session_dir": str(session_dir),
        "games": _build_game_plan(config, participant_id),
        "survey_fields": config.get("survey_fields", list(DEFAULT_SURVEY_FIELDS)),
        "operator_notes": [
            "Run games in listed order unless a participant needs a break.",
            "Record winner as human, bot, or draw.",
            "Record final VP and turns from the game/replay if available.",
        ],
    }
    write_json(session_dir / "manifest.json", manifest)
    return manifest


def start_session(manifest_path: str | Path) -> dict[str, Any]:
    """Return a printable operator checklist for a manifest."""
    manifest = read_json(manifest_path)
    checklist = {
        "participant_id": manifest["participant_id"],
        "skill_group": manifest["skill_group"],
        "status": "ready",
        "instructions": [
            "Open the local Catan play environment manually.",
            "For each game, configure the listed bot, seed, and human color.",
            "After each game, fill the result and survey fields in the finalization input.",
            "Do not reveal whether a game is main or reference until after the session if blinding is desired.",
        ],
        "games": [
            {
                "game_id": game["game_id"],
                "bot_faced": game["bot_faced"],
                "bot_role": game["bot_role"],
                "seed": game["seed"],
                "human_color": game["human_color"],
                "bot_color": game["bot_color"],
                "operator_checklist": [
                    f"Set game seed to {game['seed']}",
                    f"Seat participant as {game['human_color']}",
                    f"Seat bot as {game['bot_color']} using {game['bot_faced']}",
                    "Save or note any replay/log path before moving to the next game",
                ],
            }
            for game in manifest["games"]
        ],
    }
    session_dir = Path(manifest["session_dir"])
    write_json(session_dir / "start_checklist.json", checklist)
    return checklist


def finalize_session(
    *,
    manifest_path: str | Path,
    results: list[dict[str, Any]] | None = None,
    surveys: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate and write a completed session record."""
    manifest = read_json(manifest_path)
    if dry_run:
        results, surveys = make_dry_run_records(manifest)
    results = results or []
    surveys = surveys or []

    errors = []
    planned_ids = {game["game_id"] for game in manifest["games"]}
    for result in results:
        errors.extend(f"{result.get('game_id', '<unknown>')}: {e}" for e in validate_result(result))
        if result.get("game_id") not in planned_ids:
            errors.append(f"{result.get('game_id')}: result is not in manifest")
    for survey in surveys:
        errors.extend(f"{survey.get('game_id', '<unknown>')}: {e}" for e in validate_survey(survey))
        if survey.get("game_id") not in planned_ids:
            errors.append(f"{survey.get('game_id')}: survey is not in manifest")
    missing_results = sorted(planned_ids - {r.get("game_id") for r in results})
    missing_surveys = sorted(planned_ids - {s.get("game_id") for s in surveys})
    if missing_results:
        errors.append(f"missing result records for: {', '.join(missing_results)}")
    if missing_surveys:
        errors.append(f"missing survey records for: {', '.join(missing_surveys)}")
    if errors:
        raise ValueError("; ".join(errors))

    completed = {
        "participant_id": manifest["participant_id"],
        "skill_group": manifest["skill_group"],
        "completed_at": _now(),
        "status": "completed",
        "manifest_path": str(manifest_path),
        "results": results,
        "surveys": surveys,
    }
    session_dir = Path(manifest["session_dir"])
    write_json(session_dir / "completed_session.json", completed)
    write_json(session_dir / "results.json", {"results": results})
    write_json(session_dir / "surveys.json", {"surveys": surveys})
    return completed


def summarize_sessions(output_dir: str | Path) -> dict[str, Any]:
    """Aggregate completed session records under output_dir."""
    out = Path(output_dir)
    session_paths = sorted((out / "sessions").glob("*/completed_session.json"))
    completed = [read_json(path) for path in session_paths]
    manifests = sorted((out / "sessions").glob("*/manifest.json"))
    completed_participants = {s["participant_id"] for s in completed}
    incomplete = [
        read_json(path)["participant_id"]
        for path in manifests
        if read_json(path)["participant_id"] not in completed_participants
    ]

    result_rows = [row for session in completed for row in session["results"]]
    survey_rows = [row for session in completed for row in session["surveys"]]
    participants = {session["participant_id"]: session["skill_group"] for session in completed}

    summary = {
        "participants": len(participants),
        "count_by_skill_group": dict(Counter(participants.values())),
        "games_completed": len(result_rows),
        "main_bot_win_rate": _bot_win_rate(result_rows, "main"),
        "reference_bot_win_rate": _bot_win_rate(result_rows, "reference"),
        "average_ratings_by_bot": _average_ratings(survey_rows, "bot_faced"),
        "average_ratings_by_skill_group": _average_ratings(survey_rows, "skill_group"),
        "missing_or_incomplete_sessions": incomplete,
        "artifact_paths": {
            "aggregate_summary": str(out / "aggregate_summary.json"),
            "results_csv": str(out / "results.csv"),
            "surveys_csv": str(out / "surveys.csv"),
        },
    }
    write_json(out / "aggregate_summary.json", summary)
    write_csv(out / "results.csv", result_rows)
    write_csv(out / "surveys.csv", survey_rows)
    return summary


def make_dry_run_records(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    surveys = []
    for idx, game in enumerate(manifest["games"]):
        bot_won = idx % 2 == 0
        winner = "bot" if bot_won else "human"
        turns = 120 + idx * 7
        bot_vp = 10 if bot_won else 7
        human_vp = 7 if bot_won else 10
        results.append({
            "participant_id": manifest["participant_id"],
            "skill_group": manifest["skill_group"],
            "game_id": game["game_id"],
            "bot_faced": game["bot_faced"],
            "bot_role": game["bot_role"],
            "seed": game["seed"],
            "human_color": game["human_color"],
            "bot_color": game["bot_color"],
            "winner": winner,
            "bot_final_vp": bot_vp,
            "human_final_vp": human_vp,
            "turns": turns,
            "replay_path": "",
            "notes": "dry-run fixture",
        })
        surveys.append({
            "participant_id": manifest["participant_id"],
            "skill_group": manifest["skill_group"],
            "game_id": game["game_id"],
            "bot_faced": game["bot_faced"],
            "seed": game["seed"],
            "winner": winner,
            "final_vp": human_vp,
            "turns": turns,
            "moves_felt_sensible": 4 if game["bot_role"] == "main" else 3,
            "gameplay_felt_fair": 4,
            "challenge_level": 4 if bot_won else 3,
            "comments": "dry-run response",
        })
    return results, surveys


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_game_plan(config: dict[str, Any], participant_id: str) -> list[dict[str, Any]]:
    seeds = list(config.get("seeds", [9100, 9101, 9102, 9103]))
    main_count = int(config.get("games_per_participant_main", 1))
    reference_count = int(config.get("games_per_participant_reference", 1))
    specs = [("main", config.get("main_bot", "frequency"))] * main_count
    specs += [("reference", config.get("reference_bot", "mcts"))] * reference_count
    if bool(config.get("swap_order", True)):
        specs = _interleave_main_reference(specs)

    games = []
    for idx, (role, bot) in enumerate(specs):
        human_is_red = idx % 2 == 0
        games.append({
            "game_id": f"{participant_id}_game_{idx + 1:02d}",
            "bot_role": role,
            "bot_faced": bot,
            "seed": int(seeds[idx % len(seeds)]),
            "human_color": "RED" if human_is_red else "BLUE",
            "bot_color": "BLUE" if human_is_red else "RED",
            "status": "planned",
        })
    return games


def _interleave_main_reference(specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    main = [s for s in specs if s[0] == "main"]
    reference = [s for s in specs if s[0] == "reference"]
    result = []
    while main or reference:
        if main:
            result.append(main.pop(0))
        if reference:
            result.append(reference.pop(0))
    return result


def _bot_win_rate(rows: list[dict[str, Any]], role: str) -> float:
    filtered = [r for r in rows if r.get("bot_role") == role]
    if not filtered:
        return 0.0
    return sum(1 for r in filtered if r.get("winner") == "bot") / len(filtered)


def _average_ratings(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, ""))].append(row)
    return {
        bucket: {
            field: _mean([float(row[field]) for row in bucket_rows if field in row])
            for field in RATING_FIELDS
        }
        for bucket, bucket_rows in buckets.items()
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
