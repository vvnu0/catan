"""Build a consolidated final-results bundle from existing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_DIR = REPO / "experiments"
DEFAULT_REPORTS_DIR = REPO / "reports"
DEFAULT_OUTPUT_DIR = DEFAULT_REPORTS_DIR / "final_results"


def build_final_results_bundle(
    *,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Read available experiment outputs and write the final-results bundle."""
    experiments_dir = Path(experiments_dir)
    reports_dir = Path(reports_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _load_artifacts(experiments_dir, reports_dir)
    summary = _build_summary(artifacts, experiments_dir, reports_dir, output_dir)
    main_rows = _build_main_rows(artifacts)
    appendix_rows = _build_appendix_rows(artifacts)

    _write_json(output_dir / "final_summary.json", summary)
    _write_csv(output_dir / "main_results_table.csv", main_rows)
    _write_csv(output_dir / "appendix_results_table.csv", appendix_rows)
    (output_dir / "FINAL_RESULTS.md").write_text(
        _build_markdown(summary, main_rows, appendix_rows),
        encoding="utf-8",
    )
    (output_dir / "SUBMISSION_CHECKLIST.md").write_text(
        _build_checklist(summary),
        encoding="utf-8",
    )
    return summary


def _load_artifacts(experiments_dir: Path, reports_dir: Path) -> dict[str, Any]:
    return {
        "opponent_sweep_medium": _read_optional_json(
            experiments_dir / "opponent_modeling_sweep" / "medium" / "aggregate_summary.json"
        ),
        "opponent_sweep_small": _read_optional_json(
            experiments_dir / "opponent_modeling_sweep" / "small" / "aggregate_summary.json"
        ),
        "neural_scaling_medium": _read_optional_json(
            experiments_dir / "neural_scaling_study" / "medium" / "aggregate_summary.json"
        ),
        "neural_scaling_small": _read_optional_json(
            experiments_dir / "neural_scaling_study" / "small" / "aggregate_summary.json"
        ),
        "human_eval": _read_optional_json(experiments_dir / "human_eval" / "aggregate_summary.json"),
        "human_sessions": _read_human_sessions(experiments_dir / "human_eval" / "sessions"),
        "mini_neural": _read_optional_json(experiments_dir / "mini_neural_exp" / "summary.json")
        or _read_optional_json(reports_dir / "mini_neural_experiment_summary.json"),
        "opponent_ablation": _read_optional_json(
            experiments_dir / "opponent_modeling_ablation" / "summary.json"
        ),
    }


def _build_summary(
    artifacts: dict[str, Any],
    experiments_dir: Path,
    reports_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    sweep = artifacts["opponent_sweep_medium"] or artifacts["opponent_sweep_small"]
    neural = artifacts["neural_scaling_medium"] or artifacts["neural_scaling_small"]
    human = artifacts["human_eval"]
    human_status = _human_status(human, artifacts["human_sessions"])
    use_human_metrics = human_status["status"] == "real_or_mixed_data_present"

    return {
        "chosen_main_bot": "frequency belief MCTS",
        "chosen_reference_bot": "plain MCTS",
        "opponent_modeling_conclusion": _opponent_conclusion(sweep),
        "neural_conclusion": _neural_conclusion(neural),
        "human_eval_status": human_status,
        "main_headline_metrics": {
            "frequency_vs_none_win_rate": _avg_matchup_win_rate(
                sweep, candidate="frequency", baseline="none"
            ),
            "particle_world_scaled_vs_none_win_rate": _avg_matchup_win_rate(
                sweep, candidate="particle", baseline="none", regime="world_scaled"
            ),
            "neural_vs_frequency_win_rate": _nested_get(
                neural,
                ["aggregate_metrics", "win_rate_vs_frequency_mcts", "mean"],
            ),
            "human_main_bot_win_rate": (
                human.get("main_bot_win_rate") if human and use_human_metrics else None
            ),
            "human_reference_bot_win_rate": (
                human.get("reference_bot_win_rate") if human and use_human_metrics else None
            ),
        },
        "key_artifact_paths": {
            "final_summary": _display(output_dir / "final_summary.json"),
            "main_results_table": _display(output_dir / "main_results_table.csv"),
            "appendix_results_table": _display(output_dir / "appendix_results_table.csv"),
            "final_results_markdown": _display(output_dir / "FINAL_RESULTS.md"),
            "submission_checklist": _display(output_dir / "SUBMISSION_CHECKLIST.md"),
            "opponent_sweep_medium": _display(
                experiments_dir / "opponent_modeling_sweep" / "medium" / "aggregate_summary.json"
            ),
            "neural_scaling_medium": _display(
                experiments_dir / "neural_scaling_study" / "medium" / "aggregate_summary.json"
            ),
            "human_eval_summary": _display(experiments_dir / "human_eval" / "aggregate_summary.json"),
            "mini_neural_summary": _display(experiments_dir / "mini_neural_exp" / "summary.json"),
            "neural_audit": _display(reports_dir / "neural_phase_audit.md"),
        },
        "source_artifact_presence": {
            key: bool(value)
            for key, value in artifacts.items()
            if key != "human_sessions"
        },
    }


def _build_main_rows(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sweep = artifacts["opponent_sweep_medium"] or artifacts["opponent_sweep_small"]
    if sweep:
        rows.extend(_matchup_rows(
            sweep,
            family="opponent_modeling",
            include=lambda r: r.get("candidate_mode") == "frequency" and r.get("baseline_mode") == "none",
            notes="Main technical result: frequency belief MCTS selected as final bot.",
        ))

    human = artifacts["human_eval"]
    if human and _human_status(human, artifacts["human_sessions"])["status"] == "real_or_mixed_data_present":
        rows.append({
            "family": "human_eval",
            "experiment": "human_eval",
            "comparison": "main bot vs participants",
            "regime": "",
            "games": human.get("games_completed"),
            "win_rate": human.get("main_bot_win_rate"),
            "avg_final_vp": "",
            "avg_turns": "",
            "metric": "main_bot_win_rate",
            "value": human.get("main_bot_win_rate"),
            "artifact_path": "experiments/human_eval/aggregate_summary.json",
            "notes": "Human evaluation aggregate from completed non-dry-run sessions.",
        })
    return rows


def _build_appendix_rows(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sweep = artifacts["opponent_sweep_medium"] or artifacts["opponent_sweep_small"]
    if sweep:
        rows.extend(_matchup_rows(
            sweep,
            family="particle_appendix",
            include=lambda r: r.get("candidate_mode") == "particle"
            and r.get("regime") == "world_scaled",
            notes="Appendix particle setting; not the default final bot.",
        ))

    neural = artifacts["neural_scaling_medium"] or artifacts["neural_scaling_small"]
    if neural:
        metrics = neural.get("aggregate_metrics", {})
        for metric_name in (
            "win_rate_vs_frequency_mcts",
            "flat_policy_fraction",
            "top1_match_rate",
            "value_mae",
        ):
            metric = metrics.get(metric_name)
            if metric is None:
                continue
            rows.append({
                "family": "neural_scaling",
                "experiment": f"neural_scaling_{neural.get('preset', 'unknown')}",
                "comparison": "neural_frequency_mcts vs frequency_mcts",
                "regime": "",
                "games": "",
                "win_rate": metric.get("mean") if metric_name == "win_rate_vs_frequency_mcts" else "",
                "avg_final_vp": "",
                "avg_turns": "",
                "metric": metric_name,
                "value": metric.get("mean"),
                "artifact_path": f"experiments/neural_scaling_study/{neural.get('preset', 'medium')}/aggregate_summary.json",
                "notes": "Secondary/negative-result learning track.",
            })

    mini = artifacts["mini_neural"]
    if mini:
        for matchup in mini.get("eval_matchups", []):
            if "NeuralMCTS" not in matchup.get("label", ""):
                continue
            rows.append(_row_from_matchup(
                matchup,
                family="mini_neural",
                experiment="mini_neural_exp",
                artifact_path="experiments/mini_neural_exp/summary.json",
                notes="Earlier mini neural checkpoint benchmark.",
            ))
    return rows


def _matchup_rows(
    artifact: dict[str, Any],
    *,
    family: str,
    include,
    notes: str,
) -> list[dict[str, Any]]:
    rows = []
    for matchup in artifact.get("matchups", artifact.get("eval_matchups", [])):
        if include(matchup):
            rows.append(_row_from_matchup(
                matchup,
                family=family,
                experiment=artifact.get("experiment_name", family),
                artifact_path=_artifact_path_for(artifact),
                notes=notes,
            ))
    return rows


def _row_from_matchup(
    matchup: dict[str, Any],
    *,
    family: str,
    experiment: str,
    artifact_path: str,
    notes: str,
) -> dict[str, Any]:
    comparison = matchup.get("label") or (
        f"{matchup.get('candidate_mode')} vs {matchup.get('baseline_mode')}"
    )
    return {
        "family": family,
        "experiment": experiment,
        "comparison": comparison,
        "regime": matchup.get("regime", ""),
        "games": matchup.get("games", ""),
        "win_rate": matchup.get("win_rate", ""),
        "avg_final_vp": matchup.get("avg_final_vp", ""),
        "avg_turns": matchup.get("avg_turns", ""),
        "metric": "win_rate",
        "value": matchup.get("win_rate", ""),
        "artifact_path": artifact_path,
        "notes": notes,
    }


def _build_markdown(
    summary: dict[str, Any],
    main_rows: list[dict[str, Any]],
    appendix_rows: list[dict[str, Any]],
) -> str:
    metrics = summary["main_headline_metrics"]
    human = summary["human_eval_status"]
    lines = [
        "# Final Results",
        "",
        "## Final System Choice",
        "",
        f"- Main bot: `{summary['chosen_main_bot']}`.",
        f"- Reference bot: `{summary['chosen_reference_bot']}`.",
        "- Particle `world_scaled` remains an appendix/experimental variant.",
        "- Neural guidance remains a secondary/negative-result track for now.",
        "",
        "## Main Findings",
        "",
        f"- Opponent modeling: {summary['opponent_modeling_conclusion']}",
        f"- Frequency vs none mean win rate: `{_fmt(metrics.get('frequency_vs_none_win_rate'))}`.",
        "",
        "## Secondary Findings",
        "",
        f"- Particle world-scaled vs none mean win rate: `{_fmt(metrics.get('particle_world_scaled_vs_none_win_rate'))}`.",
        f"- Neural vs frequency mean win rate: `{_fmt(metrics.get('neural_vs_frequency_win_rate'))}`.",
        f"- Neural conclusion: {summary['neural_conclusion']}",
        "",
        "## Human Evaluation Status",
        "",
        f"- Status: `{human['status']}`.",
        f"- Participants in current aggregate: `{human.get('participants', 0)}`.",
        f"- Note: {human['note']}",
        "",
        "## Limitations",
        "",
        "- Evidence is strongest for the scripted bot-vs-bot experiments already present in `experiments/`.",
        "- Human-evaluation tooling is ready, but real participant data should replace dry-run data before final claims.",
        "- Neural guidance is not presented as a positive main result because current scaling runs did not beat frequency MCTS.",
        "",
        "## Reproducibility Notes",
        "",
        "- Rebuild this bundle with `python scripts/build_final_results_bundle.py`.",
        "- Validate expected artifacts with `python scripts/reproduce_final_artifacts.py`.",
        "- Main rows are in `main_results_table.csv`; appendix rows are in `appendix_results_table.csv`.",
        "",
        "## Table Counts",
        "",
        f"- Main result rows: `{len(main_rows)}`.",
        f"- Appendix result rows: `{len(appendix_rows)}`.",
        "",
    ]
    return "\n".join(lines)


def _build_checklist(summary: dict[str, Any]) -> str:
    presence = summary["source_artifact_presence"]
    human = summary["human_eval_status"]
    checked = lambda ok: "x" if ok else " "
    return "\n".join([
        "# Submission Checklist",
        "",
        f"- [{checked(True)}] Main bot selected: frequency belief MCTS.",
        f"- [{checked(bool(presence.get('opponent_sweep_medium') or presence.get('opponent_sweep_small')))}] Opponent-modeling evidence generated.",
        f"- [{checked(bool(presence.get('neural_scaling_medium') or presence.get('neural_scaling_small')))}] Neural scaling study generated.",
        f"- [{checked(bool(presence.get('human_eval')))}] Human-eval toolkit/aggregate artifacts present.",
        f"- [{checked(human['status'] == 'real_or_mixed_data_present')}] Real human sessions run.",
        f"- [{checked(True)}] Final results bundle built.",
        "- [ ] Final README/report prose reviewed for submission.",
        "- [ ] Demo command verified on the submission machine.",
        "",
    ])


def _opponent_conclusion(sweep: dict[str, Any] | None) -> str:
    if not sweep:
        return "Opponent-modeling sweep artifact missing; final choice uses prior project decision."
    rec = sweep.get("recommendation", {})
    metrics = rec.get("metrics", {})
    freq = metrics.get("frequency_vs_none_avg_win_rate")
    particle_scaled = metrics.get("particle_vs_none_world_scaled_avg_win_rate")
    return (
        "Frequency belief MCTS is the default final setting; "
        f"frequency-vs-none average win rate={_fmt(freq)}. "
        "Particle compute-matched is not a main setting; "
        f"world-scaled particle is appendix-only (particle-vs-none average win rate={_fmt(particle_scaled)})."
    )


def _neural_conclusion(neural: dict[str, Any] | None) -> str:
    if not neural:
        return "Neural scaling artifact missing; neural guidance is not used as the main bot."
    rec = neural.get("recommendation", {})
    metrics = rec.get("metrics", {})
    return (
        "Neural guidance is kept as a secondary/negative-result track; "
        f"mean win rate vs frequency MCTS={_fmt(metrics.get('mean_win_rate_vs_frequency_mcts'))}, "
        f"top-1 match={_fmt(metrics.get('mean_top1_match_rate'))}, "
        f"flat-policy fraction={_fmt(metrics.get('mean_flat_policy_fraction'))}."
    )


def _human_status(human: dict[str, Any] | None, sessions: list[dict[str, Any]]) -> dict[str, Any]:
    if not human:
        return {
            "status": "not_run",
            "participants": 0,
            "note": "No human-eval aggregate summary found.",
        }
    dry_run_sessions = [
        session for session in sessions
        if _session_looks_dry_run(session)
    ]
    if sessions and len(dry_run_sessions) == len(sessions):
        status = "dry_run_only"
        note = "Only dry-run/fixture human-eval sessions are present; do not claim real participant results yet."
    else:
        status = "real_or_mixed_data_present"
        note = "At least one completed session does not look like dry-run fixture data."
    return {
        "status": status,
        "participants": human.get("participants", 0),
        "games_completed": human.get("games_completed", 0),
        "main_bot_win_rate": human.get("main_bot_win_rate"),
        "reference_bot_win_rate": human.get("reference_bot_win_rate"),
        "missing_or_incomplete_sessions": human.get("missing_or_incomplete_sessions", []),
        "note": note,
    }


def _session_looks_dry_run(session: dict[str, Any]) -> bool:
    if str(session.get("participant_id", "")).upper().endswith("DRYRUN"):
        return True
    results = session.get("results", [])
    return bool(results) and all("dry-run" in str(r.get("notes", "")).lower() for r in results)


def _avg_matchup_win_rate(
    artifact: dict[str, Any] | None,
    *,
    candidate: str,
    baseline: str,
    regime: str | None = None,
) -> float | None:
    if not artifact:
        return None
    rows = [
        r for r in artifact.get("matchups", artifact.get("eval_matchups", []))
        if r.get("candidate_mode") == candidate
        and r.get("baseline_mode") == baseline
        and (regime is None or r.get("regime") == regime)
    ]
    if not rows:
        return None
    return sum(float(r["win_rate"]) for r in rows) / len(rows)


def _nested_get(payload: dict[str, Any] | None, keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _artifact_path_for(artifact: dict[str, Any]) -> str:
    name = artifact.get("experiment_name", "")
    preset = artifact.get("preset")
    if name == "opponent_modeling_sweep" and preset:
        return f"experiments/opponent_modeling_sweep/{preset}/aggregate_summary.json"
    if name == "opponent_modeling_ablation":
        return "experiments/opponent_modeling_ablation/summary.json"
    return "experiments"


def _read_human_sessions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        payload for payload in (
            _read_optional_json(session_path)
            for session_path in sorted(path.glob("*/completed_session.json"))
        )
        if payload
    ]


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "experiment",
        "comparison",
        "regime",
        "games",
        "win_rate",
        "avg_final_vp",
        "avg_turns",
        "metric",
        "value",
        "artifact_path",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final project results bundle")
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build_final_results_bundle(
        experiments_dir=args.experiments_dir,
        reports_dir=args.reports_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary["key_artifact_paths"], indent=2))
    print(json.dumps({
        "chosen_main_bot": summary["chosen_main_bot"],
        "human_eval_status": summary["human_eval_status"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
