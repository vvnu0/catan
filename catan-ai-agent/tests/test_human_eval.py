from __future__ import annotations

import json

from scripts.finalize_human_eval_session import run_finalize
from scripts.prepare_human_eval import run_prepare
from scripts.start_human_eval_session import run_start
from scripts.summarize_human_eval import run_summarize


def test_human_eval_dry_run_workflow(tmp_path):
    manifest = run_prepare(
        participant_id="PTEST",
        skill_group="B",
        output_dir=tmp_path,
    )
    manifest_path = tmp_path / "sessions" / "PTEST" / "manifest.json"
    assert manifest_path.exists()
    assert manifest["main_bot"] == "frequency"

    checklist = run_start(manifest_path=manifest_path)
    assert (tmp_path / "sessions" / "PTEST" / "start_checklist.json").exists()
    assert len(checklist["games"]) == len(manifest["games"])

    completed = run_finalize(manifest_path=manifest_path, dry_run=True)
    assert (tmp_path / "sessions" / "PTEST" / "completed_session.json").exists()
    assert (tmp_path / "sessions" / "PTEST" / "results.json").exists()
    assert (tmp_path / "sessions" / "PTEST" / "surveys.json").exists()
    assert len(completed["results"]) == len(manifest["games"])
    assert len(completed["surveys"]) == len(manifest["games"])

    summary = run_summarize(output_dir=tmp_path)
    assert (tmp_path / "aggregate_summary.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "surveys.csv").exists()
    assert summary["participants"] == 1
    assert summary["count_by_skill_group"]["B"] == 1
    assert summary["games_completed"] == len(manifest["games"])

    loaded = json.loads((tmp_path / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert "main_bot_win_rate" in loaded
    assert "average_ratings_by_bot" in loaded
