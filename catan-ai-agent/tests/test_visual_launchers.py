from __future__ import annotations

from scripts.run_browser_human_vs_bot import create_human_vs_bot
from scripts.run_visual_match import run_visual_match


def test_visual_match_dry_run_manifest(tmp_path):
    summary = run_visual_match(output_dir=tmp_path, dry_run=True)

    assert summary["status"] == "dry_run"
    assert (tmp_path / "visual_match_manifest.json").exists()
    assert (tmp_path / "visual_match_summary.json").exists()


def test_human_vs_bot_dry_run_request(tmp_path):
    plan = create_human_vs_bot(output_dir=tmp_path, dry_run=True, bot="MAIN_BOT")

    assert plan["status"] == "dry_run"
    assert plan["request_payload"]["players"] == ["HUMAN", "MAIN_BOT"]
    assert plan["request_payload"]["seed"] == 9910
    assert (tmp_path / "human_vs_bot_plan.json").exists()
