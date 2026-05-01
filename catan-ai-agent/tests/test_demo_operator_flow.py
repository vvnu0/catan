from __future__ import annotations

import json

from catan_ai.eval.human_eval import load_simple_yaml, prepare_session
from scripts.check_human_eval_status import check_status
from scripts.demo_preflight import run_preflight
from scripts.run_final_demo import run_demo
from scripts.start_human_eval_session import run_start


def test_demo_operator_dry_run_flow(tmp_path):
    repo_root = tmp_path
    (repo_root / "reports" / "final_results").mkdir(parents=True)
    (repo_root / "reports" / "final_results" / "final_summary.json").write_text("{}", encoding="utf-8")
    (repo_root / "reports" / "final_results" / "FINAL_RESULTS.md").write_text("# Final", encoding="utf-8")
    (repo_root / "experiments" / "opponent_modeling_sweep" / "medium").mkdir(parents=True)
    (repo_root / "experiments" / "opponent_modeling_sweep" / "medium" / "aggregate_summary.json").write_text("{}", encoding="utf-8")
    (repo_root / "experiments" / "neural_scaling_study" / "medium").mkdir(parents=True)
    (repo_root / "experiments" / "neural_scaling_study" / "medium" / "aggregate_summary.json").write_text("{}", encoding="utf-8")
    (repo_root / "configs" / "human_eval").mkdir(parents=True)
    (repo_root / "configs" / "human_eval" / "default.yaml").write_text("main_bot: frequency\n", encoding="utf-8")

    preflight = run_preflight(repo_root=repo_root)
    assert (repo_root / "reports" / "final_results" / "demo_preflight.json").exists()
    assert preflight["ok"]

    demo_dir = tmp_path / "demo"
    demo = run_demo(output_dir=demo_dir, dry_run=True)
    assert (demo_dir / "demo_manifest.json").exists()
    assert (demo_dir / "demo_summary.md").exists()
    assert demo["status"] == "dry_run"

    cfg = load_simple_yaml(repo_root / "configs" / "human_eval" / "default.yaml")
    manifest = prepare_session(
        config=cfg,
        participant_id="PTEST",
        skill_group="A",
        output_dir=tmp_path / "human_eval",
    )
    manifest_path = tmp_path / "human_eval" / "sessions" / "PTEST" / "manifest.json"
    checklist = run_start(manifest_path=manifest_path)
    session_dir = tmp_path / "human_eval" / "sessions" / "PTEST"
    assert (session_dir / "session_commands.json").exists()
    assert (session_dir / "session_commands.md").exists()
    assert checklist["games"][0]["expected_result_file"]

    status = check_status(output_dir=tmp_path / "human_eval")
    assert (tmp_path / "human_eval" / "session_status.json").exists()
    assert status["counts"]["started"] == 1

    loaded = json.loads((tmp_path / "human_eval" / "session_status.json").read_text(encoding="utf-8"))
    assert loaded["session_count"] == 1
