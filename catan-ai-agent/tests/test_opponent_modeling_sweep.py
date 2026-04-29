from __future__ import annotations

import json

from catan_ai.eval.arena import MatchResult
from scripts.run_opponent_modeling_sweep import run_sweep


def test_opponent_modeling_sweep_tiny_smoke(tmp_path, monkeypatch):
    def fake_matchup(**kwargs):
        candidate_mode = kwargs["candidate_mode"]
        baseline_mode = kwargs["baseline_mode"]
        result = MatchResult(label=f"{candidate_mode.value} vs {baseline_mode.value}")
        result.games = 2
        result.wins = 1
        result.losses = 1
        result.turn_counts = [10, 12]
        result.final_vps = [6, 7]
        result.move_times = [1.0, 1.2]
        return result

    monkeypatch.setattr(
        "scripts.run_opponent_modeling_sweep._run_matchup",
        fake_matchup,
    )

    summary = run_sweep(
        preset="small",
        output_dir=tmp_path,
        seeds=[1234],
        sims_per_move=[1],
        particle_world_counts=[1],
    )

    aggregate_path = tmp_path / "aggregate_summary.json"
    csv_path = tmp_path / "matchups.csv"
    diagnostics_path = tmp_path / "belief_diagnostics.json"

    assert tmp_path.exists()
    assert aggregate_path.exists()
    assert csv_path.exists()
    assert diagnostics_path.exists()

    loaded = json.loads(aggregate_path.read_text(encoding="utf-8"))
    regimes = {row["regime"] for row in loaded["matchups"]}
    assert regimes == {"compute_matched", "world_scaled"}
    assert loaded["recommendation"]["primary"]
    assert summary["artifact_paths"]["aggregate_summary"]

    per_setting_csvs = list(tmp_path.glob("sims_*/*/*/matchups.csv"))
    assert per_setting_csvs
