from __future__ import annotations

import json

from scripts.build_final_results_bundle import build_final_results_bundle


def test_final_results_bundle_smoke(tmp_path):
    experiments = tmp_path / "experiments"
    reports = tmp_path / "reports"

    sweep_dir = experiments / "opponent_modeling_sweep" / "medium"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate_summary.json").write_text(json.dumps({
        "experiment_name": "opponent_modeling_sweep",
        "preset": "medium",
        "matchups": [
            {
                "label": "frequency vs none",
                "candidate_mode": "frequency",
                "baseline_mode": "none",
                "regime": "compute_matched",
                "games": 4,
                "win_rate": 0.75,
                "avg_final_vp": 7.5,
                "avg_turns": 150,
            },
            {
                "label": "particle vs none",
                "candidate_mode": "particle",
                "baseline_mode": "none",
                "regime": "world_scaled",
                "games": 4,
                "win_rate": 0.5,
                "avg_final_vp": 6.0,
                "avg_turns": 180,
            },
        ],
        "recommendation": {
            "metrics": {
                "frequency_vs_none_avg_win_rate": 0.75,
                "particle_vs_none_world_scaled_avg_win_rate": 0.5,
            }
        },
    }), encoding="utf-8")

    neural_dir = experiments / "neural_scaling_study" / "medium"
    neural_dir.mkdir(parents=True)
    (neural_dir / "aggregate_summary.json").write_text(json.dumps({
        "experiment_name": "neural_scaling_study",
        "preset": "medium",
        "aggregate_metrics": {
            "win_rate_vs_frequency_mcts": {"mean": 0.0, "std": 0.0},
            "flat_policy_fraction": {"mean": 0.7, "std": 0.1},
            "top1_match_rate": {"mean": 0.45, "std": 0.02},
            "value_mae": {"mean": 0.8, "std": 0.1},
        },
        "recommendation": {
            "metrics": {
                "mean_win_rate_vs_frequency_mcts": 0.0,
                "mean_top1_match_rate": 0.45,
                "mean_flat_policy_fraction": 0.7,
            }
        },
    }), encoding="utf-8")

    out = reports / "final_results"
    summary = build_final_results_bundle(
        experiments_dir=experiments,
        reports_dir=reports,
        output_dir=out,
    )

    assert (out / "final_summary.json").exists()
    assert (out / "main_results_table.csv").exists()
    assert (out / "FINAL_RESULTS.md").exists()
    assert summary["chosen_main_bot"] == "frequency belief MCTS"

    loaded = json.loads((out / "final_summary.json").read_text(encoding="utf-8"))
    assert loaded["main_headline_metrics"]["frequency_vs_none_win_rate"] == 0.75
