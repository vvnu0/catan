from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.run_neural_scaling_study import run_study


def test_neural_scaling_study_tiny_smoke(tmp_path, monkeypatch):
    def fake_self_play(cfg):
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        sample = {
            "state_feats": torch.zeros(53),
            "action_feats": torch.zeros(2, 19),
            "target_policy": torch.tensor([1.0, 0.0]),
            "target_value": torch.tensor(1.0),
            "encoded_actions": ["RED:ROLL", "RED:END_TURN"],
            "chosen_action": "RED:ROLL",
            "meta": {"seed": cfg.seed},
        }
        torch.save([sample], out / "shard_0000.pt")
        return out

    def fake_train(cfg):
        ckpt = Path(cfg.checkpoint_dir) / "best.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"fake")
        (ckpt.parent / "training_history.json").write_text("[]", encoding="utf-8")
        return ckpt

    def fake_load_checkpoint(path):
        return object(), {"epoch": 1, "metrics": {"val_total": 1.0}}

    def fake_diagnostics(**_kwargs):
        return {
            "self_play_sample_count": 1,
            "diagnostic_sample_count": 1,
            "mean_policy_entropy": 0.1,
            "mean_normalized_policy_entropy": 0.2,
            "flat_policy_fraction": 0.0,
            "nonflat_policy_fraction": 1.0,
            "top1_match_rate": 1.0,
            "value_mae": 0.1,
            "value_mse": 0.01,
        }

    def fake_eval(_cfg, *, seed, model):
        return [
            {
                "seed": seed,
                "label": "frequency_mcts vs frequency_mcts",
                "games": 2,
                "wins": 1,
                "losses": 1,
                "draws": 0,
                "win_rate": 0.5,
                "avg_final_vp": 7.0,
                "avg_turns": 100.0,
                "mean_latency_ms": 1.0,
                "belief_mode": "frequency",
                "search_budget": {"eval_sims": 1, "eval_depth": 1},
            },
            {
                "seed": seed,
                "label": "neural_frequency_mcts vs frequency_mcts",
                "games": 2,
                "wins": 1,
                "losses": 1,
                "draws": 0,
                "win_rate": 0.5,
                "avg_final_vp": 7.5,
                "avg_turns": 90.0,
                "mean_latency_ms": 1.2,
                "belief_mode": "frequency",
                "search_budget": {"eval_sims": 1, "eval_depth": 1},
            },
        ]

    monkeypatch.setattr("scripts.run_neural_scaling_study.run_self_play", fake_self_play)
    monkeypatch.setattr("scripts.run_neural_scaling_study.train", fake_train)
    monkeypatch.setattr("scripts.run_neural_scaling_study.load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        "scripts.run_neural_scaling_study.compute_model_diagnostics",
        fake_diagnostics,
    )
    monkeypatch.setattr("scripts.run_neural_scaling_study._run_eval", fake_eval)

    summary = run_study(preset="small", output_dir=tmp_path, seeds=[1234])

    seed_dir = tmp_path / "seed_1234"
    assert seed_dir.exists()
    assert (seed_dir / "checkpoints" / "best.pt").exists()
    assert (seed_dir / "diagnostics.json").exists()
    assert (tmp_path / "aggregate_summary.json").exists()
    assert (tmp_path / "diagnostics.json").exists()

    loaded = json.loads((tmp_path / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert loaded["belief_mode"] == "frequency"
    assert loaded["aggregate_metrics"]["top1_match_rate"]["mean"] == 1.0
    assert summary["artifact_paths"]["aggregate_summary"]
