from __future__ import annotations

import json

from catanatron import Color

from catan_ai.eval.opponent_modeling import (
    OpponentModelEvalConfig,
    OpponentModelMode,
    make_opponent_model_player,
)
from scripts.run_opponent_modeling_ablation import run_ablation


def test_opponent_modeling_modes_instantiate():
    cfg = OpponentModelEvalConfig(total_simulations=1, max_depth=1, particle_worlds=1)

    for mode in OpponentModelMode:
        player = make_opponent_model_player(mode, Color.RED, cfg)
        assert player.color == Color.RED


def test_opponent_modeling_ablation_tiny_smoke(tmp_path):
    summary = run_ablation(
        output_dir=tmp_path,
        seeds=[1234],
        swap_seats=False,
        search_cfg=OpponentModelEvalConfig(
            total_simulations=1,
            max_depth=1,
            top_k_roads=1,
            top_k_trades=1,
            top_k_robber=1,
            particle_worlds=1,
            enable_particle_devcards=False,
        ),
    )

    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "matchups.csv"
    diag_path = tmp_path / "belief_diagnostics.json"

    assert summary_path.exists()
    assert csv_path.exists()
    assert diag_path.exists()
    assert {m["candidate_mode"] for m in summary["eval_matchups"]} == {
        "frequency",
        "particle",
    }

    diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
    assert set(diagnostics["diagnostics"]) == {"none", "frequency", "particle"}
    assert diagnostics["diagnostics"]["frequency"]["players_created"] > 0
    assert diagnostics["diagnostics"]["particle"]["players_created"] > 0
