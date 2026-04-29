from __future__ import annotations

import json

from scripts.run_mini_neural_experiment import run_experiment


def test_mini_neural_experiment_tiny_smoke(tmp_path):
    config_dir = tmp_path / "configs"
    output_dir = tmp_path / "experiment"
    config_dir.mkdir()

    (config_dir / "self_play.yaml").write_text(
        "\n".join([
            "num_games: 1",
            f"output_dir: {(output_dir / 'data' / 'self_play').as_posix()}",
            "seed: 111",
            "teacher_type: mcts",
            "max_simulations: 1",
            "max_depth: 1",
            "exploration_c: 1.41",
            "top_k_roads: 1",
            "top_k_trades: 1",
            "top_k_robber: 1",
            "shard_size: 9999",
        ]),
        encoding="utf-8",
    )
    (config_dir / "train.yaml").write_text(
        "\n".join([
            f"data_dir: {(output_dir / 'data' / 'self_play').as_posix()}",
            f"checkpoint_dir: {(output_dir / 'checkpoints').as_posix()}",
            "batch_size: 8",
            "lr: 0.001",
            "weight_decay: 0.0001",
            "epochs: 1",
            "validation_split: 0.5",
            "value_loss_weight: 1.0",
            "patience: 1",
            "overfit_batches: 0",
            "hidden_dim: 16",
            "dropout: 0.0",
        ]),
        encoding="utf-8",
    )
    (config_dir / "eval.yaml").write_text(
        "\n".join([
            "experiment_name: tiny_neural_exp",
            f"output_dir: {output_dir.as_posix()}",
            f"summary_path: {(output_dir / 'summary_for_test.json').as_posix()}",
            f"csv_path: {(output_dir / 'matchups.csv').as_posix()}",
            "num_games: 1",
            "base_seed: 222",
            "swap_seats: false",
            "search_seed: 333",
            "max_simulations: 1",
            "max_depth: 1",
            "exploration_c: 1.41",
            "puct_c: 2.5",
            "top_k_roads: 1",
            "top_k_trades: 1",
            "top_k_robber: 1",
            "include_heuristic: false",
        ]),
        encoding="utf-8",
    )

    summary = run_experiment(config_dir=config_dir)

    shards = list((output_dir / "data" / "self_play").glob("shard_*.pt"))
    checkpoints = list((output_dir / "checkpoints").glob("*.pt"))
    summary_path = output_dir / "summary_for_test.json"

    assert shards
    assert checkpoints
    assert summary_path.exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "matchups.csv").exists()
    assert summary["self_play_samples"] > 0
    assert summary["eval_matchups"]

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["experiment_name"] == "tiny_neural_exp"
    assert loaded["artifact_paths"]["checkpoint_path"]
