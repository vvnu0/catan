"""Run a small reproducible NeuralMCTS training/evaluation experiment.

The workflow is intentionally simple:
  1. Generate MCTS self-play shards.
  2. Train a PolicyValueNet checkpoint.
  3. Evaluate plain MCTS vs plain MCTS and NeuralMCTS vs MCTS.
  4. Write JSON and CSV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from catanatron import Color

from catan_ai.eval.arena import Arena, MatchResult
from catan_ai.players import HeuristicBot, MCTSPlayer
from catan_ai.players.neural_mcts_player import NeuralMCTSConfig, NeuralMCTSPlayer
from catan_ai.training.checkpoints import load_checkpoint
from catan_ai.training.dataset import SelfPlayDataset
from catan_ai.training.self_play import SelfPlayConfig, run_self_play
from catan_ai.training.train_policy_value import TrainConfig, train

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO / "configs" / "mini_neural_exp"


def run_experiment(
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    output_dir: str | Path | None = None,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Run the configured mini experiment and return its summary dict."""
    config_dir = Path(config_dir)
    self_play_cfg = _dataclass_from_yaml(SelfPlayConfig, config_dir / "self_play.yaml")
    train_cfg = _dataclass_from_yaml(TrainConfig, config_dir / "train.yaml")
    eval_cfg = _load_simple_yaml(config_dir / "eval.yaml")

    if output_dir is not None:
        eval_cfg["output_dir"] = str(output_dir)
        out = _resolve_path(output_dir)
        self_play_cfg.output_dir = str(out / "data" / "self_play")
        train_cfg.data_dir = self_play_cfg.output_dir
        train_cfg.checkpoint_dir = str(out / "checkpoints")
        eval_cfg["csv_path"] = str(out / "matchups.csv")

    experiment_name = str(eval_cfg.get("experiment_name", "mini_neural_exp"))
    output_root = _resolve_path(eval_cfg.get("output_dir", f"experiments/{experiment_name}"))
    summary_path = _resolve_path(
        eval_cfg.get("summary_path", f"reports/{experiment_name}_summary.json")
    )
    csv_path = _resolve_path(eval_cfg.get("csv_path", output_root / "matchups.csv"))

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    self_play_dir = _resolve_path(self_play_cfg.output_dir)
    checkpoint_dir = _resolve_path(train_cfg.checkpoint_dir)
    self_play_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if not reuse_existing:
        _remove_shards(self_play_dir)

    shard_paths = sorted(self_play_dir.glob("shard_*.pt"))
    if reuse_existing and shard_paths:
        log.info("Reusing existing self-play shards in %s", self_play_dir)
    else:
        log.info("Running self-play: %s", self_play_cfg)
        run_self_play(self_play_cfg)

    dataset = SelfPlayDataset(self_play_dir)
    shard_paths = sorted(self_play_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise RuntimeError(f"No self-play shards written in {self_play_dir}")

    best_checkpoint = checkpoint_dir / "best.pt"
    if reuse_existing and best_checkpoint.exists():
        checkpoint_path = best_checkpoint
        log.info("Reusing existing checkpoint %s", checkpoint_path)
    else:
        train_cfg.data_dir = str(self_play_dir)
        train_cfg.checkpoint_dir = str(checkpoint_dir)
        log.info("Training policy/value net: %s", train_cfg)
        checkpoint_path = train(train_cfg)

    model, ckpt = load_checkpoint(checkpoint_path)
    matchups = _run_eval(eval_cfg, model)

    summary = _build_summary(
        experiment_name=experiment_name,
        self_play_cfg=self_play_cfg,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        self_play_dir=self_play_dir,
        shard_paths=shard_paths,
        self_play_samples=len(dataset),
        checkpoint_path=checkpoint_path,
        checkpoint=ckpt,
        matchups=matchups,
        summary_path=summary_path,
        csv_path=csv_path,
        elapsed_sec=time.perf_counter() - t0,
    )

    _write_matchups_csv(csv_path, summary["eval_matchups"])
    _write_json(summary_path, summary)
    experiment_summary_path = output_root / "summary.json"
    if experiment_summary_path.resolve() != summary_path.resolve():
        _write_json(experiment_summary_path, summary)

    return summary


def _run_eval(eval_cfg: dict[str, Any], model) -> list[MatchResult]:
    num_games = int(eval_cfg.get("num_games", 4))
    base_seed = int(eval_cfg.get("base_seed", 9000))
    swap_seats = bool(eval_cfg.get("swap_seats", True))
    search_seed = int(eval_cfg.get("search_seed", 2026))
    max_simulations = int(eval_cfg.get("max_simulations", 25))
    max_depth = int(eval_cfg.get("max_depth", 8))
    top_k_roads = int(eval_cfg.get("top_k_roads", 3))
    top_k_trades = int(eval_cfg.get("top_k_trades", 2))
    top_k_robber = int(eval_cfg.get("top_k_robber", 4))
    exploration_c = float(eval_cfg.get("exploration_c", 1.41))
    puct_c = float(eval_cfg.get("puct_c", 2.5))

    arena = Arena(num_games=num_games, base_seed=base_seed, swap_seats=swap_seats)

    def make_mcts(color: Color) -> MCTSPlayer:
        return MCTSPlayer(
            color,
            max_simulations=max_simulations,
            max_depth=max_depth,
            exploration_c=exploration_c,
            top_k_roads=top_k_roads,
            top_k_trades=top_k_trades,
            top_k_robber=top_k_robber,
            seed=search_seed,
        )

    neural_cfg = NeuralMCTSConfig(
        max_simulations=max_simulations,
        max_depth=max_depth,
        puct_c=puct_c,
        top_k_roads=top_k_roads,
        top_k_trades=top_k_trades,
        top_k_robber=top_k_robber,
        seed=search_seed,
        use_model_priors=True,
        use_model_value=True,
    )

    def make_neural(color: Color) -> NeuralMCTSPlayer:
        return NeuralMCTSPlayer(color, model=model, config=neural_cfg)

    results = [
        arena.compare(make_mcts, make_mcts, "MCTS vs MCTS sanity"),
        arena.compare(make_neural, make_mcts, "NeuralMCTS checkpoint vs MCTS"),
    ]

    if bool(eval_cfg.get("include_heuristic", True)):
        results.append(
            arena.compare(make_neural, lambda c: HeuristicBot(c), "NeuralMCTS checkpoint vs HeuristicBot")
        )

    return results


def _build_summary(
    *,
    experiment_name: str,
    self_play_cfg: SelfPlayConfig,
    train_cfg: TrainConfig,
    eval_cfg: dict[str, Any],
    self_play_dir: Path,
    shard_paths: list[Path],
    self_play_samples: int,
    checkpoint_path: Path,
    checkpoint: dict,
    matchups: list[MatchResult],
    summary_path: Path,
    csv_path: Path,
    elapsed_sec: float,
) -> dict[str, Any]:
    eval_matchups = [_matchup_summary(m, eval_cfg) for m in matchups]
    output_root = _resolve_path(eval_cfg.get("output_dir", f"experiments/{experiment_name}"))

    return {
        "experiment_name": experiment_name,
        "elapsed_sec": elapsed_sec,
        "self_play_games": self_play_cfg.num_games,
        "self_play_samples": self_play_samples,
        "self_play_shards": [_display_path(p) for p in shard_paths],
        "train_epochs": train_cfg.epochs,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "checkpoint_path": _display_path(checkpoint_path),
        "eval_matchups": eval_matchups,
        "artifact_paths": {
            "dataset_dir": _display_path(self_play_dir),
            "checkpoint_path": _display_path(checkpoint_path),
            "summary_path": _display_path(summary_path),
            "experiment_summary_path": _display_path(output_root / "summary.json"),
            "csv_path": _display_path(csv_path),
        },
        "configs": {
            "self_play": asdict(self_play_cfg),
            "train": asdict(train_cfg),
            "eval": eval_cfg,
        },
    }


def _matchup_summary(result: MatchResult, eval_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": result.label,
        "games": result.games,
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "win_rate": result.win_rate,
        "avg_final_vp": result.avg_final_vp,
        "avg_turns": result.avg_turns,
        "mean_latency_ms": result.avg_move_ms,
        "seed_info": {
            "base_seed": int(eval_cfg.get("base_seed", 9000)),
            "search_seed": int(eval_cfg.get("search_seed", 2026)),
            "swap_seats": bool(eval_cfg.get("swap_seats", True)),
        },
        "search_budget": {
            "max_simulations": int(eval_cfg.get("max_simulations", 25)),
            "max_depth": int(eval_cfg.get("max_depth", 8)),
            "top_k_roads": int(eval_cfg.get("top_k_roads", 3)),
            "top_k_trades": int(eval_cfg.get("top_k_trades", 2)),
            "top_k_robber": int(eval_cfg.get("top_k_robber", 4)),
        },
    }


def _write_matchups_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "avg_final_vp",
        "avg_turns",
        "mean_latency_ms",
        "base_seed",
        "search_seed",
        "swap_seats",
        "max_simulations",
        "max_depth",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "label": row["label"],
                "games": row["games"],
                "wins": row["wins"],
                "losses": row["losses"],
                "draws": row["draws"],
                "win_rate": row["win_rate"],
                "avg_final_vp": row["avg_final_vp"],
                "avg_turns": row["avg_turns"],
                "mean_latency_ms": row["mean_latency_ms"],
                "base_seed": row["seed_info"]["base_seed"],
                "search_seed": row["seed_info"]["search_seed"],
                "swap_seats": row["seed_info"]["swap_seats"],
                "max_simulations": row["search_budget"]["max_simulations"],
                "max_depth": row["search_budget"]["max_depth"],
            })


def _dataclass_from_yaml(cls, path: Path):
    values = _load_simple_yaml(path)
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in allowed})


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the flat key/value YAML used by the mini experiment configs."""
    data: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line in {path}: {raw!r}")
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _remove_shards(path: Path) -> None:
    for shard in path.glob("shard_*.pt"):
        shard.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mini NeuralMCTS experiment")
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing self_play.yaml, train.yaml, and eval.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for the experiment output directory",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing shards/checkpoint when present",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run_experiment(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(summary["artifact_paths"], indent=2))


if __name__ == "__main__":
    main()
