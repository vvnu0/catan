"""Run neural scaling studies against the frequency-belief MCTS baseline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random as _random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from catanatron.models.player import Color, Player

from catan_ai.belief.determinizer import Determinizer
from catan_ai.eval.arena import Arena, MatchResult
from catan_ai.eval.model_diagnostics import compute_model_diagnostics
from catan_ai.players.belief_mcts_player import BeliefMCTSConfig, BeliefMCTSPlayer
from catan_ai.players.decision_context import DecisionContext
from catan_ai.players.neural_mcts_player import NeuralMCTS, NeuralMCTSConfig
from catan_ai.search.candidate_filter import CandidateFilter
from catan_ai.training.checkpoints import load_checkpoint
from catan_ai.training.dataset import SelfPlayDataset
from catan_ai.training.self_play import SelfPlayConfig, run_self_play
from catan_ai.training.train_policy_value import TrainConfig, train

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs" / "neural_scaling_study"


class NeuralFrequencyMCTSPlayer(Player):
    """NeuralMCTS over one count-only determinized frequency-belief world."""

    def __init__(
        self,
        color,
        *,
        model,
        config: NeuralMCTSConfig,
        belief_seed: int,
        is_bot: bool = True,
    ):
        super().__init__(color, is_bot=is_bot)
        self.model = model
        self.config = config
        self.belief_seed = belief_seed
        self._calls = 0
        self._total_search_ms = 0.0

    def decide(self, game, playable_actions):
        self._calls += 1
        if len(playable_actions) == 1:
            return playable_actions[0]

        root_ctx = DecisionContext(game, playable_actions, self.color)
        rng = _random.Random(self.belief_seed + game.state.num_turns * 997)
        world = Determinizer(
            acting_color=self.color,
            belief_mode="count_only",
            enable_devcard_sampling=False,
        ).sample_world(game, rng)
        if world is None:
            world = game

        t0 = time.perf_counter()
        engine = NeuralMCTS(
            root_color=self.color,
            model=self.model,
            cfg=self.config,
            candidate_filter=CandidateFilter(
                top_k_roads=self.config.top_k_roads,
                top_k_trades=self.config.top_k_trades,
                top_k_robber=self.config.top_k_robber,
            ),
        )
        best_ea, _stats = engine.search(world)
        self._total_search_ms += (time.perf_counter() - t0) * 1000
        return root_ctx.get_raw_action(best_ea)

    @property
    def avg_move_ms(self) -> float:
        return self._total_search_ms / self._calls if self._calls > 0 else 0.0


def run_study(
    *,
    preset: str = "small",
    output_dir: str | Path | None = None,
    seeds: list[int] | None = None,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    cfg = _load_preset(preset)
    if seeds is not None:
        cfg["seeds"] = seeds
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)

    if cfg.get("belief_mode", "frequency") != "frequency":
        raise ValueError("Neural scaling study main path requires belief_mode='frequency'")

    out = _resolve_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed_summaries = []
    matchup_rows = []
    diagnostics_by_seed = {}
    t0 = time.perf_counter()

    for seed in cfg["seeds"]:
        seed_dir = out / f"seed_{seed}"
        seed_summary = _run_seed(
            cfg=cfg,
            seed=int(seed),
            seed_dir=seed_dir,
            reuse_existing=reuse_existing,
        )
        seed_summaries.append(seed_summary)
        matchup_rows.extend(seed_summary["eval_matchups"])
        diagnostics_by_seed[str(seed)] = seed_summary["diagnostics"]

    aggregate = _aggregate_summary(
        cfg=cfg,
        output_dir=out,
        seed_summaries=seed_summaries,
        matchup_rows=matchup_rows,
        diagnostics_by_seed=diagnostics_by_seed,
        elapsed_sec=time.perf_counter() - t0,
    )
    _write_json(out / "aggregate_summary.json", aggregate)
    _write_json(out / "diagnostics.json", diagnostics_by_seed)
    _write_matchups_csv(out / "matchups.csv", matchup_rows)
    return aggregate


def _run_seed(
    *,
    cfg: dict[str, Any],
    seed: int,
    seed_dir: Path,
    reuse_existing: bool,
) -> dict[str, Any]:
    seed_dir.mkdir(parents=True, exist_ok=True)
    data_dir = seed_dir / "data" / "self_play"
    ckpt_dir = seed_dir / "checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    self_play_cfg = SelfPlayConfig(
        num_games=int(cfg["self_play_games"]),
        output_dir=str(data_dir),
        seed=seed,
        teacher_type="frequency",
        max_simulations=int(cfg["self_play_sims"]),
        max_depth=int(cfg["self_play_depth"]),
        top_k_roads=int(cfg["top_k_roads"]),
        top_k_trades=int(cfg["top_k_trades"]),
        top_k_robber=int(cfg["top_k_robber"]),
        shard_size=int(cfg["shard_size"]),
    )
    if not reuse_existing:
        for shard in data_dir.glob("shard_*.pt"):
            shard.unlink()
    if not (reuse_existing and list(data_dir.glob("shard_*.pt"))):
        run_self_play(self_play_cfg)

    train_cfg = TrainConfig(
        data_dir=str(data_dir),
        checkpoint_dir=str(ckpt_dir),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["learning_rate"]),
        epochs=int(cfg["train_epochs"]),
        hidden_dim=int(cfg["hidden_dim"]),
        patience=max(2, int(cfg["train_epochs"])),
    )
    checkpoint_path = ckpt_dir / "best.pt"
    if not (reuse_existing and checkpoint_path.exists()):
        checkpoint_path = train(train_cfg)

    model, checkpoint = load_checkpoint(checkpoint_path)
    dataset = SelfPlayDataset(data_dir)
    diagnostics = compute_model_diagnostics(
        model=model,
        data_dir=data_dir,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        max_samples=int(cfg["diagnostic_samples"]),
    )
    matchups = _run_eval(cfg, seed=seed, model=model)

    summary = {
        "seed": seed,
        "belief_mode": "frequency",
        "self_play_games": self_play_cfg.num_games,
        "self_play_samples": len(dataset),
        "checkpoint_path": _display_path(Path(checkpoint_path)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "diagnostics": diagnostics,
        "eval_matchups": matchups,
        "artifact_paths": {
            "seed_dir": _display_path(seed_dir),
            "dataset_dir": _display_path(data_dir),
            "checkpoint_path": _display_path(Path(checkpoint_path)),
            "summary": _display_path(seed_dir / "summary.json"),
            "matchups_csv": _display_path(seed_dir / "matchups.csv"),
            "diagnostics": _display_path(seed_dir / "diagnostics.json"),
        },
    }
    _write_json(seed_dir / "summary.json", summary)
    _write_json(seed_dir / "diagnostics.json", diagnostics)
    _write_matchups_csv(seed_dir / "matchups.csv", matchups)
    return summary


def _run_eval(cfg: dict[str, Any], *, seed: int, model) -> list[dict[str, Any]]:
    arena = Arena(
        num_games=int(cfg["eval_games"]),
        base_seed=seed + 10000,
        swap_seats=True,
    )
    eval_sims = int(cfg["eval_sims"])
    eval_depth = int(cfg["eval_depth"])
    search_seed = int(cfg["search_seed"])

    def make_frequency(color: Color) -> BeliefMCTSPlayer:
        return BeliefMCTSPlayer(
            color,
            config=BeliefMCTSConfig(
                num_worlds=1,
                sims_per_world=eval_sims,
                belief_seed=search_seed,
                belief_mode="count_only",
                enable_devcard_sampling=False,
                max_depth=eval_depth,
                top_k_roads=int(cfg["top_k_roads"]),
                top_k_trades=int(cfg["top_k_trades"]),
                top_k_robber=int(cfg["top_k_robber"]),
            ),
        )

    neural_cfg = NeuralMCTSConfig(
        max_simulations=eval_sims,
        max_depth=eval_depth,
        puct_c=2.5,
        top_k_roads=int(cfg["top_k_roads"]),
        top_k_trades=int(cfg["top_k_trades"]),
        top_k_robber=int(cfg["top_k_robber"]),
        seed=search_seed,
        use_model_priors=True,
        use_model_value=True,
    )

    def make_neural_frequency(color: Color) -> NeuralFrequencyMCTSPlayer:
        return NeuralFrequencyMCTSPlayer(
            color,
            model=model,
            config=neural_cfg,
            belief_seed=search_seed,
        )

    results = [
        arena.compare(make_frequency, make_frequency, "frequency_mcts vs frequency_mcts"),
        arena.compare(make_neural_frequency, make_frequency, "neural_frequency_mcts vs frequency_mcts"),
    ]
    return [_matchup_summary(result, cfg, seed) for result in results]


def _matchup_summary(result: MatchResult, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "label": result.label,
        "games": result.games,
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "win_rate": result.win_rate,
        "avg_final_vp": result.avg_final_vp,
        "avg_turns": result.avg_turns,
        "mean_latency_ms": result.avg_move_ms,
        "belief_mode": "frequency",
        "search_budget": {
            "eval_sims": int(cfg["eval_sims"]),
            "eval_depth": int(cfg["eval_depth"]),
            "top_k_roads": int(cfg["top_k_roads"]),
            "top_k_trades": int(cfg["top_k_trades"]),
            "top_k_robber": int(cfg["top_k_robber"]),
        },
    }


def _aggregate_summary(
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    seed_summaries: list[dict[str, Any]],
    matchup_rows: list[dict[str, Any]],
    diagnostics_by_seed: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    neural_rows = [
        row for row in matchup_rows
        if row["label"] == "neural_frequency_mcts vs frequency_mcts"
    ]
    diag_values = list(diagnostics_by_seed.values())
    return {
        "experiment_name": "neural_scaling_study",
        "preset": cfg["preset"],
        "seeds": cfg["seeds"],
        "belief_mode": "frequency",
        "self_play_games_per_seed": int(cfg["self_play_games"]),
        "total_samples": sum(s["self_play_samples"] for s in seed_summaries),
        "checkpoint_paths": [s["checkpoint_path"] for s in seed_summaries],
        "elapsed_sec": elapsed_sec,
        "per_seed": seed_summaries,
        "eval_matchups": matchup_rows,
        "aggregate_metrics": {
            "win_rate_vs_frequency_mcts": _mean_std([r["win_rate"] for r in neural_rows]),
            "avg_final_vp_vs_frequency_mcts": _mean_std([r["avg_final_vp"] for r in neural_rows]),
            "avg_turns_vs_frequency_mcts": _mean_std([r["avg_turns"] for r in neural_rows]),
            "policy_entropy": _mean_std([d["mean_policy_entropy"] for d in diag_values]),
            "flat_policy_fraction": _mean_std([d["flat_policy_fraction"] for d in diag_values]),
            "top1_match_rate": _mean_std([d["top1_match_rate"] for d in diag_values]),
            "value_mae": _mean_std([d["value_mae"] for d in diag_values]),
            "value_mse": _mean_std([d["value_mse"] for d in diag_values]),
        },
        "recommendation": _recommend(neural_rows, diag_values),
        "artifact_paths": {
            "aggregate_summary": _display_path(output_dir / "aggregate_summary.json"),
            "matchups_csv": _display_path(output_dir / "matchups.csv"),
            "diagnostics": _display_path(output_dir / "diagnostics.json"),
        },
    }


def _recommend(neural_rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    win_rate = _mean([r["win_rate"] for r in neural_rows])
    top1 = _mean([d["top1_match_rate"] for d in diagnostics])
    flat = _mean([d["flat_policy_fraction"] for d in diagnostics])
    value_mae = _mean([d["value_mae"] for d in diagnostics])
    flags: list[str] = []
    if win_rate >= 0.55:
        flags.append("keep_neural_main")
    elif win_rate >= 0.45:
        flags.append("neural_inconclusive")
    else:
        flags.append("keep_neural_secondary")
    if flat >= 0.8:
        flags.append("policy_too_flat")
    if top1 < 0.25:
        flags.append("weak_policy_top1")
    if value_mae > 0.9:
        flags.append("weak_value_head")
    return {
        "primary": flags[0],
        "flags": flags,
        "metrics": {
            "mean_win_rate_vs_frequency_mcts": win_rate,
            "mean_top1_match_rate": top1,
            "mean_flat_policy_fraction": flat,
            "mean_value_mae": value_mae,
        },
    }


def _write_matchups_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "label",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "avg_final_vp",
        "avg_turns",
        "mean_latency_ms",
        "belief_mode",
        "eval_sims",
        "eval_depth",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "seed": row["seed"],
                "label": row["label"],
                "games": row["games"],
                "wins": row["wins"],
                "losses": row["losses"],
                "draws": row["draws"],
                "win_rate": row["win_rate"],
                "avg_final_vp": row["avg_final_vp"],
                "avg_turns": row["avg_turns"],
                "mean_latency_ms": row["mean_latency_ms"],
                "belief_mode": row["belief_mode"],
                "eval_sims": row["search_budget"]["eval_sims"],
                "eval_depth": row["search_budget"]["eval_depth"],
            })


def _load_preset(preset: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{preset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown neural scaling preset {preset!r}: {path}")
    return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = _mean(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "std": variance ** 0.5}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run neural scaling study")
    parser.add_argument("--preset", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated explicit seeds")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    summary = run_study(
        preset=args.preset,
        output_dir=args.output_dir,
        seeds=_parse_csv_ints(args.seeds),
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(summary["artifact_paths"], indent=2))
    print(json.dumps(summary["recommendation"], indent=2))


if __name__ == "__main__":
    main()
