"""Run neural scaling study with combined training across all seeds.

Instead of training a separate model per seed, this script:
  1. Generates self-play data independently per seed (reproducible).
  2. Merges all shards into one combined dataset.
  3. Trains a single model on all data.
  4. Runs diagnostics and evaluation with that one model.
"""

from __future__ import annotations

import argparse
import json
import logging
import random as _random
import shutil
import time
from pathlib import Path
from typing import Any

from catanatron.models.player import Color, Player

from catan_ai.belief.determinizer import Determinizer
from catan_ai.eval.arena import Arena
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


# ── NeuralFrequencyMCTSPlayer (reused from scaling study) ────────────

class NeuralFrequencyMCTSPlayer(Player):
    """NeuralMCTS over one count-only determinized frequency-belief world."""

    def __init__(self, color, *, model, config: NeuralMCTSConfig,
                 belief_seed: int, is_bot: bool = True):
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


# ── Main pipeline ────────────────────────────────────────────────────

def run_combined(*, preset: str, output_dir: str | Path | None = None,
                 reuse_existing: bool = False) -> dict[str, Any]:
    """Generate data per seed, combine, train one model, evaluate."""
    cfg = _load_preset(preset)
    out = _resolve_path(output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    seeds = cfg["seeds"]
    t0 = time.perf_counter()

    # ── Phase 1: Self-play per seed ──────────────────────────────────
    seed_data_dirs: list[Path] = []
    total_samples = 0
    for seed in seeds:
        data_dir = out / f"seed_{seed}" / "data" / "self_play"
        data_dir.mkdir(parents=True, exist_ok=True)

        sp_cfg = SelfPlayConfig(
            num_games=int(cfg["self_play_games"]),
            output_dir=str(data_dir),
            seed=int(seed),
            teacher_type="frequency",
            max_simulations=int(cfg["self_play_sims"]),
            max_depth=int(cfg["self_play_depth"]),
            top_k_roads=int(cfg["top_k_roads"]),
            top_k_trades=int(cfg["top_k_trades"]),
            top_k_robber=int(cfg["top_k_robber"]),
            shard_size=int(cfg["shard_size"]),
        )

        if reuse_existing and list(data_dir.glob("shard_*.pt")):
            log.info("Reusing existing data for seed %d in %s", seed, data_dir)
        else:
            for shard in data_dir.glob("shard_*.pt"):
                shard.unlink()
            log.info("Self-play seed %d: %d games ...", seed, sp_cfg.num_games)
            run_self_play(sp_cfg)

        n = len(SelfPlayDataset(data_dir))
        total_samples += n
        seed_data_dirs.append(data_dir)
        log.info("Seed %d: %d samples", seed, n)

    # ── Phase 2: Merge shards ────────────────────────────────────────
    combined_dir = out / "combined_data"
    combined_dir.mkdir(parents=True, exist_ok=True)
    for f in combined_dir.glob("shard_*.pt"):
        f.unlink()

    shard_idx = 0
    for data_dir in seed_data_dirs:
        for shard_path in sorted(data_dir.glob("shard_*.pt")):
            dst = combined_dir / f"shard_{shard_idx:04d}.pt"
            shutil.copy2(shard_path, dst)
            shard_idx += 1

    dataset = SelfPlayDataset(combined_dir)
    log.info("Combined dataset: %d samples from %d shards", len(dataset), shard_idx)

    # ── Phase 3: Train one model ─────────────────────────────────────
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainConfig(
        data_dir=str(combined_dir),
        checkpoint_dir=str(ckpt_dir),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["learning_rate"]),
        epochs=int(cfg["train_epochs"]),
        hidden_dim=int(cfg["hidden_dim"]),
        patience=max(5, int(cfg["train_epochs"])),
    )

    ckpt_path = ckpt_dir / "best.pt"
    if reuse_existing and ckpt_path.exists():
        log.info("Reusing existing checkpoint %s", ckpt_path)
    else:
        log.info("Training on %d combined samples ...", len(dataset))
        ckpt_path = train(train_cfg)

    model, checkpoint = load_checkpoint(ckpt_path)

    # ── Phase 4: Diagnostics ─────────────────────────────────────────
    diagnostics = compute_model_diagnostics(
        model=model,
        data_dir=str(combined_dir),
        checkpoint=checkpoint,
        checkpoint_path=str(ckpt_path),
        max_samples=int(cfg["diagnostic_samples"]),
    )
    log.info("Diagnostics: flat=%.3f  top1=%.3f  value_mae=%.3f",
             diagnostics["flat_policy_fraction"],
             diagnostics["top1_match_rate"],
             diagnostics["value_mae"])

    # ── Phase 5: Evaluation ──────────────────────────────────────────
    eval_games = int(cfg["eval_games"])
    eval_sims = int(cfg["eval_sims"])
    eval_depth = int(cfg["eval_depth"])
    search_seed = int(cfg["search_seed"])

    arena = Arena(num_games=eval_games, base_seed=5000, swap_seats=True)

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
        puct_c=1.5,
        top_k_roads=int(cfg["top_k_roads"]),
        top_k_trades=int(cfg["top_k_trades"]),
        top_k_robber=int(cfg["top_k_robber"]),
        seed=search_seed,
        use_model_priors=True,
        use_model_value=True,
    )

    def make_neural_frequency(color: Color) -> NeuralFrequencyMCTSPlayer:
        return NeuralFrequencyMCTSPlayer(
            color, model=model, config=neural_cfg, belief_seed=search_seed,
        )

    baseline_result = arena.compare(
        make_frequency, make_frequency, "frequency_mcts vs frequency_mcts")
    neural_result = arena.compare(
        make_neural_frequency, make_frequency, "neural_frequency_mcts vs frequency_mcts")

    # ── Write summary ────────────────────────────────────────────────
    summary = {
        "experiment": "v3_combined",
        "seeds": [int(s) for s in seeds],
        "self_play_games_per_seed": int(cfg["self_play_games"]),
        "total_samples": len(dataset),
        "combined_shards": shard_idx,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "elapsed_sec": time.perf_counter() - t0,
        "diagnostics": diagnostics,
        "eval_matchups": {
            "baseline": _result_dict(baseline_result),
            "neural_vs_frequency": _result_dict(neural_result),
        },
        "config": cfg,
    }

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str),
                            encoding="utf-8")
    log.info("Summary written to %s", summary_path)

    # Print key results
    print("\n" + "=" * 60)
    print("V3 COMBINED RESULTS")
    print("=" * 60)
    print(f"Total samples:        {len(dataset)}")
    print(f"Best checkpoint:      epoch {checkpoint.get('epoch')}")
    print(f"Flat policy fraction: {diagnostics['flat_policy_fraction']:.3f}")
    print(f"Top-1 match rate:     {diagnostics['top1_match_rate']:.3f}")
    print(f"Value MAE:            {diagnostics['value_mae']:.3f}")
    print(f"")
    print(f"Baseline (freq vs freq): {baseline_result.summary()}")
    print(f"Neural vs frequency:     {neural_result.summary()}")
    print("=" * 60)

    return summary


def _result_dict(result) -> dict[str, Any]:
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
    }


# ── Config loading (reused from scaling study) ──────────────────────

def _load_preset(preset: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{preset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown preset {preset!r}: {path}")
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


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neural scaling study with combined training")
    parser.add_argument("--preset", default="v3_combined")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Skip self-play/training if artifacts exist")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_combined(
        preset=args.preset,
        output_dir=args.output_dir,
        reuse_existing=args.reuse_existing,
    )


if __name__ == "__main__":
    main()
