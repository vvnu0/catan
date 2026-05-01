"""Preflight checks for demos and human-eval sessions."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


REQUIRED_ARTIFACTS = (
    "reports/final_results/final_summary.json",
    "reports/final_results/FINAL_RESULTS.md",
    "experiments/opponent_modeling_sweep/medium/aggregate_summary.json",
    "experiments/neural_scaling_study/medium/aggregate_summary.json",
    "configs/human_eval/default.yaml",
)

OPTIONAL_ARTIFACTS = (
    "experiments/human_eval/aggregate_summary.json",
    "reports/final_results/SUBMISSION_CHECKLIST.md",
)

IMPORT_CHECKS = (
    "catanatron",
    "catan_ai.players.mcts_player",
    "catan_ai.players.belief_mcts_player",
    "catan_ai.eval.human_eval",
)


def run_preflight(*, repo_root: str | Path = REPO, strict: bool = False) -> dict[str, Any]:
    repo_root = Path(repo_root)
    checks: list[dict[str, Any]] = []

    for module in IMPORT_CHECKS:
        checks.append(_check_import(module))

    for rel in REQUIRED_ARTIFACTS:
        checks.append(_check_path(repo_root / rel, rel, required=True))
    for rel in OPTIONAL_ARTIFACTS:
        checks.append(_check_path(repo_root / rel, rel, required=False))

    checks.append(_check_bot_instantiation())
    checks.append({
        "name": "catanatron_cli",
        "required": False,
        "ok": shutil.which("catanatron") is not None,
        "detail": shutil.which("catanatron") or "not found on PATH; manual/demo scripts can still use Python APIs",
    })
    checks.append({
        "name": "docker_cli",
        "required": False,
        "ok": shutil.which("docker") is not None,
        "detail": shutil.which("docker") or "not found on PATH; only relevant if using a Docker UI",
    })

    required_failures = [c for c in checks if c.get("required") and not c.get("ok")]
    report = {
        "ok": not required_failures,
        "strict": strict,
        "required_failures": required_failures,
        "checks": checks,
        "artifact_path": str(repo_root / "reports" / "final_results" / "demo_preflight.json"),
    }
    out = repo_root / "reports" / "final_results" / "demo_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if strict and required_failures:
        raise SystemExit(1)
    return report


def _check_import(module: str) -> dict[str, Any]:
    try:
        importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - detail path
        return {"name": f"import:{module}", "required": True, "ok": False, "detail": repr(exc)}
    return {"name": f"import:{module}", "required": True, "ok": True, "detail": "ok"}


def _check_path(path: Path, rel: str, *, required: bool) -> dict[str, Any]:
    return {
        "name": f"path:{rel}",
        "required": required,
        "ok": path.exists(),
        "detail": str(path),
    }


def _check_bot_instantiation() -> dict[str, Any]:
    try:
        from catanatron import Color
        from catan_ai.players.belief_mcts_player import BeliefMCTSConfig, BeliefMCTSPlayer
        from catan_ai.players.mcts_player import MCTSPlayer

        BeliefMCTSPlayer(
            Color.RED,
            config=BeliefMCTSConfig(
                num_worlds=1,
                sims_per_world=16,
                belief_mode="count_only",
                enable_devcard_sampling=False,
            ),
        )
        MCTSPlayer(Color.BLUE, max_simulations=16, max_depth=8, seed=2026)
    except Exception as exc:  # pragma: no cover - detail path
        return {"name": "bot_instantiation", "required": True, "ok": False, "detail": repr(exc)}
    return {"name": "bot_instantiation", "required": True, "ok": True, "detail": "frequency and mcts instantiate"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo/session preflight checks")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = run_preflight(strict=args.strict)
    print(json.dumps({
        "ok": report["ok"],
        "required_failures": [f["name"] for f in report["required_failures"]],
        "artifact_path": report["artifact_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
