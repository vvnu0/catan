"""Validate expected artifacts and rebuild the final results bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_final_results_bundle import build_final_results_bundle

REQUIRED_MAIN_ARTIFACTS = (
    "experiments/opponent_modeling_sweep/medium/aggregate_summary.json",
    "experiments/neural_scaling_study/medium/aggregate_summary.json",
)

OPTIONAL_ARTIFACTS = (
    "experiments/human_eval/aggregate_summary.json",
    "experiments/mini_neural_exp/summary.json",
    "experiments/opponent_modeling_ablation/summary.json",
    "reports/neural_phase_audit.md",
)


def validate_and_rebuild(
    *,
    repo_root: str | Path = REPO,
    strict: bool = False,
) -> dict:
    repo_root = Path(repo_root)
    checks = []
    missing_required = []
    for rel in REQUIRED_MAIN_ARTIFACTS:
        present = (repo_root / rel).exists()
        checks.append({"path": rel, "required": True, "present": present})
        if not present:
            missing_required.append(rel)
    for rel in OPTIONAL_ARTIFACTS:
        checks.append({"path": rel, "required": False, "present": (repo_root / rel).exists()})

    if strict and missing_required:
        return {
            "ok": False,
            "missing_required": missing_required,
            "checks": checks,
            "rebuilt": False,
        }

    summary = build_final_results_bundle(
        experiments_dir=repo_root / "experiments",
        reports_dir=repo_root / "reports",
        output_dir=repo_root / "reports" / "final_results",
    )
    return {
        "ok": not missing_required,
        "missing_required": missing_required,
        "checks": checks,
        "rebuilt": True,
        "final_summary": summary["key_artifact_paths"]["final_summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and rebuild final artifacts")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    result = validate_and_rebuild(strict=args.strict)
    print(json.dumps(result, indent=2))
    if args.strict and not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
