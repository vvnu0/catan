"""Prepare a human-evaluation participant session manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catan_ai.eval.human_eval import load_simple_yaml, prepare_session

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "human_eval" / "default.yaml"


def run_prepare(
    *,
    participant_id: str,
    skill_group: str,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
) -> dict:
    cfg = load_simple_yaml(config_path)
    return prepare_session(
        config=cfg,
        participant_id=participant_id,
        skill_group=skill_group,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a human-eval session")
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--skill-group", required=True, choices=("A", "B", "C", "a", "b", "c"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    manifest = run_prepare(
        participant_id=args.participant_id,
        skill_group=args.skill_group,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "manifest_path": str(Path(manifest["session_dir"]) / "manifest.json"),
        "participant_id": manifest["participant_id"],
        "games": len(manifest["games"]),
    }, indent=2))


if __name__ == "__main__":
    main()
