"""Print the operator checklist for a prepared human-eval session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catan_ai.eval.human_eval import start_session


def run_start(*, manifest_path: str | Path) -> dict:
    return start_session(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start/check a human-eval session")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    checklist = run_start(manifest_path=args.manifest)
    print(json.dumps(checklist, indent=2))


if __name__ == "__main__":
    main()
