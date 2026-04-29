"""Aggregate human-evaluation sessions into JSON and CSV artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catan_ai.eval.human_eval import summarize_sessions

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO / "experiments" / "human_eval"


def run_summarize(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    return summarize_sessions(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize human-eval sessions")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    summary = run_summarize(output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
