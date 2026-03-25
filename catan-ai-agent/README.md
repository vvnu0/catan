# catan-ai

AI agent wrapper layer around the [Catanatron](https://github.com/bcollazo/catanatron) game engine.

This repo does **not** reimplement the Catan game engine. It treats
`catanatron` as an external dependency (installed in editable mode from the
sibling `../catanatron` repo) and adds an AI project layer on top.

This project requires the sibling `../catanatron` repository to be installed
in editable mode in the **active virtual environment**. It should not rely on a
regular PyPI install of `catanatron`.

## Prerequisites

- Python >= 3.11
- Sibling `../catanatron` repo available locally and installed in editable mode:

```bash
pip install -e ../catanatron
```

## Install (editable)

```bash
pip install -e ".[dev]"
```

## Environment Check

```bash
cd ../catanatron
pip install -e ".[web,gym,dev]"

cd ../catan-ai-agent
pip install -e .
python scripts/cli_smoke.py
```

## Project Layout

```
src/catan_ai/           Main package (AI agent code goes here)
  utils/                 Shared helpers (logging, seeding, …)
scripts/
  cli_smoke.py           Smoke test — env info + catanatron-play check
  sim_probe.py           Step through a game tick-by-tick
  state_probe.py         Dump raw game state for manual inspection
reports/
  simulator_notes_template.md   Template for recording simulator observations
tests/
  test_imports.py        Verify catanatron is importable
```

## Quick Start

```bash
# Confirm everything is wired up
python scripts/cli_smoke.py

# Watch a game unfold tick by tick
python scripts/sim_probe.py

# Inspect raw state after 50 ticks
python scripts/state_probe.py

# Run tests
pytest tests/
```
