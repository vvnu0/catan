# Human Evaluation Instructions

This workflow supports local, file-based human evaluation for the final project report.

## Bot Choice

- Main bot: `frequency` belief MCTS.
- Reference bot: `mcts` by default.
- `particle world_scaled` and neural guidance are not main human-study bots.

## Skill Groups

Assign each participant before the session:

- A: little or no Catan experience.
- B: casual Catan player.
- C: experienced Catan player.

Use pseudonymous participant IDs such as `P001`, `P002`, etc.

## Prepare A Session

```powershell
python scripts/prepare_human_eval.py --participant-id P001 --skill-group B
```

This writes:

- `experiments/human_eval/sessions/P001/manifest.json`

## Start A Session

```powershell
python scripts/start_human_eval_session.py --manifest experiments/human_eval/sessions/P001/manifest.json
```

This writes and prints:

- `experiments/human_eval/sessions/P001/start_checklist.json`

Use the checklist to configure each local game manually: bot, seed, human color, and bot color.

## Finalize A Session

For rehearsal or automated smoke testing:

```powershell
python scripts/finalize_human_eval_session.py --manifest experiments/human_eval/sessions/P001/manifest.json --dry-run
```

For a real session, create JSON files with `results` and `surveys` arrays, then run:

```powershell
python scripts/finalize_human_eval_session.py --manifest experiments/human_eval/sessions/P001/manifest.json --results-json path\to\results.json --surveys-json path\to\surveys.json
```

This writes:

- `experiments/human_eval/sessions/P001/completed_session.json`
- `experiments/human_eval/sessions/P001/results.json`
- `experiments/human_eval/sessions/P001/surveys.json`

## Summarize All Sessions

```powershell
python scripts/summarize_human_eval.py
```

This writes:

- `experiments/human_eval/aggregate_summary.json`
- `experiments/human_eval/results.csv`
- `experiments/human_eval/surveys.csv`

Inspect `aggregate_summary.json` first for participant counts, skill-group coverage, main/reference bot win rates, average ratings, and incomplete sessions.
