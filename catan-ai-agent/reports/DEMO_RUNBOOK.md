# Demo And Human Session Runbook

This runbook is for local demos and participant sessions. It does not launch a frontend or change the engine.

## 1. Preflight

```powershell
python scripts/demo_preflight.py
```

Strict mode:

```powershell
python scripts/demo_preflight.py --strict
```

Output:

- `reports/final_results/demo_preflight.json`

Fix required failures before a live demo or participant session.

## 2. Canonical Final Demo

Dry-run the plan:

```powershell
python scripts/run_final_demo.py --dry-run
```

Run the canonical demo match:

```powershell
python scripts/run_final_demo.py
```

Optional heuristic baseline:

```powershell
python scripts/run_final_demo.py --matchup main_vs_heuristic
```

Outputs are written under `experiments/demo_runs/<demo_timestamp>/`.

## 3. Browser Visual Demo

Start the Catanatron web stack from the sibling repo:

```powershell
cd ..\catanatron
docker compose up --build
```

Open:

- `http://localhost:3000`

The create-game page includes:

- `Main Bot (Frequency MCTS)`
- `Reference Bot (Plain MCTS)`

Watch a custom bot-vs-bot replay:

```powershell
cd ..\catan-ai-agent
python scripts/run_visual_match.py --matchup main_vs_reference
```

Dry-run only:

```powershell
python scripts/run_visual_match.py --matchup main_vs_reference --dry-run
```

Start a human-vs-main-bot browser game:

```powershell
python scripts/run_browser_human_vs_bot.py --bot MAIN_BOT --human-color RED
```

Start a human-vs-reference-bot browser game:

```powershell
python scripts/run_browser_human_vs_bot.py --bot REFERENCE_BOT --human-color RED
```

If the browser game cannot be created, confirm the backend is reachable at `http://localhost:5001` and that the Docker server mounted `../catan-ai-agent` through `PYTHONPATH=/catan-ai-agent/src`.

## 4. Prepare A Participant Session

```powershell
python scripts/prepare_human_eval.py --participant-id P001 --skill-group B
python scripts/start_human_eval_session.py --manifest experiments/human_eval/sessions/P001/manifest.json
```

Start writes:

- `start_checklist.json`
- `session_commands.json`
- `session_commands.md`

Use `session_commands.md` as the operator playbook.

## 5. If A Game Crashes

Record the crash in the notes for that game. If the game cannot be completed, leave the session incomplete and rerun:

```powershell
python scripts/check_human_eval_status.py
```

Do not fabricate result or survey values. Either replay the game with the same seed or mark the session incomplete in your report notes.

## 6. Finalize And Summarize

After all games are complete and result/survey JSON files are ready:

```powershell
python scripts/finalize_human_eval_session.py --manifest experiments/human_eval/sessions/P001/manifest.json --results-json experiments/human_eval/sessions/P001/results.json --surveys-json experiments/human_eval/sessions/P001/surveys.json
python scripts/summarize_human_eval.py
python scripts/build_final_results_bundle.py
```

Check status any time:

```powershell
python scripts/check_human_eval_status.py
```

Status output:

- `experiments/human_eval/session_status.json`
