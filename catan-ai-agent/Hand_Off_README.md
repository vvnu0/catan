# Catan AI Agent: Team Handoff README

This README is the handoff document for teammates continuing the project. It maps the original proposal to what is implemented, summarizes the current results, explains where things live, and gives the commands needed to run tests, demos, experiments, and human-evaluation sessions.

The short version: the main technical story is now frozen. The final bot is **frequency belief MCTS**, the reference bot is **plain MCTS**, particle belief is appendix-only, and neural guidance is currently a secondary/negative-result track.

---

## Current Project Status

### Final Bot Choices

| Role | Choice | Why |
|---|---|---|
| Main final bot | `frequency` belief MCTS | Best practical choice from the opponent-modeling evidence. Cheap, stable, and proposal-aligned. |
| Reference bot | plain MCTS | Clean baseline with the same search stack but no opponent-belief layer. |
| Appendix variant | `particle world_scaled` | Interesting but not the default; only reasonable when budget scales with sampled worlds. |
| Not main | `particle compute_matched` | Under-budgeted and underperforms. |
| Not main | neural MCTS | Pipeline works, but current scaling results do not beat frequency MCTS. |

### Current Results Snapshot

Canonical source:

- `reports/final_results/final_summary.json`
- `reports/final_results/FINAL_RESULTS.md`

Current headline metrics from the final bundle:

| Result | Current value | Source |
|---|---:|---|
| `frequency` vs `none` win rate | `0.75` | `experiments/opponent_modeling_sweep/medium/aggregate_summary.json` |
| `particle world_scaled` vs `none` win rate | `0.75` | `experiments/opponent_modeling_sweep/medium/aggregate_summary.json` |
| `neural_frequency_mcts` vs `frequency_mcts` win rate | `0.00` | `experiments/neural_scaling_study/medium/aggregate_summary.json` |
| Human eval status | `dry_run_only` | `experiments/human_eval/aggregate_summary.json` |

Important interpretation:

- `frequency` is the final main bot, not because it dominates everything, but because it is the best cheap, stable, explainable belief mode.
- `particle world_scaled` can be discussed in the appendix as a budget-sensitive variant.
- Neural guidance is a valid negative/inconclusive result: the model/training/search wiring works, but the trained checkpoints do not currently improve gameplay over the chosen frequency baseline.
- Human-evaluation tooling exists, but real participant sessions still need to be run manually before claiming human-study findings.

---

## Proposal Mapping

This section maps each proposal item to what has been done and where to find it.

### 1. Setup

Proposal:

> Choose a simulator stack (Catanatron) and get deterministic runs + logging working.

Done:

- Simulator stack: Catanatron.
- This repo (`catan-ai-agent`) is the AI/control layer around the sibling engine repo (`../catanatron`).
- Deterministic seeds are used across MCTS, belief sweeps, neural studies, demo runs, and human-eval manifests.
- Browser demo integration now uses the existing Catanatron web UI.

Key files:

- `MCTS_PHASE_README.md`
- `reports/DEMO_RUNBOOK.md`
- `scripts/demo_preflight.py`
- `scripts/run_final_demo.py`
- `scripts/run_visual_match.py`
- `scripts/run_browser_human_vs_bot.py`

Sibling repo changes:

- `../catanatron/catanatron/catanatron/web/api.py`
- `../catanatron/ui/src/pages/HomePage.tsx`
- `../catanatron/ui/src/utils/apiClient.ts`
- `../catanatron/docker-compose.yml`

Status: complete for local demos and reproducible experiments.

---

### 2. Baseline Bots

Proposal:

> Implement 2-3 baseline bots (random, greedy, simple heuristics).

Done:

- Built-ins from Catanatron:
  - `RandomPlayer`
  - `WeightedRandomPlayer`
  - Catanatron built-in search bot (`CATANATRON` in UI)
- Project-owned bots:
  - `HeuristicBot`
  - `MCTSPlayer`
  - `BeliefMCTSPlayer`
  - `NeuralMCTSPlayer`

Key files:

- `src/catan_ai/players/heuristic_player.py`
- `src/catan_ai/players/mcts_player.py`
- `src/catan_ai/players/belief_mcts_player.py`
- `src/catan_ai/players/neural_mcts_player.py`

Status: complete.

---

### 3. State/Action Representation + Evaluation Harness

Proposal:

> Define state/action representations and evaluation harness (match runner + metrics).

Done:

- Public-state adapter and encoded action abstraction are implemented.
- Candidate filtering and decision context map stable encoded actions back to raw Catanatron actions.
- Arena/match runner tracks win rate, final VP, turns, and latency.
- Experiment summaries are written as JSON and CSV.

Key files:

- `src/catan_ai/adapters/public_state.py`
- `src/catan_ai/adapters/action_codec.py`
- `src/catan_ai/players/decision_context.py`
- `src/catan_ai/search/candidate_filter.py`
- `src/catan_ai/eval/arena.py`
- `scripts/run_mcts_match.py`
- `scripts/run_opponent_modeling_ablation.py`
- `scripts/run_opponent_modeling_sweep.py`
- `scripts/run_neural_scaling_study.py`

Status: complete.

---

### 4. Planning Agent v1

Proposal:

> Implement MCTS with configurable rollout policy and time/iteration budgets.

Done:

- MCTS implemented with:
  - simulation budget
  - max depth
  - exploration constant
  - candidate filter
  - deterministic tie-breaks
- Search player wrapper added.

Key files:

- `src/catan_ai/search/mcts.py`
- `src/catan_ai/search/tree_node.py`
- `src/catan_ai/search/leaf_evaluator.py`
- `src/catan_ai/players/mcts_player.py`

Status: complete.

---

### 5. Belief Sampling / Opponent Modeling Scaffolding

Proposal:

> Add determinization/belief sampling scaffolding (even if beliefs start simple).

Done:

- Belief/determinization layer implemented.
- Opponent-modeling modes:
  - `none`: plain MCTS
  - `frequency`: one count-only determinized world
  - `particle`: multi-world determinized belief sampling
- Diagnostics track belief activity, world sampling, rejection/fallback behavior, and update counts.

Key files:

- `src/catan_ai/belief/*`
- `src/catan_ai/players/belief_mcts_player.py`
- `src/catan_ai/eval/opponent_modeling.py`
- `src/catan_ai/eval/belief_diagnostics.py`

Experiments:

- `scripts/run_opponent_modeling_ablation.py`
- `scripts/run_opponent_modeling_sweep.py`
- `configs/opponent_modeling_sweep/small.yaml`
- `configs/opponent_modeling_sweep/medium.yaml`

Results:

- `experiments/opponent_modeling_sweep/medium/aggregate_summary.json`
- `experiments/opponent_modeling_sweep/medium/matchups.csv`

Status: complete enough for final project story.

Decision:

- Keep `frequency`.
- Keep `particle world_scaled` as appendix.
- Do not use `particle compute_matched` as main.

---

### 6. Learning Component + Self-Play

Proposal:

> Add policy/value model and self play data generation.

Done:

- Self-play data generation implemented.
- Policy/value network implemented.
- Dataset, collate, training loop, and checkpoint loading implemented.
- Mini neural experiment and neural scaling study implemented.
- Model diagnostics report policy entropy, flat-policy fraction, top-1 target match rate, and value error.

Key files:

- `src/catan_ai/training/self_play.py`
- `src/catan_ai/training/dataset.py`
- `src/catan_ai/training/collate.py`
- `src/catan_ai/training/train_policy_value.py`
- `src/catan_ai/training/checkpoints.py`
- `src/catan_ai/models/policy_value_net.py`
- `src/catan_ai/eval/model_diagnostics.py`

Experiment drivers:

- `scripts/run_mini_neural_experiment.py`
- `scripts/run_neural_scaling_study.py`

Configs:

- `configs/mini_neural_exp/`
- `configs/neural_scaling_study/small.yaml`
- `configs/neural_scaling_study/medium.yaml`
- `configs/neural_scaling_study/large.yaml`

Results:

- `experiments/mini_neural_exp/summary.json`
- `experiments/neural_scaling_study/small/aggregate_summary.json`
- `experiments/neural_scaling_study/medium/aggregate_summary.json`

Status: implemented, but not a positive main result.

Decision:

- Neural guidance remains a secondary/negative-result track.
- Current medium result: `neural_frequency_mcts` win rate vs `frequency_mcts` is `0.00`.

---

### 7. Integrate Value Guidance Into MCTS + Controlled Experiments

Proposal:

> Integrate value guidance into MCTS and run controlled experiments (compute-matched).

Done:

- NeuralMCTS supports model priors and model value.
- A prior-influence bug was fixed: model priors now reorder unexpanded actions before progressive widening hides model preference.
- Regression tests and audit validate that neural priors can affect root behavior.
- Neural scaling study is compute-matched against `frequency_mcts`.

Key files:

- `src/catan_ai/players/neural_mcts_player.py`
- `tests/test_policy_value.py`
- `scripts/audit_neural_phase.py`
- `reports/neural_phase_audit.md`
- `scripts/run_neural_scaling_study.py`

Status: complete technically, but results are negative/inconclusive.

---

### 8. Midpoint Report / Bottleneck Diagnosis

Proposal:

> Midpoint report: which components help and what the bottlenecks are.

Done:

- Search and neural audit reports exist.
- Final bundle summarizes current conclusions.
- Opponent-modeling sweep separates compute-matched and world-scaled particle regimes.
- Neural diagnostics identify weak model influence/quality as a bottleneck.

Key reports:

- `reports/neural_phase_audit.md`
- `reports/final_results/FINAL_RESULTS.md`
- `reports/final_results/final_summary.json`
- `MCTS_PHASE_README.md`

Current bottlenecks:

- Neural policy often remains too flat.
- Value head is noisy at current data/training scale.
- Human results are not real yet, only dry-run workflow data.

Status: complete for handoff/writeup.

---

### 9. Opponent Modeling + Trade Logic

Proposal:

> Implement structured trade proposal + acceptance policy and evaluate impact on win rate and economy.

Done:

- Opponent modeling: yes.
- Trade logic: no.

Reason:

- Later evidence showed the project needed to freeze a final technical story and prioritize learning/human-eval/final packaging.
- Trading was explicitly deferred to avoid expanding scope and destabilizing the final demo.

How to phrase this in report:

- “Opponent modeling was implemented and evaluated. Structured trade proposal/acceptance was scoped out in favor of completing belief, neural, human-evaluation, and reproducibility milestones.”

Status: partially complete; trade logic remains future work.

---

### 10. Human Studies

Proposal:

> Human study: recruit participants, run sessions, collect survey data.

Done:

- Human-eval toolkit exists.
- Participant manifests, skill groups, per-game plans, survey schema, result schema, aggregation, and session status checks are implemented.
- Browser UI can run Human vs `MAIN_BOT` or `REFERENCE_BOT`.

Key files:

- `src/catan_ai/eval/human_eval.py`
- `src/catan_ai/eval/survey.py`
- `scripts/prepare_human_eval.py`
- `scripts/start_human_eval_session.py`
- `scripts/finalize_human_eval_session.py`
- `scripts/summarize_human_eval.py`
- `scripts/check_human_eval_status.py`
- `reports/human_eval_instructions.md`
- `reports/human_survey_template.md`
- `reports/DEMO_RUNBOOK.md`

Current status:

- Only dry-run fixture data is stored right now.
- Real participant sessions still need to be run.

Status: tooling complete; data collection pending.

---

### 11. Iterate Once Based On Early Evaluation

Proposal:

> Iterate once based on early evaluation (bug fixes, reward shaping, search budget tuning).

Done:

- MCTS candidate filtering/progressive widening tuned.
- Neural prior influence bug fixed.
- Belief diagnostics added after initial ablation.
- Opponent sweep added after particle underperformed.
- Neural scaling/diagnostics added after mini neural result was insufficient.
- Human-eval and demo operator layers added after final packaging.

Status: complete.

---

### 12. Final

Proposal:

> Finalize figures/tables, summarize results, and write failure analysis. Package code entry point and reproducible configs.

Done:

- Final bundle builder consolidates results into JSON/CSV/Markdown.
- Repro validator checks expected artifact structure.
- Submission checklist exists.
- Demo runbook exists.
- Browser UI integration exists.

Key files:

- `scripts/build_final_results_bundle.py`
- `scripts/reproduce_final_artifacts.py`
- `reports/final_results/final_summary.json`
- `reports/final_results/main_results_table.csv`
- `reports/final_results/appendix_results_table.csv`
- `reports/final_results/FINAL_RESULTS.md`
- `reports/final_results/SUBMISSION_CHECKLIST.md`
- `reports/DEMO_RUNBOOK.md`

Status: mostly complete. Needs real human data and final report prose.

---

## Repo Tour

### This Repo: `catan-ai-agent`

Main folders:

- `src/catan_ai/adapters/`: bridge from Catanatron state/actions to project representations.
- `src/catan_ai/search/`: MCTS tree search, candidate filtering, leaf evaluation.
- `src/catan_ai/players/`: playable bot wrappers.
- `src/catan_ai/belief/`: belief extraction and determinization.
- `src/catan_ai/models/`: neural policy/value model.
- `src/catan_ai/training/`: self-play data, dataset, collate, train, checkpoints.
- `src/catan_ai/eval/`: arena, opponent-modeling helpers, diagnostics, human-eval helpers.
- `scripts/`: experiment drivers, demo tools, human-eval tools, final bundle tools.
- `configs/`: experiment presets.
- `experiments/`: generated experiment outputs.
- `reports/`: final summaries, runbooks, templates.
- `tests/`: focused smoke/regression tests.

### Sibling Repo: `../catanatron`

Only modified for visual integration:

- `catanatron/catanatron/web/api.py`: backend can instantiate `MAIN_BOT` and `REFERENCE_BOT`.
- `ui/src/pages/HomePage.tsx`: dropdown shows new bot choices.
- `ui/src/utils/apiClient.ts`: TypeScript player type includes new bot keys.
- `docker-compose.yml`: mounts `../catan-ai-agent` into backend container and sets `PYTHONPATH`.
- `README.md`: documents local custom bot integration.

---

## How To Run

### 1. Python Environment

From `../catanatron`:

```powershell
pip install -e ".[web,gym,dev]"
```

From `catan-ai-agent`:

```powershell
pip install -e ".[dev]"
pip install -e .
```

Note: for browser UI local Node builds, Catanatron UI requires Node `>=24 <25`. Docker is the easiest path.

---

### 2. Run Tests

In `catan-ai-agent`:

```powershell
python -m pytest -q
```

Latest known result:

- `97 passed`

In `../catanatron`, focused web API tests:

```powershell
python -m pytest tests/web/test_api.py -q -o addopts=--disable-warnings
```

Why the `-o addopts=...` override?

- This local machine did not have the pytest benchmark plugin installed, but `pytest.ini` references benchmark options.

Latest known result:

- `11 passed`

---

### 3. Final Results Bundle

Rebuild final report artifacts:

```powershell
python scripts/build_final_results_bundle.py
```

Validate expected artifacts:

```powershell
python scripts/reproduce_final_artifacts.py
```

Strict validation:

```powershell
python scripts/reproduce_final_artifacts.py --strict
```

Outputs:

- `reports/final_results/final_summary.json`
- `reports/final_results/FINAL_RESULTS.md`
- `reports/final_results/main_results_table.csv`
- `reports/final_results/appendix_results_table.csv`
- `reports/final_results/SUBMISSION_CHECKLIST.md`

---

### 4. Demo Preflight

```powershell
python scripts/demo_preflight.py
python scripts/demo_preflight.py --strict
```

Output:

- `reports/final_results/demo_preflight.json`

Latest known status:

- Required checks pass.
- Optional Catanatron CLI was not on PATH on the development machine.
- Docker CLI was present.

---

### 5. Browser Visual Demo

Start the Catanatron web stack from `../catanatron`:

```powershell
docker compose up --build
```

Open:

```text
http://localhost:3000
```

The dropdown includes:

- `Main Bot (Frequency MCTS)`
- `Reference Bot (Plain MCTS)`

Watch bot-vs-bot in browser:

```powershell
cd ..\catan-ai-agent
python scripts/run_visual_match.py --matchup main_vs_reference
```

Dry-run:

```powershell
python scripts/run_visual_match.py --matchup main_vs_reference --dry-run
```

Start Human vs Main Bot from script:

```powershell
python scripts/run_browser_human_vs_bot.py --bot MAIN_BOT --human-color RED
```

Or manually in UI:

1. Open `http://localhost:3000`.
2. Set one player to `Human`.
3. Set the other to `Main Bot (Frequency MCTS)`.
4. Click Start.

---

### 6. Human Evaluation Workflow

Prepare a participant:

```powershell
python scripts/prepare_human_eval.py --participant-id P001 --skill-group B
```

Generate operator checklist and per-game commands:

```powershell
python scripts/start_human_eval_session.py --manifest experiments/human_eval/sessions/P001/manifest.json
```

This writes:

- `experiments/human_eval/sessions/P001/start_checklist.json`
- `experiments/human_eval/sessions/P001/session_commands.json`
- `experiments/human_eval/sessions/P001/session_commands.md`

Use `session_commands.md` during the session.

Finalize after real games:

```powershell
python scripts/finalize_human_eval_session.py --manifest experiments/human_eval/sessions/P001/manifest.json --results-json experiments/human_eval/sessions/P001/results.json --surveys-json experiments/human_eval/sessions/P001/surveys.json
```

Summarize all sessions:

```powershell
python scripts/summarize_human_eval.py
python scripts/check_human_eval_status.py
python scripts/build_final_results_bundle.py
```

Status output:

- `experiments/human_eval/session_status.json`

---

### 7. Rerun Experiments If Needed

Opponent-modeling sweep:

```powershell
python scripts/run_opponent_modeling_sweep.py --preset small
python scripts/run_opponent_modeling_sweep.py --preset medium
```

Neural scaling study:

```powershell
python scripts/run_neural_scaling_study.py --preset small
python scripts/run_neural_scaling_study.py --preset medium
```

Mini neural experiment:

```powershell
python scripts/run_mini_neural_experiment.py
```

Usually, teammates should not rerun everything unless they need fresh evidence. The generated artifacts already exist.

---

## Important Artifacts

### Read These First

1. `reports/final_results/FINAL_RESULTS.md`
2. `reports/final_results/final_summary.json`
3. `reports/DEMO_RUNBOOK.md`
4. `MCTS_PHASE_README.md`

### Main Evidence

- `experiments/opponent_modeling_sweep/medium/aggregate_summary.json`
- `experiments/opponent_modeling_sweep/medium/matchups.csv`
- `experiments/neural_scaling_study/medium/aggregate_summary.json`
- `experiments/human_eval/aggregate_summary.json`

### Human Evaluation

- `reports/human_eval_instructions.md`
- `reports/human_survey_template.md`
- `experiments/human_eval/session_status.json`

### Browser Demo

- `scripts/run_visual_match.py`
- `scripts/run_browser_human_vs_bot.py`
- `reports/DEMO_RUNBOOK.md`

---

## What Teammates Should Do Next

### Highest Priority

1. Run real human participant sessions.
2. Summarize human results.
3. Rebuild the final results bundle.
4. Update final report/slides with real human data.

Commands after real sessions:

```powershell
python scripts/summarize_human_eval.py
python scripts/check_human_eval_status.py
python scripts/build_final_results_bundle.py
```

### Demo Rehearsal

Before presenting:

```powershell
python scripts/demo_preflight.py --strict
cd ..\catanatron
docker compose up --build
```

Then from `catan-ai-agent`:

```powershell
python scripts/run_visual_match.py --matchup main_vs_reference --dry-run
python scripts/run_browser_human_vs_bot.py --bot MAIN_BOT --human-color RED --dry-run
```

If dry-runs are correct, run the real commands without `--dry-run`.

### Final Report Work

Use this story:

- We built a baseline MCTS bot.
- We added belief/determinization opponent modeling.
- We evaluated `none`, `frequency`, and `particle`.
- Frequency belief MCTS became the main final bot.
- Particle belief only looked viable with scaled budget, so it is appendix-only.
- Neural policy/value guidance was implemented and audited, but current results did not improve over frequency MCTS.
- Human-eval infrastructure exists; real sessions should be used for final human-study claims.

---

## Known Caveats

- Trade proposal/acceptance logic from the proposal was not implemented. It is future work.
- Neural results are not positive yet. Do not oversell them.
- Current human-eval aggregate is dry-run-only. Do not claim real participant findings until sessions are run.
- Catanatron UI local build requires Node `>=24 <25`; Docker is recommended.
- The browser integration assumes the backend can import `catan_ai`. Docker compose handles this by mounting `../catan-ai-agent` and setting `PYTHONPATH=/catan-ai-agent/src`.

---

## If You Only Have 10 Minutes

Read:

1. `reports/final_results/FINAL_RESULTS.md`
2. `reports/DEMO_RUNBOOK.md`
3. This README’s “What Teammates Should Do Next” section.

Run:

```powershell
python scripts/demo_preflight.py
python scripts/reproduce_final_artifacts.py
```

Then start the UI:

```powershell
cd ..\catanatron
docker compose up --build
```

Open:

```text
http://localhost:3000
```

Choose:

- `Human` vs `Main Bot (Frequency MCTS)` for participant/demo play.
- `Main Bot (Frequency MCTS)` vs `Reference Bot (Plain MCTS)` for bot-vs-bot viewing.

