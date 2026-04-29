# Search Phases README (MCTS -> Belief -> Neural)

This document now tracks the search stack across three completed phases:

1. **MCTS v1** (single-world scaffold)
2. **BeliefMCTS** (determinized multi-world wrapper)
3. **Neural phase** (self-play + policy/value + NeuralMCTS)

This repo remains an AI/control layer around `../catanatron` and does **not** reimplement the game engine.

---

## Quick status snapshot

| Phase | Status | Main result |
|---|---|---|
| MCTS v1 | Complete | Strong baseline vs Debug/Random; competitive vs Heuristic |
| BeliefMCTS | Complete | Architecturally correct wrapper; in 2-player mostly limited gain |
| Neural phase | Implemented + audited | End-to-end pipeline works; fixed-state search influence now passes, with one smoke-model quality caveat |

Neural audit artifacts:
- `reports/neural_phase_audit.md`
- `reports/neural_phase_audit.json`

---

## Phase 1: MCTS v1 (single-world)

### What was added
- `src/catan_ai/search/*` (`TreeNode`, `CandidateFilter`, `evaluate_leaf`, `MCTS`)
- `src/catan_ai/players/mcts_player.py`
- `scripts/run_mcts_match.py`
- `tests/test_mcts.py`

### Core behavior
1. Build root `DecisionContext` from live `game + playable_actions`.
2. Search over copied states with UCT selection, one-action expansion, heuristic leaf value.
3. Choose root action by highest visit count (deterministic tie-break).
4. Map encoded action back to exact raw root playable action.

### Key defaults (`MCTSPlayer`)
- `max_simulations=50`
- `max_depth=10`
- `exploration_c=1.41`
- `top_k_roads=3`
- `top_k_trades=2`
- `top_k_robber=4`

### MCTS v1 benchmark snapshot
| Matchup | Games | Record | Win rate | Avg turns | Avg move latency |
|---|---:|---|---:|---:|---:|
| MCTSPlayer vs DebugPlayer | 20 | 16W / 3L / 1D | 80% | 244.4 | 11.6 ms |
| MCTSPlayer vs HeuristicBot | 20 | 9W / 11L / 0D | 45% | 150.2 | 10.0 ms |
| MCTSPlayer vs RandomPlayer | 20 | 19W / 1L / 0D | 95% | 158.5 | 11.4 ms |

### Phase 1 audit summary

This phase was audited against four practical criteria:
1. MCTS should beat `DebugPlayer` comfortably.
2. MCTS should be at least competitive with `HeuristicBot`.
3. Root visit counts should be meaningful (not flat/random-looking).
4. Search latency should remain controlled under moderate budgets.

Status:
- **PASS** Beats `DebugPlayer` and `RandomPlayer` by large margins in repeated batches.
- **PASS** Competitive vs `HeuristicBot` (not dominant, but not obviously weaker baseline behavior).
- **PASS** Root-child visit distributions are non-flat after filter+widening tuning.
- **PASS** Latency remained practical in tested budgets (roughly 10-20 ms range in sampled runs).

### What worked in Phase 1
- Deterministic search behavior for fixed seed + budget.
- No live-game mutation during search (`Game.copy()` tree simulation only).
- Stable encoded-action mapping back to raw root actions.
- Candidate filtering + progressive widening improved signal concentration and latency.

### Bugs fixed in Phase 1
- **EncodedAction sorting bug** in candidate filtering:
  - Symptom: `TypeError: '<' not supported between instances of 'EncodedAction'`.
  - Fix: switched tie-break sorting to `ActionCodec.sort_key(...)`.
- **Package circular import bug** involving `MCTSPlayer`/search imports:
  - Symptom: partial initialization import error.
  - Fix: lazy imports in `src/catan_ai/players/__init__.py`.

---

## Phase 2: BeliefMCTS (determinized wrapper)

### What changed
- Added belief extraction and determinizer (`src/catan_ai/belief/*`)
- Added `BeliefMCTSPlayer`
- Added belief tests and runner

### Main takeaway (2-player)
- Resource identities are mostly constrained by conservation.
- Most uncertainty/value from belief search comes from dev-card uncertainty.
- Equal-time checks still favored single-world MCTS at this budget.

### Phase 2 audit summary

This phase was audited with the requested four concrete checks:
1. **Equal-time comparison** (`BeliefMCTS` vs single-world `MCTSPlayer` under same `move_time_ms` cap).
2. **Dev-card sampling ablation** (ON vs OFF).
3. **Manual sampled-world audit** (public state invariants preserved, hidden assignments vary only where allowed).
4. **Rejection/fallback robustness stats** (invalid sample rate and fallback behavior).

Status:
- **PASS** Equal-time run completed and confirmed expected tradeoff (single-world MCTS remained stronger in 2-player budget setting).
- **PASS** Dev-card ON/OFF ablation completed; impact was measurable but modest.
- **PASS** Manual world audit confirmed only hidden opponent/dev-deck fields changed.
- **PASS** Rejection/fallback stats were healthy (low/zero invalid samples in audited settings).

### What worked in Phase 2
- Determinized world sampling preserves public consistency constraints.
- Own hand/public board/legal-root-actions invariants are maintained across sampled worlds.
- Aggregation of root stats across worlds is deterministic and stable.
- Belief-layer instrumentation made robustness observable (attempted/used/invalid/fallback).

### Bugs fixed in Phase 2
- **Dev-card unseen-pool accounting bug**:
  - Symptom: high invalid-sample behavior when dev-card sampling enabled.
  - Root cause: acting player's already-played dev cards were omitted from unseen-pool accounting.
  - Fix: added/propagated `own_played_dev_cards` in public evidence and included it in `DevCardBelief`.
- **Belief time-budget propagation bug**:
  - Symptom: outer `move_time_ms` was not effectively constraining per-world inner searches.
  - Fix: propagate remaining time budget from `BeliefMCTSPlayer` to inner `MCTS` calls.

---

## Phase 3: Neural phase

### What was added
- `src/catan_ai/models/*`
- `src/catan_ai/training/*`
- `src/catan_ai/players/neural_mcts_player.py`
- `src/catan_ai/eval/arena.py`
- `scripts/run_self_play.py`
- `scripts/train_policy_value.py`
- `scripts/run_neural_mcts_match.py`
- `tests/test_self_play.py`
- `tests/test_policy_value.py`

### Design choice used
No giant fixed action vocabulary.  
Model scores **current legal actions** conditioned on state:
- state features from `PublicState`
- action features from `EncodedAction`
- one logit per legal action + one state value

### Neural defaults (`NeuralMCTSConfig`)
- `max_simulations=50`
- `max_depth=10`
- `puct_c=2.5`
- `top_k_roads=3`
- `top_k_trades=2`
- `top_k_robber=4`
- `use_model_priors=True`
- `use_model_value=True`

---

## Neural phase audit summary (concrete runtime audit)

This section summarizes the exact audit that was just run end-to-end.

| Audit section | Result |
|---|---|
| Phase file inventory and architecture boundary | PASS |
| Test suite results | PASS |
| Self-play sample audit | PASS |
| Dataset / collate / mask audit | PASS |
| Tiny overfit and checkpoint audit | PASS |
| Action-ordering consistency audit | PASS |
| Raw simulator leakage audit | PASS |
| NeuralMCTS fixed-state audit | PASS |
| Small arena benchmark audit | PASS |
| Model output quality audit | **FAIL** |

### What the audit confirmed as working
- Self-play data is generated and serialized correctly.
- Targets are normalized visit distributions (not one-hot action labels).
- Value targets are perspective-correct.
- Dataset/collate/mask behavior is aligned for variable legal-action counts.
- Training loop is trainable on tiny data; checkpoint save/load fidelity is correct.
- Action ordering remains deterministic and aligned between teacher targets and inference path.
- No disallowed raw simulator leakage into model/training core modules.
- Arena runs are reproducible under fixed seeds and complete without crashes/illegal actions.

### Current audit caveat
- **NeuralMCTS fixed-state check is now passing:** enabling priors/value changes root ranking or visit distribution on the audited branchable fixed states.
- **Model output quality caveat:** the latest smoke-trained model missed the heuristic "non-flat priors on most states" threshold by one sampled state, while still showing varied priors, varied values, and top-1 signal beyond the constant baseline.

---

## Bug found and fixed during audit

### Bug
`SelfPlayConfig.teacher_type` existed but was silently ignored in self-play generation.

### Fix (smallest possible)
In `src/catan_ai/training/self_play.py`, `run_self_play(...)` now validates `teacher_type`:
- currently supported: `teacher_type='mcts'`
- unsupported values now raise clear `ValueError` (fail-fast instead of silent mismatch)

### Why this matters
Prevents configuration drift where users think they are collecting one teacher type while actually always collecting MCTS targets.

---

## Repro commands

### Environment
```powershell
.\.venv\Scripts\Activate.ps1
cd ..\catanatron
pip install -e ".[web,gym,dev]"
cd ..\catan-ai-agent
pip install -e ".[dev]"
```

### Tests
```powershell
pytest -q
pytest tests/test_self_play.py tests/test_policy_value.py -q
```

### Full neural audit
```powershell
python scripts/audit_neural_phase.py
```

### Artifacts produced
- `reports/neural_phase_audit.md`
- `reports/neural_phase_audit.json`

---

## Mini neural experiment

The mini experiment is the first reproducible end-to-end checkpoint benchmark. It is larger than the audit smoke run, but still sized for a local laptop.

Run:
```powershell
python scripts/run_mini_neural_experiment.py
```

Configs:
- `configs/mini_neural_exp/self_play.yaml`
- `configs/mini_neural_exp/train.yaml`
- `configs/mini_neural_exp/eval.yaml`

Artifacts:
- `experiments/mini_neural_exp/data/self_play/shard_*.pt`
- `experiments/mini_neural_exp/checkpoints/best.pt`
- `experiments/mini_neural_exp/matchups.csv`
- `experiments/mini_neural_exp/summary.json`
- `reports/mini_neural_experiment_summary.json`

Read `reports/mini_neural_experiment_summary.json` first. The main line to inspect is `NeuralMCTS checkpoint vs MCTS`, which is compute-matched against plain MCTS using the same simulation budget, depth, candidate filter settings, and search seed.

---

## Opponent-modeling ablation

The opponent-modeling ablation compares the three proposal modes under matched search budgets:
- `none`: plain single-world MCTS, with no opponent-belief intervention.
- `frequency`: one count-only determinized world from simple frequency/resource-count beliefs.
- `particle`: multi-world belief sampling through determinized hidden states.

Run:
```powershell
python scripts/run_opponent_modeling_ablation.py
```

Useful smaller run:
```powershell
python scripts/run_opponent_modeling_ablation.py --games 1 --sims 12 --particle-worlds 3
```

Artifacts:
- `experiments/opponent_modeling_ablation/matchups.csv`
- `experiments/opponent_modeling_ablation/summary.json`
- `experiments/opponent_modeling_ablation/belief_diagnostics.json`

Read `summary.json` first for `frequency vs none`, `particle vs none`, and `particle vs frequency`. Then check `belief_diagnostics.json` to confirm belief modes were active: worlds attempted/used, invalid-sample rejection rate, fallback rate, and average worlds used per decision.

---

## Opponent-modeling sweep

The sweep turns the ablation into decision-making evidence by running the same matchups across seeds, simulation budgets, particle world counts, and two particle regimes:
- `compute_matched`: particle uses the same total simulation budget as `none` and `frequency`, so its budget is split across sampled worlds.
- `world_scaled`: particle total simulations scale with `particle_worlds`, so each sampled world receives the same per-world budget as the plain search.

Run quick validation:
```powershell
python scripts/run_opponent_modeling_sweep.py --preset small
```

Run the evidence preset:
```powershell
python scripts/run_opponent_modeling_sweep.py --preset medium
```

Useful overrides:
```powershell
python scripts/run_opponent_modeling_sweep.py --preset small --seeds 7100,7101 --sims 12,24 --particle-worlds 2,4
```

Configs:
- `configs/opponent_modeling_sweep/small.yaml`
- `configs/opponent_modeling_sweep/medium.yaml`

Artifacts:
- `experiments/opponent_modeling_sweep/<preset>/aggregate_summary.json`
- `experiments/opponent_modeling_sweep/<preset>/matchups.csv`
- `experiments/opponent_modeling_sweep/<preset>/belief_diagnostics.json`
- per-setting `summary.json`, `matchups.csv`, and `belief_diagnostics.json` under `sims_<n>/worlds_<k>/<regime>/`

Interpretation:
- `frequency vs none` answers whether simple count/frequency beliefs beat no opponent modeling.
- `particle vs none` under `compute_matched` answers whether determinization helps at equal total compute.
- `particle vs none` under `world_scaled` answers whether particle belief only needs enough per-world budget.
- The aggregate `recommendation` field summarizes whether to keep `none`, keep `frequency`, drop `particle`, or treat particle as budget-sensitive.

---

## Neural scaling study

The main final-project path now locks opponent modeling to:
- `frequency`: default non-neural baseline.
- `particle world_scaled`: optional appendix/experimental variant.
- `particle compute_matched`: not used as a main setting.

The neural scaling study asks whether learned priors/value improve on top of the chosen `frequency` baseline.

Run quick sanity:
```powershell
python scripts/run_neural_scaling_study.py --preset small
```

Run local evidence:
```powershell
python scripts/run_neural_scaling_study.py --preset medium
```

Run larger preset when time allows:
```powershell
python scripts/run_neural_scaling_study.py --preset large
```

Useful resume:
```powershell
python scripts/run_neural_scaling_study.py --preset medium --reuse-existing
```

Configs:
- `configs/neural_scaling_study/small.yaml`
- `configs/neural_scaling_study/medium.yaml`
- `configs/neural_scaling_study/large.yaml`

Artifacts:
- `experiments/neural_scaling_study/<preset>/aggregate_summary.json`
- `experiments/neural_scaling_study/<preset>/matchups.csv`
- `experiments/neural_scaling_study/<preset>/diagnostics.json`
- per-seed `summary.json`, `matchups.csv`, `diagnostics.json`, `checkpoints/best.pt`, and self-play shards under `seed_<N>/`

Interpretation:
- `win_rate_vs_frequency_mcts`: whether neural guidance improves gameplay over the locked frequency-belief baseline.
- `flat_policy_fraction`: high values mean the policy head is close to uniform over legal actions and may not guide search strongly.
- `top1_match_rate`: how often model argmax matches the MCTS target argmax on held-out self-play samples.
- `value_mae` / `value_mse`: value-head error against final outcome targets; high error means neural leaf evaluation is likely noisy.

---

## Human evaluation workflow

The final human-evaluation path uses:
- main bot: `frequency` belief MCTS.
- reference bot: plain `mcts`.
- appendix only: `particle world_scaled`.
- secondary/negative-result track only: neural guidance.

Prepare a participant session:
```powershell
python scripts/prepare_human_eval.py --participant-id P001 --skill-group B
```

Start/check the session plan:
```powershell
python scripts/start_human_eval_session.py --manifest experiments/human_eval/sessions/P001/manifest.json
```

Finalize with dry-run fixture data:
```powershell
python scripts/finalize_human_eval_session.py --manifest experiments/human_eval/sessions/P001/manifest.json --dry-run
```

Finalize a real session:
```powershell
python scripts/finalize_human_eval_session.py --manifest experiments/human_eval/sessions/P001/manifest.json --results-json path\to\results.json --surveys-json path\to\surveys.json
```

Summarize all completed sessions:
```powershell
python scripts/summarize_human_eval.py
```

Artifacts:
- `experiments/human_eval/sessions/<participant_id>/manifest.json`
- `experiments/human_eval/sessions/<participant_id>/start_checklist.json`
- `experiments/human_eval/sessions/<participant_id>/completed_session.json`
- `experiments/human_eval/aggregate_summary.json`
- `experiments/human_eval/results.csv`
- `experiments/human_eval/surveys.csv`

Read `experiments/human_eval/aggregate_summary.json` first. It reports participant counts, skill-group coverage, completed games, main/reference bot win rates, average ratings by bot and skill group, and incomplete sessions.

Human-facing templates:
- `reports/human_eval_instructions.md`
- `reports/human_survey_template.md`

---

## Final results bundle

The authoritative submission/demo summary lives under `reports/final_results/`.

Final project positioning:
- main bot: `frequency` belief MCTS.
- reference bot: plain `mcts`.
- appendix experiment: `particle world_scaled`.
- secondary/negative-result track: neural guidance.
- pending manual work: real human participant sessions.

Rebuild the bundle from existing artifacts:
```powershell
python scripts/build_final_results_bundle.py
```

Validate expected artifact structure and rebuild:
```powershell
python scripts/reproduce_final_artifacts.py
```

Strict validation:
```powershell
python scripts/reproduce_final_artifacts.py --strict
```

Artifacts:
- `reports/final_results/final_summary.json`
- `reports/final_results/main_results_table.csv`
- `reports/final_results/appendix_results_table.csv`
- `reports/final_results/FINAL_RESULTS.md`
- `reports/final_results/SUBMISSION_CHECKLIST.md`

Read `reports/final_results/FINAL_RESULTS.md` first for the writeup narrative, then use `final_summary.json` and the CSV tables for report-ready metrics.

---

## Practical interpretation

As of this update:
- the neural pipeline is **correctly integrated and trainable** for a first pass,
- NeuralMCTS priors now affect root expansion before progressive widening can hide preferred actions,
- the fixed-state ranking/visit sensitivity gate now passes,
- and the current remaining caveat is smoke-model prior sharpness, not search wiring.

This is a solid baseline for next iteration: improve model quality and calibration without breaking the now-verified data/training/search-alignment path.
