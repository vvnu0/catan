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
| Neural phase | Implemented + audited | End-to-end pipeline works; one targeted integration concern remains |

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
| NeuralMCTS fixed-state audit | **FAIL** |
| Small arena benchmark audit | PASS |
| Model output quality audit | PASS |

### What the audit confirmed as working
- Self-play data is generated and serialized correctly.
- Targets are normalized visit distributions (not one-hot action labels).
- Value targets are perspective-correct.
- Dataset/collate/mask behavior is aligned for variable legal-action counts.
- Training loop is trainable on tiny data; checkpoint save/load fidelity is correct.
- Action ordering remains deterministic and aligned between teacher targets and inference path.
- No disallowed raw simulator leakage into model/training core modules.
- Arena runs are reproducible under fixed seeds and complete without crashes/illegal actions.

### Remaining concern from audit
- **NeuralMCTS fixed-state check:** one required sub-check failed:
  - enabling priors/value did not always change root action ranking on the fixed audited states.
- Interpretation:
  - model wiring is active (priors present, legal actions valid),
  - but measurable influence is not yet consistently strong on those probes.

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

## Practical interpretation

As of this update:
- the neural pipeline is **correctly integrated and trainable** for a first pass,
- one integration quality gate still fails (fixed-state ranking sensitivity),
- and one real bug was fixed (`teacher_type` validation in self-play).

This is a solid baseline for next iteration: improve model impact in search (without breaking the now-verified data/training/alignment path).
