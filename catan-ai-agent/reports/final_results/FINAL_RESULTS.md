# Final Results

## Final System Choice

- Main bot: `frequency belief MCTS`.
- Reference bot: `plain MCTS`.
- Particle `world_scaled` remains an appendix/experimental variant.
- Neural guidance remains a secondary/negative-result track for now.

## Main Findings

- Opponent modeling: Frequency belief MCTS is the default final setting; frequency-vs-none average win rate=0.750. Particle compute-matched is not a main setting; world-scaled particle is appendix-only (particle-vs-none average win rate=0.750).
- Frequency vs none mean win rate: `0.750`.

## Secondary Findings

- Particle world-scaled vs none mean win rate: `0.750`.
- Neural vs frequency mean win rate: `0.000`.
- Neural conclusion: Neural guidance is kept as a secondary/negative-result track; mean win rate vs frequency MCTS=0.000, top-1 match=0.451, flat-policy fraction=0.768.

## Human Evaluation Status

- Status: `dry_run_only`.
- Participants in current aggregate: `1`.
- Note: Only dry-run/fixture human-eval sessions are present; do not claim real participant results yet.

## Limitations

- Evidence is strongest for the scripted bot-vs-bot experiments already present in `experiments/`.
- Human-evaluation tooling is ready, but real participant data should replace dry-run data before final claims.
- Neural guidance is not presented as a positive main result because current scaling runs did not beat frequency MCTS.

## Reproducibility Notes

- Rebuild this bundle with `python scripts/build_final_results_bundle.py`.
- Validate expected artifacts with `python scripts/reproduce_final_artifacts.py`.
- Main rows are in `main_results_table.csv`; appendix rows are in `appendix_results_table.csv`.

## Table Counts

- Main result rows: `4`.
- Appendix result rows: `10`.
