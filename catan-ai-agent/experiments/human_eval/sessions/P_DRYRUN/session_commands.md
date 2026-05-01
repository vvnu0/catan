# Human Eval Session Commands: P_DRYRUN

- Skill group: `B`
- Session dir: `experiments\human_eval\sessions\P_DRYRUN`

## P_DRYRUN_game_01

- Bot faced: `frequency` (main)
- Seed: `9100`
- Human color: `RED`
- Bot color: `BLUE`
- Operator action: Start a local Catan game with seed 9100; participant=RED; bot=BLUE using frequency.
- Result file: `experiments\human_eval\sessions\P_DRYRUN\P_DRYRUN_game_01_result.json`
- Survey file: `experiments\human_eval\sessions\P_DRYRUN\P_DRYRUN_game_01_survey.json`

Completion checklist:
- [ ] Game completed or failure reason recorded
- [ ] Winner recorded as human, bot, or draw
- [ ] Human and bot final VP recorded
- [ ] Turns recorded if available
- [ ] Survey ratings recorded for this game
- [ ] Replay/log path recorded if available

## P_DRYRUN_game_02

- Bot faced: `mcts` (reference)
- Seed: `9101`
- Human color: `BLUE`
- Bot color: `RED`
- Operator action: Start a local Catan game with seed 9101; participant=BLUE; bot=RED using mcts.
- Result file: `experiments\human_eval\sessions\P_DRYRUN\P_DRYRUN_game_02_result.json`
- Survey file: `experiments\human_eval\sessions\P_DRYRUN\P_DRYRUN_game_02_survey.json`

Completion checklist:
- [ ] Game completed or failure reason recorded
- [ ] Winner recorded as human, bot, or draw
- [ ] Human and bot final VP recorded
- [ ] Turns recorded if available
- [ ] Survey ratings recorded for this game
- [ ] Replay/log path recorded if available

## Finalize

Run after all games: `python scripts/finalize_human_eval_session.py --manifest experiments\human_eval\sessions\P_DRYRUN\manifest.json --results-json experiments\human_eval\sessions\P_DRYRUN\results.json --surveys-json experiments\human_eval\sessions\P_DRYRUN\surveys.json`
