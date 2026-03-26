# Neural Phase Audit

Concrete end-to-end correctness audit of the self-play + policy/value + NeuralMCTS phase.

## PASS/FAIL Summary

| Section | Status |
|---|---|
| Phase file inventory and architecture boundary | PASS |
| Test suite results | PASS |
| Self-play sample audit | PASS |
| Dataset / collate / mask audit | PASS |
| Tiny overfit and checkpoint audit | PASS |
| Action-ordering consistency audit | PASS |
| Raw simulator leakage audit | PASS |
| NeuralMCTS fixed-state audit | FAIL |
| Small arena benchmark audit | PASS |
| Model output quality audit | PASS |

## Phase file inventory and architecture boundary

- **PASS** `All phase files exist` — missing=[]
- **PASS** `Repo boundary respected (no engine reimplementation, no ../catanatron edits)` — No edits outside repo were performed by this audit
- **PASS** `Model/training features based on PublicState + EncodedAction` — action_features imports only adapter dataclasses
- **PASS** `Raw catanatron imports confined to expected files` — locations=['src/catan_ai/training/self_play.py', 'src/catan_ai/players/neural_mcts_player.py', 'src/catan_ai/eval/arena.py', 'scripts/run_neural_mcts_match.py', 'tests/test_policy_value.py']

### Details

```json
{
  "raw_import_locations": [
    "src/catan_ai/training/self_play.py",
    "src/catan_ai/players/neural_mcts_player.py",
    "src/catan_ai/eval/arena.py",
    "scripts/run_neural_mcts_match.py",
    "tests/test_policy_value.py"
  ]
}
```

## Test suite results

- **PASS** `pytest -q passes`
- **PASS** `pytest tests/test_self_play.py tests/test_policy_value.py -q passes`

### Details

```json
{
  "pytest_q_output": "........................................................................ [ 83%]\n..............                                                           [100%]\n86 passed in 25.38s",
  "pytest_neural_output": ".........................                                                [100%]\n25 passed in 22.85s",
  "full_pass_count": 86,
  "target_pass_count": 25,
  "flaky_signals": "none observed in single-run audit"
}
```

## Self-play sample audit

- **PASS** `target_policy is normalized visit distribution (not one-hot)` — norms_ok=True, non_one_hot_signal=True
- **PASS** `target_value reflects acting-player perspective` — {"game_id": 0, "final_winner_inferred": "BLUE", "samples": [{"acting_color": "RED", "target_value": -1.0}, {"acting_color": "BLUE", "target_value": 1.0}]}
- **PASS** `encoded actions / action features / target_policy align index-by-index`
- **PASS** `No raw Catanatron objects serialized in samples`
- **PASS** `Self-play produced nontrivial sample count` — samples=363

### Details

```json
{
  "output_dir": "C:\\Users\\nairv\\Downloads\\classes\\catan\\catan-ai-agent\\data\\audit_self_play_smoke",
  "num_shards": 1,
  "num_samples": 363,
  "first_5_samples": [
    {
      "game_id": 0,
      "ply_index": 1,
      "acting_color": "RED",
      "chosen_encoded_action": "RED:BUILD_SETTLEMENT(0)",
      "num_encoded_legal_actions": 54,
      "num_action_feature_rows": 54,
      "target_policy_length": 54,
      "target_policy_sum": 1.0,
      "target_policy_min": 0.0,
      "target_policy_max": 0.3333333432674408,
      "target_policy_argmax": 0,
      "target_value": -1.0,
      "contains_raw_sim_object": false
    },
    {
      "game_id": 0,
      "ply_index": 2,
      "acting_color": "RED",
      "chosen_encoded_action": "RED:BUILD_ROAD((0,1))",
      "num_encoded_legal_actions": 3,
      "num_action_feature_rows": 3,
      "target_policy_length": 3,
      "target_policy_sum": 1.0,
      "target_policy_min": 0.3333333432674408,
      "target_policy_max": 0.3333333432674408,
      "target_policy_argmax": 0,
      "target_value": -1.0,
      "contains_raw_sim_object": false
    },
    {
      "game_id": 0,
      "ply_index": 3,
      "acting_color": "RED",
      "chosen_encoded_action": "RED:BUILD_SETTLEMENT(14)",
      "num_encoded_legal_actions": 43,
      "num_action_feature_rows": 43,
      "target_policy_length": 43,
      "target_policy_sum": 1.0,
      "target_policy_min": 0.0,
      "target_policy_max": 0.3333333432674408,
      "target_policy_argmax": 0,
      "target_value": -1.0,
      "contains_raw_sim_object": false
    },
    {
      "game_id": 0,
      "ply_index": 4,
      "acting_color": "RED",
      "chosen_encoded_action": "RED:BUILD_ROAD((13,14))",
      "num_encoded_legal_actions": 3,
      "num_action_feature_rows": 3,
      "target_policy_length": 3,
      "target_policy_sum": 1.0,
      "target_policy_min": 0.3333333432674408,
      "target_policy_max": 0.3333333432674408,
      "target_policy_argmax": 0,
      "target_value": -1.0,
      "contains_raw_sim_object": false
    },
    {
      "game_id": 0,
      "ply_index": 10,
      "acting_color": "RED",
      "chosen_encoded_action": "RED:MOVE_ROBBER(((-1,-1,2),BLUE))",
      "num_encoded_legal_actions": 18,
      "num_action_feature_rows": 18,
      "target_policy_length": 18,
      "target_policy_sum": 1.0,
      "target_policy_min": 0.0,
      "target_policy_max": 0.3333333432674408,
      "target_policy_argmax": 0,
      "target_value": -1.0,
      "contains_raw_sim_object": false
    }
  ],
  "perspective_demo": {
    "game_id": 0,
    "final_winner_inferred": "BLUE",
    "samples": [
      {
        "acting_color": "RED",
        "target_value": -1.0
      },
      {
        "acting_color": "BLUE",
        "target_value": 1.0
      }
    ]
  }
}
```

## Dataset / collate / mask audit

- **PASS** `Mask sums match legal-action counts` — legal=[2, 5, 9, 54], mask=[2, 5, 9, 54]
- **PASS** `Padded actions get zero probability after masking` — padded_mass=[0.0, 0.0, 0.0, 0.0]
- **PASS** `Padded target_policy entries are zero` — padded_target_abs_sum=[0.0, 0.0, 0.0, 0.0]
- **PASS** `Variable-length batching aligns without shape errors` — action_shape=[4, 54, 19]
- **PASS** `Model forward supports variable legal-action counts`

### Details

```json
{
  "state_tensor_shape": [
    4,
    53
  ],
  "action_tensor_shape": [
    4,
    54,
    19
  ],
  "action_mask_shape": [
    4,
    54
  ],
  "target_policy_shape": [
    4,
    54
  ],
  "target_value_shape": [
    4
  ],
  "legal_action_count_per_sample": [
    2,
    5,
    9,
    54
  ],
  "mask_sum_per_sample": [
    2,
    5,
    9,
    54
  ],
  "policy_logits_shape": [
    4,
    54
  ],
  "value_output_shape": [
    4
  ],
  "padded_action_probability_mass_per_row": [
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "padded_target_policy_abs_sum_per_row": [
    0.0,
    0.0,
    0.0,
    0.0
  ]
}
```

## Tiny overfit and checkpoint audit

- **PASS** `Model can overfit tiny dataset`
- **PASS** `Policy loss drops substantially` — before=2.0012, after=1.9207
- **PASS** `Value loss drops substantially` — before=1.1226, after=0.0001
- **PASS** `Checkpoint reload preserves outputs on fixed batch` — logits_diff=0.000e+00, value_diff=0.000e+00

### Details

```json
{
  "num_samples_used": 64,
  "initial_policy_loss": 2.0012357234954834,
  "final_policy_loss": 1.9207054376602173,
  "initial_value_loss": 1.122551441192627,
  "final_value_loss": 8.317181345773861e-05,
  "top1_agreement_before": 0.46875,
  "top1_agreement_after": 0.859375,
  "checkpoint_max_abs_logits_diff": 0.0,
  "checkpoint_max_abs_value_diff": 0.0,
  "checkpoint_path": "C:\\Users\\nairv\\Downloads\\classes\\catan\\catan-ai-agent\\reports\\_audit_tmp_checkpoint.pt"
}
```

## Action-ordering consistency audit

- **PASS** `Training uses deterministic encoded-action ordering consistent with inference`
- **PASS** `Saved target_policy aligns with teacher visit counts in same order`
- **PASS** `NeuralMCTSPlayer scores actions using DecisionContext encoded ordering` — neural_mcts_player uses node.context.encoded_actions
- **PASS** `Action filtering config/path is consistent between self-play teacher and neural search` — both use CandidateFilter with top_k_roads/trades/robber

### Details

```json
{
  "sample_meta": {
    "game_id": 0,
    "acting_color": "RED",
    "ply": 1
  },
  "saved_encoded_legal_actions": [
    "RED:BUILD_SETTLEMENT(0)",
    "RED:BUILD_SETTLEMENT(1)",
    "RED:BUILD_SETTLEMENT(10)",
    "RED:BUILD_SETTLEMENT(11)",
    "RED:BUILD_SETTLEMENT(12)",
    "RED:BUILD_SETTLEMENT(13)",
    "RED:BUILD_SETTLEMENT(14)",
    "RED:BUILD_SETTLEMENT(15)",
    "RED:BUILD_SETTLEMENT(16)",
    "RED:BUILD_SETTLEMENT(17)",
    "RED:BUILD_SETTLEMENT(18)",
    "RED:BUILD_SETTLEMENT(19)",
    "RED:BUILD_SETTLEMENT(2)",
    "RED:BUILD_SETTLEMENT(20)",
    "RED:BUILD_SETTLEMENT(21)",
    "RED:BUILD_SETTLEMENT(22)",
    "RED:BUILD_SETTLEMENT(23)",
    "RED:BUILD_SETTLEMENT(24)",
    "RED:BUILD_SETTLEMENT(25)",
    "RED:BUILD_SETTLEMENT(26)",
    "RED:BUILD_SETTLEMENT(27)",
    "RED:BUILD_SETTLEMENT(28)",
    "RED:BUILD_SETTLEMENT(29)",
    "RED:BUILD_SETTLEMENT(3)",
    "RED:BUILD_SETTLEMENT(30)",
    "RED:BUILD_SETTLEMENT(31)",
    "RED:BUILD_SETTLEMENT(32)",
    "RED:BUILD_SETTLEMENT(33)",
    "RED:BUILD_SETTLEMENT(34)",
    "RED:BUILD_SETTLEMENT(35)",
    "RED:BUILD_SETTLEMENT(36)",
    "RED:BUILD_SETTLEMENT(37)",
    "RED:BUILD_SETTLEMENT(38)",
    "RED:BUILD_SETTLEMENT(39)",
    "RED:BUILD_SETTLEMENT(4)",
    "RED:BUILD_SETTLEMENT(40)",
    "RED:BUILD_SETTLEMENT(41)",
    "RED:BUILD_SETTLEMENT(42)",
    "RED:BUILD_SETTLEMENT(43)",
    "RED:BUILD_SETTLEMENT(44)",
    "RED:BUILD_SETTLEMENT(45)",
    "RED:BUILD_SETTLEMENT(46)",
    "RED:BUILD_SETTLEMENT(47)",
    "RED:BUILD_SETTLEMENT(48)",
    "RED:BUILD_SETTLEMENT(49)",
    "RED:BUILD_SETTLEMENT(5)",
    "RED:BUILD_SETTLEMENT(50)",
    "RED:BUILD_SETTLEMENT(51)",
    "RED:BUILD_SETTLEMENT(52)",
    "RED:BUILD_SETTLEMENT(53)",
    "RED:BUILD_SETTLEMENT(6)",
    "RED:BUILD_SETTLEMENT(7)",
    "RED:BUILD_SETTLEMENT(8)",
    "RED:BUILD_SETTLEMENT(9)"
  ],
  "teacher_root_actions": [
    "RED:BUILD_SETTLEMENT(0)",
    "RED:BUILD_SETTLEMENT(1)",
    "RED:BUILD_SETTLEMENT(10)",
    "RED:BUILD_SETTLEMENT(11)",
    "RED:BUILD_SETTLEMENT(12)",
    "RED:BUILD_SETTLEMENT(13)",
    "RED:BUILD_SETTLEMENT(14)",
    "RED:BUILD_SETTLEMENT(15)",
    "RED:BUILD_SETTLEMENT(16)",
    "RED:BUILD_SETTLEMENT(17)",
    "RED:BUILD_SETTLEMENT(18)",
    "RED:BUILD_SETTLEMENT(19)",
    "RED:BUILD_SETTLEMENT(2)",
    "RED:BUILD_SETTLEMENT(20)",
    "RED:BUILD_SETTLEMENT(21)",
    "RED:BUILD_SETTLEMENT(22)",
    "RED:BUILD_SETTLEMENT(23)",
    "RED:BUILD_SETTLEMENT(24)",
    "RED:BUILD_SETTLEMENT(25)",
    "RED:BUILD_SETTLEMENT(26)",
    "RED:BUILD_SETTLEMENT(27)",
    "RED:BUILD_SETTLEMENT(28)",
    "RED:BUILD_SETTLEMENT(29)",
    "RED:BUILD_SETTLEMENT(3)",
    "RED:BUILD_SETTLEMENT(30)",
    "RED:BUILD_SETTLEMENT(31)",
    "RED:BUILD_SETTLEMENT(32)",
    "RED:BUILD_SETTLEMENT(33)",
    "RED:BUILD_SETTLEMENT(34)",
    "RED:BUILD_SETTLEMENT(35)",
    "RED:BUILD_SETTLEMENT(36)",
    "RED:BUILD_SETTLEMENT(37)",
    "RED:BUILD_SETTLEMENT(38)",
    "RED:BUILD_SETTLEMENT(39)",
    "RED:BUILD_SETTLEMENT(4)",
    "RED:BUILD_SETTLEMENT(40)",
    "RED:BUILD_SETTLEMENT(41)",
    "RED:BUILD_SETTLEMENT(42)",
    "RED:BUILD_SETTLEMENT(43)",
    "RED:BUILD_SETTLEMENT(44)",
    "RED:BUILD_SETTLEMENT(45)",
    "RED:BUILD_SETTLEMENT(46)",
    "RED:BUILD_SETTLEMENT(47)",
    "RED:BUILD_SETTLEMENT(48)",
    "RED:BUILD_SETTLEMENT(49)",
    "RED:BUILD_SETTLEMENT(5)",
    "RED:BUILD_SETTLEMENT(50)",
    "RED:BUILD_SETTLEMENT(51)",
    "RED:BUILD_SETTLEMENT(52)",
    "RED:BUILD_SETTLEMENT(53)",
    "RED:BUILD_SETTLEMENT(6)",
    "RED:BUILD_SETTLEMENT(7)",
    "RED:BUILD_SETTLEMENT(8)",
    "RED:BUILD_SETTLEMENT(9)"
  ],
  "teacher_root_visit_counts": [
    5,
    5,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
  ],
  "saved_target_policy": [
    0.3333333432674408,
    0.3333333432674408,
    0.3333333432674408,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "teacher_policy_from_visits": [
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ]
}
```

## Raw simulator leakage audit

- **PASS** `Raw simulator access only in expected layers` — violations=0

### Details

```json
{
  "raw_access_references": [
    {
      "file": "src/catan_ai/training/self_play.py",
      "line": 22,
      "text": "from catanatron import Color, Game"
    },
    {
      "file": "src/catan_ai/training/self_play.py",
      "line": 23,
      "text": "from catanatron.models.player import Player"
    },
    {
      "file": "src/catan_ai/training/self_play.py",
      "line": 88,
      "text": "def decide(self, game, playable_actions):"
    },
    {
      "file": "src/catan_ai/training/self_play.py",
      "line": 91,
      "text": "if len(playable_actions) == 1:"
    },
    {
      "file": "src/catan_ai/training/self_play.py",
      "line": 92,
      "text": "return playable_actions[0]"
    },
    {
      "file": "src/catan_ai/training/self_play.py",
      "line": 95,
      "text": "root_ctx = DecisionContext(game, playable_actions, self.color)"
    },
    {
      "file": "src/catan_ai/training/self_play.py",
      "line": 174,
      "text": "game = Game([red, blue], seed=game_seed)"
    },
    {
      "file": "src/catan_ai/training/self_play.py",
      "line": 196,
      "text": "game.state.num_turns,"
    },
    {
      "file": "src/catan_ai/players/neural_mcts_player.py",
      "line": 27,
      "text": "from catanatron.game import Game"
    },
    {
      "file": "src/catan_ai/players/neural_mcts_player.py",
      "line": 28,
      "text": "from catanatron.models.player import Color, Player"
    },
    {
      "file": "src/catan_ai/players/neural_mcts_player.py",
      "line": 87,
      "text": "random.seed(self.cfg.seed + game.state.num_turns * 997)"
    },
    {
      "file": "src/catan_ai/players/neural_mcts_player.py",
      "line": 207,
      "text": "is_root_turn = node.game.state.current_color() == self.root_color"
    },
    {
      "file": "src/catan_ai/players/neural_mcts_player.py",
      "line": 283,
      "text": "acting = node.game.state.current_color()"
    },
    {
      "file": "src/catan_ai/players/neural_mcts_player.py",
      "line": 355,
      "text": "def decide(self, game, playable_actions):"
    },
    {
      "file": "src/catan_ai/players/neural_mcts_player.py",
      "line": 358,
      "text": "if len(playable_actions) == 1:"
    },
    {
      "file": "src/catan_ai/players/neural_mcts_player.py",
      "line": 359,
      "text": "return playable_actions[0]"
    },
    {
      "file": "src/catan_ai/players/neural_mcts_player.py",
      "line": 361,
      "text": "root_ctx = DecisionContext(game, playable_actions, self.color)"
    },
    {
      "file": "src/catan_ai/eval/arena.py",
      "line": 17,
      "text": "from catanatron import Color, Game"
    },
    {
      "file": "src/catan_ai/eval/arena.py",
      "line": 18,
      "text": "from catanatron.models.player import Player"
    },
    {
      "file": "src/catan_ai/eval/arena.py",
      "line": 119,
      "text": "game = Game(players, seed=seed)"
    },
    {
      "file": "src/catan_ai/eval/arena.py",
      "line": 131,
      "text": "result.turn_counts.append(game.state.num_turns)"
    },
    {
      "file": "scripts/run_neural_mcts_match.py",
      "line": 15,
      "text": "from catanatron import Color, RandomPlayer"
    },
    {
      "file": "tests/test_policy_value.py",
      "line": 13,
      "text": "from catanatron import Color, Game, RandomPlayer"
    },
    {
      "file": "tests/test_policy_value.py",
      "line": 34,
      "text": "game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)"
    },
    {
      "file": "tests/test_policy_value.py",
      "line": 269,
      "text": "game = Game([player, RandomPlayer(Color.BLUE)], seed=1)"
    },
    {
      "file": "tests/test_policy_value.py",
      "line": 281,
      "text": "game = Game([player, RandomPlayer(Color.BLUE)], seed=1)"
    },
    {
      "file": "tests/test_policy_value.py",
      "line": 293,
      "text": "game = Game("
    },
    {
      "file": "tests/test_policy_value.py",
      "line": 304,
      "text": "turns_before = game.state.num_turns"
    },
    {
      "file": "tests/test_policy_value.py",
      "line": 305,
      "text": "actions = game.playable_actions"
    },
    {
      "file": "tests/test_policy_value.py",
      "line": 308,
      "text": "player = game.state.current_player()"
    },
    {
      "file": "tests/test_policy_value.py",
      "line": 311,
      "text": "assert game.state.num_turns == turns_before"
    }
  ],
  "disallowed_violations": []
}
```

## NeuralMCTS fixed-state audit

- **PASS** `Chosen raw action is always from original live playable_actions`
- **PASS** `NeuralMCTS (priors/value off) matches plain MCTS, or explained` — matched plain MCTS exactly
- **FAIL** `Enabling priors/value changes root ranking on fixed state`
- **PASS** `Model is actively used in search (non-empty priors recorded)`

### Details

```json
{
  "fixed_state_comparisons": [
    {
      "seed": 777,
      "ticks": 45,
      "acting_color": "BLUE",
      "legal_ok": true,
      "off_matches_plain": true,
      "ranking_changed_with_model": false,
      "model_active": true,
      "plain": {
        "chosen_encoded_action": "BLUE:END_TURN",
        "chosen_raw_action": "Action(color=C.BLUE, action_type=AT.END_TURN, value=None)",
        "top5_root_actions": [
          [
            "BLUE:END_TURN",
            {
              "visits": 40,
              "avg_value": 0.0314
            }
          ]
        ]
      },
      "neural_off": {
        "chosen_encoded_action": "BLUE:END_TURN",
        "chosen_raw_action": "Action(color=C.BLUE, action_type=AT.END_TURN, value=None)",
        "top5_root_actions": [
          [
            "BLUE:END_TURN",
            {
              "visits": 40,
              "avg_value": 0.0339,
              "prior": 0.0
            }
          ]
        ]
      },
      "neural_on": {
        "chosen_encoded_action": "BLUE:END_TURN",
        "chosen_raw_action": "Action(color=C.BLUE, action_type=AT.END_TURN, value=None)",
        "top5_root_actions": [
          [
            "BLUE:END_TURN",
            {
              "visits": 40,
              "avg_value": 0.6862,
              "prior": 1.0
            }
          ]
        ],
        "root_value_estimate": 0.9884172677993774
      }
    },
    {
      "seed": 778,
      "ticks": 40,
      "acting_color": "RED",
      "legal_ok": true,
      "off_matches_plain": true,
      "ranking_changed_with_model": false,
      "model_active": true,
      "plain": {
        "chosen_encoded_action": "RED:ROLL",
        "chosen_raw_action": "Action(color=C.RED, action_type=AT.ROLL, value=None)",
        "top5_root_actions": [
          [
            "RED:ROLL",
            {
              "visits": 40,
              "avg_value": -0.1103
            }
          ]
        ]
      },
      "neural_off": {
        "chosen_encoded_action": "RED:ROLL",
        "chosen_raw_action": "Action(color=C.RED, action_type=AT.ROLL, value=None)",
        "top5_root_actions": [
          [
            "RED:ROLL",
            {
              "visits": 40,
              "avg_value": -0.1103,
              "prior": 0.0
            }
          ]
        ]
      },
      "neural_on": {
        "chosen_encoded_action": "RED:ROLL",
        "chosen_raw_action": "Action(color=C.RED, action_type=AT.ROLL, value=None)",
        "top5_root_actions": [
          [
            "RED:ROLL",
            {
              "visits": 40,
              "avg_value": -0.5411,
              "prior": 1.0
            }
          ]
        ],
        "root_value_estimate": 0.6839459538459778
      }
    },
    {
      "seed": 779,
      "ticks": 50,
      "acting_color": "BLUE",
      "legal_ok": true,
      "off_matches_plain": true,
      "ranking_changed_with_model": false,
      "model_active": true,
      "plain": {
        "chosen_encoded_action": "BLUE:ROLL",
        "chosen_raw_action": "Action(color=C.BLUE, action_type=AT.ROLL, value=None)",
        "top5_root_actions": [
          [
            "BLUE:ROLL",
            {
              "visits": 40,
              "avg_value": -0.0475
            }
          ]
        ]
      },
      "neural_off": {
        "chosen_encoded_action": "BLUE:ROLL",
        "chosen_raw_action": "Action(color=C.BLUE, action_type=AT.ROLL, value=None)",
        "top5_root_actions": [
          [
            "BLUE:ROLL",
            {
              "visits": 40,
              "avg_value": -0.0475,
              "prior": 0.0
            }
          ]
        ]
      },
      "neural_on": {
        "chosen_encoded_action": "BLUE:ROLL",
        "chosen_raw_action": "Action(color=C.BLUE, action_type=AT.ROLL, value=None)",
        "top5_root_actions": [
          [
            "BLUE:ROLL",
            {
              "visits": 40,
              "avg_value": 0.2284,
              "prior": 1.0
            }
          ]
        ],
        "root_value_estimate": 0.9403495788574219
      }
    }
  ]
}
```

## Small arena benchmark audit

- **PASS** `NeuralMCTSPlayer completes games without crashing`
- **PASS** `NeuralMCTSPlayer returns legal actions (no illegal-action crashes observed)`
- **PASS** `No obvious END_TURN spam in buildable states` — {"rate": 0.0, "end_turn_while_build_city_or_settlement": 0, "total_decisions": 7921}
- **PASS** `Arena results reproducible under fixed seeds` — first=(6, 2, 0) repeat=(6, 2, 0)

### Details

```json
{
  "neural_vs_debug": {
    "label": "NeuralMCTS vs DebugPlayer",
    "games": 8,
    "wins": 8,
    "losses": 0,
    "draws": 0,
    "turn_counts": [
      198,
      291,
      131,
      159,
      179,
      85,
      248,
      474
    ],
    "move_times": [
      21.358521976729943,
      18.389542775952982,
      17.179679271518953,
      20.780501356221972,
      18.68232333411773,
      24.517580274014588,
      24.92853883434577,
      22.09164611633748
    ]
  },
  "neural_vs_random": {
    "label": "NeuralMCTS vs RandomPlayer",
    "games": 8,
    "wins": 6,
    "losses": 2,
    "draws": 0,
    "turn_counts": [
      500,
      547,
      159,
      172,
      149,
      233,
      70,
      213
    ],
    "move_times": [
      19.398574311491274,
      23.67746287852176,
      16.507142308987994,
      20.838434438941615,
      19.943238121160878,
      25.124989462601224,
      19.692356636401563,
      17.611533094351614
    ]
  },
  "mcts_vs_debug": {
    "label": "MCTSPlayer vs DebugPlayer",
    "games": 8,
    "wins": 8,
    "losses": 0,
    "draws": 0,
    "turn_counts": [
      154,
      117,
      133,
      137,
      127,
      141,
      224,
      442
    ],
    "move_times": [
      10.201879370865734,
      11.184394660898867,
      7.2271964357544975,
      9.746837237740847,
      9.98074944363907,
      9.768467137432644,
      10.26656770740043,
      9.108156593097455
    ]
  },
  "neural_vs_random_repeat": {
    "label": "NeuralMCTS vs RandomPlayer (repeat)",
    "games": 8,
    "wins": 6,
    "losses": 2,
    "draws": 0,
    "turn_counts": [
      500,
      547,
      159,
      172,
      149,
      233,
      70,
      213
    ],
    "move_times": [
      19.64304831864409,
      23.42982218861792,
      17.306110678383938,
      20.583146219888597,
      20.503952018624265,
      24.760710542561508,
      19.746832749435463,
      18.51784604226353
    ]
  },
  "pathology_counters": {
    "end_turn_while_build_city_or_settlement": 0,
    "total_decisions": 7921
  }
}
```

## Model output quality audit

- **PASS** `Priors are not uniformly flat on most states` — max_probs=[0.018518518656492233, 0.12681978940963745, 0.2282189130783081, 0.2720021605491638, 0.34168586134910583, 0.5340031385421753, 0.3716244697570801, 0.37162521481513977, 0.36428558826446533, 0.05555564910173416]
- **PASS** `Priors vary across states` — max_probs=[0.018518518656492233, 0.12681978940963745, 0.2282189130783081, 0.2720021605491638, 0.34168586134910583, 0.5340031385421753, 0.3716244697570801, 0.37162521481513977, 0.36428558826446533, 0.05555564910173416]
- **PASS** `Value predictions vary across states` — value_range=2.0000
- **PASS** `Model learns beyond constant baseline` — top1_match=0.80

### Details

```json
{
  "num_states": 10,
  "top1_match_rate": 0.8,
  "max_pred_probabilities": [
    0.018518518656492233,
    0.12681978940963745,
    0.2282189130783081,
    0.2720021605491638,
    0.34168586134910583,
    0.5340031385421753,
    0.3716244697570801,
    0.37162521481513977,
    0.36428558826446533,
    0.05555564910173416
  ],
  "value_predictions": [
    -0.9956057667732239,
    -0.9999178647994995,
    -1.0,
    0.9999999403953552,
    1.0,
    -0.9998242855072021,
    -0.9998610019683838,
    -0.9999126195907593,
    0.9999822378158569,
    0.9999998211860657
  ],
  "state_rows": [
    {
      "meta": {
        "game_id": 0,
        "ply": 1,
        "color": "RED"
      },
      "pred_top5": [
        [
          "RED:BUILD_SETTLEMENT(0)",
          0.018518518656492233
        ],
        [
          "RED:BUILD_SETTLEMENT(1)",
          0.018518518656492233
        ],
        [
          "RED:BUILD_SETTLEMENT(10)",
          0.018518518656492233
        ],
        [
          "RED:BUILD_SETTLEMENT(11)",
          0.018518518656492233
        ],
        [
          "RED:BUILD_SETTLEMENT(12)",
          0.018518518656492233
        ]
      ],
      "teacher_top5": [
        [
          "RED:BUILD_SETTLEMENT(0)",
          0.3333333432674408
        ],
        [
          "RED:BUILD_SETTLEMENT(1)",
          0.3333333432674408
        ],
        [
          "RED:BUILD_SETTLEMENT(10)",
          0.3333333432674408
        ],
        [
          "RED:BUILD_SETTLEMENT(11)",
          0.0
        ],
        [
          "RED:BUILD_SETTLEMENT(12)",
          0.0
        ]
      ],
      "value_prediction": -0.9956057667732239
    },
    {
      "meta": {
        "game_id": 0,
        "ply": 73,
        "color": "RED"
      },
      "pred_top5": [
        [
          "RED:BUILD_ROAD((0,20))",
          0.12681978940963745
        ],
        [
          "RED:BUILD_ROAD((0,5))",
          0.12681978940963745
        ],
        [
          "RED:BUILD_ROAD((1,2))",
          0.12681978940963745
        ],
        [
          "RED:BUILD_ROAD((1,6))",
          0.12681978940963745
        ],
        [
          "RED:BUILD_ROAD((13,34))",
          0.12681978940963745
        ]
      ],
      "teacher_top5": [
        [
          "RED:BUILD_ROAD((0,20))",
          0.3333333432674408
        ],
        [
          "RED:BUILD_ROAD((14,15))",
          0.3333333432674408
        ],
        [
          "RED:END_TURN",
          0.3333333432674408
        ],
        [
          "RED:BUILD_ROAD((0,5))",
          0.0
        ],
        [
          "RED:BUILD_ROAD((1,2))",
          0.0
        ]
      ],
      "value_prediction": -0.9999178647994995
    },
    {
      "meta": {
        "game_id": 0,
        "ply": 154,
        "color": "RED"
      },
      "pred_top5": [
        [
          "RED:END_TURN",
          0.2282189130783081
        ],
        [
          "RED:MARITIME_TRADE((ORE,ORE,ORE,ORE,BRICK))",
          0.09647264331579208
        ],
        [
          "RED:MARITIME_TRADE((ORE,ORE,ORE,ORE,SHEEP))",
          0.09647264331579208
        ],
        [
          "RED:MARITIME_TRADE((ORE,ORE,ORE,ORE,WHEAT))",
          0.09647264331579208
        ],
        [
          "RED:MARITIME_TRADE((ORE,ORE,ORE,ORE,WOOD))",
          0.09647264331579208
        ]
      ],
      "teacher_top5": [
        [
          "RED:END_TURN",
          0.3333333432674408
        ],
        [
          "RED:MARITIME_TRADE((ORE,ORE,ORE,ORE,BRICK))",
          0.3333333432674408
        ],
        [
          "RED:MARITIME_TRADE((ORE,ORE,ORE,ORE,SHEEP))",
          0.3333333432674408
        ],
        [
          "RED:MARITIME_TRADE((ORE,ORE,ORE,ORE,WHEAT))",
          0.0
        ],
        [
          "RED:MARITIME_TRADE((ORE,ORE,ORE,ORE,WOOD))",
          0.0
        ]
      ],
      "value_prediction": -1.0
    },
    {
      "meta": {
        "game_id": 0,
        "ply": 73,
        "color": "BLUE"
      },
      "pred_top5": [
        [
          "BLUE:BUILD_SETTLEMENT(4)",
          0.2720021605491638
        ],
        [
          "BLUE:END_TURN",
          0.26303282380104065
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,BRICK))",
          0.11527669429779053
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,ORE))",
          0.11527625471353531
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,WHEAT))",
          0.11527625471353531
        ]
      ],
      "teacher_top5": [
        [
          "BLUE:BUILD_ROAD((14,15))",
          0.3333333432674408
        ],
        [
          "BLUE:BUILD_SETTLEMENT(4)",
          0.3333333432674408
        ],
        [
          "BLUE:END_TURN",
          0.3333333432674408
        ],
        [
          "BLUE:BUILD_ROAD((11,32))",
          0.0
        ],
        [
          "BLUE:BUILD_ROAD((13,34))",
          0.0
        ]
      ],
      "value_prediction": 0.9999999403953552
    },
    {
      "meta": {
        "game_id": 0,
        "ply": 146,
        "color": "BLUE"
      },
      "pred_top5": [
        [
          "BLUE:END_TURN",
          0.34168586134910583
        ],
        [
          "BLUE:MARITIME_TRADE((BRICK,BRICK,BRICK,BRICK,ORE))",
          0.08228929340839386
        ],
        [
          "BLUE:MARITIME_TRADE((BRICK,BRICK,BRICK,BRICK,SHEEP))",
          0.08228929340839386
        ],
        [
          "BLUE:MARITIME_TRADE((BRICK,BRICK,BRICK,BRICK,WHEAT))",
          0.08228929340839386
        ],
        [
          "BLUE:MARITIME_TRADE((BRICK,BRICK,BRICK,BRICK,WOOD))",
          0.08228929340839386
        ]
      ],
      "teacher_top5": [
        [
          "BLUE:END_TURN",
          0.3333333432674408
        ],
        [
          "BLUE:MARITIME_TRADE((BRICK,BRICK,BRICK,BRICK,ORE))",
          0.3333333432674408
        ],
        [
          "BLUE:MARITIME_TRADE((BRICK,BRICK,BRICK,BRICK,SHEEP))",
          0.3333333432674408
        ],
        [
          "BLUE:MARITIME_TRADE((BRICK,BRICK,BRICK,BRICK,WHEAT))",
          0.0
        ],
        [
          "BLUE:MARITIME_TRADE((BRICK,BRICK,BRICK,BRICK,WOOD))",
          0.0
        ]
      ],
      "value_prediction": 1.0
    },
    {
      "meta": {
        "game_id": 1,
        "ply": 40,
        "color": "RED"
      },
      "pred_top5": [
        [
          "RED:PLAY_KNIGHT_CARD",
          0.5340031385421753
        ],
        [
          "RED:ROLL",
          0.4659968316555023
        ]
      ],
      "teacher_top5": [
        [
          "RED:PLAY_KNIGHT_CARD",
          0.5333333611488342
        ],
        [
          "RED:ROLL",
          0.46666666865348816
        ]
      ],
      "value_prediction": -0.9998242855072021
    },
    {
      "meta": {
        "game_id": 1,
        "ply": 140,
        "color": "RED"
      },
      "pred_top5": [
        [
          "RED:END_TURN",
          0.3716244697570801
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,BRICK))",
          0.15709389746189117
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,ORE))",
          0.15709389746189117
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,SHEEP))",
          0.15709389746189117
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,WHEAT))",
          0.15709389746189117
        ]
      ],
      "teacher_top5": [
        [
          "RED:END_TURN",
          0.3333333432674408
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,BRICK))",
          0.3333333432674408
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,ORE))",
          0.3333333432674408
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,SHEEP))",
          0.0
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,WHEAT))",
          0.0
        ]
      ],
      "value_prediction": -0.9998610019683838
    },
    {
      "meta": {
        "game_id": 1,
        "ply": 224,
        "color": "RED"
      },
      "pred_top5": [
        [
          "RED:END_TURN",
          0.37162521481513977
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,WHEAT))",
          0.15709392726421356
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,BRICK))",
          0.1570936143398285
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,ORE))",
          0.1570936143398285
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,SHEEP))",
          0.1570936143398285
        ]
      ],
      "teacher_top5": [
        [
          "RED:END_TURN",
          0.3333333432674408
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,BRICK))",
          0.3333333432674408
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,ORE))",
          0.3333333432674408
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,SHEEP))",
          0.0
        ],
        [
          "RED:MARITIME_TRADE((WOOD,WOOD,WOOD,WOOD,WHEAT))",
          0.0
        ]
      ],
      "value_prediction": -0.9999126195907593
    },
    {
      "meta": {
        "game_id": 1,
        "ply": 111,
        "color": "BLUE"
      },
      "pred_top5": [
        [
          "BLUE:END_TURN",
          0.36428558826446533
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,BRICK))",
          0.15892860293388367
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,ORE))",
          0.15892860293388367
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,WHEAT))",
          0.15892860293388367
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,WOOD))",
          0.15892860293388367
        ]
      ],
      "teacher_top5": [
        [
          "BLUE:END_TURN",
          0.3333333432674408
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,BRICK))",
          0.3333333432674408
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,ORE))",
          0.3333333432674408
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,WHEAT))",
          0.0
        ],
        [
          "BLUE:MARITIME_TRADE((SHEEP,SHEEP,SHEEP,SHEEP,WOOD))",
          0.0
        ]
      ],
      "value_prediction": 0.9999822378158569
    },
    {
      "meta": {
        "game_id": 1,
        "ply": 199,
        "color": "BLUE"
      },
      "pred_top5": [
        [
          "BLUE:MOVE_ROBBER(((2,-2,0),None))",
          0.05555564910173416
        ],
        [
          "BLUE:MOVE_ROBBER(((2,0,-2),None))",
          0.05555564910173416
        ],
        [
          "BLUE:MOVE_ROBBER(((-1,-1,2),RED))",
          0.055555544793605804
        ],
        [
          "BLUE:MOVE_ROBBER(((-1,1,0),None))",
          0.055555544793605804
        ],
        [
          "BLUE:MOVE_ROBBER(((-1,2,-1),None))",
          0.055555544793605804
        ]
      ],
      "teacher_top5": [
        [
          "BLUE:MOVE_ROBBER(((-1,-1,2),RED))",
          0.3333333432674408
        ],
        [
          "BLUE:MOVE_ROBBER(((0,-2,2),RED))",
          0.3333333432674408
        ],
        [
          "BLUE:MOVE_ROBBER(((1,-2,1),RED))",
          0.3333333432674408
        ],
        [
          "BLUE:MOVE_ROBBER(((-1,1,0),None))",
          0.0
        ],
        [
          "BLUE:MOVE_ROBBER(((-1,2,-1),None))",
          0.0
        ]
      ],
      "value_prediction": 0.9999998211860657
    }
  ]
}
```

## Confirmed Correct and Expected

- Phase file inventory and architecture boundary
- Test suite results
- Self-play sample audit
- Dataset / collate / mask audit
- Tiny overfit and checkpoint audit
- Action-ordering consistency audit
- Raw simulator leakage audit
- Small arena benchmark audit
- Model output quality audit

## Potential Issues or Mismatches

- NeuralMCTS fixed-state audit
  - Enabling priors/value changes root ranking on fixed state: 

## Changes Made During Audit

- Added audit runner `scripts/audit_neural_phase.py` and generated audit reports.
- Fixed product bug in `src/catan_ai/training/self_play.py`: unsupported `teacher_type` is now rejected explicitly (no silent ignore).

## Final Decision

Based on this concrete run, the implementation is **not fully correct** for first-pass expectations; see failing sections above for exact mismatches.
