"""Self-play data generation and policy/value training pipeline.

AlphaZero-lite loop:
  1. Run self-play games using a search teacher (MCTSPlayer or BeliefMCTSPlayer).
  2. At each decision point record state features, per-action features,
     the MCTS visit distribution as a policy target, and the final game
     outcome as a value target.
  3. Train a small PolicyValueNet on these samples.
  4. Load the trained model into NeuralMCTSPlayer to guide future search.
"""
