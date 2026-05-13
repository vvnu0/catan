# Neural MCTS Model Architecture

This document describes the specific neural network architecture implemented for the Neural Monte Carlo Tree Search (MCTS) agent in this project, specifically the model defined in `src/catan_ai/models/policy_value_net.py`.

## Overview

The model uses a **lightweight, dynamic two-headed Multi-Layer Perceptron (MLP)** architecture. 

It loosely follows the AlphaZero paradigm by having separate Policy and Value heads, but it makes a major structural adaptation to handle the massive and highly variable action space of Catan. 

### The Core Innovation: Dynamic Action Scoring
In traditional AlphaZero implementations (like Chess or Go), the Policy Network outputs a fixed-size vector containing a probability for every conceivable move in the game. In Catan, the number of possible actions—especially considering complex trades—is enormous and highly situational. 

To solve this, **this model avoids a giant fixed action vocabulary**. Instead of outputting a massive list of probabilities, the model takes *features of the current legal actions* as input and scores them dynamically.

## The Four Components

### 1. State Encoder (MLP)
- **Input:** A flattened vector of raw game state features (`state_feats`). These features explicitly contain raw information like "who is winning", "where the robber is", "how many resource cards each player has", etc.
- **Structure:** 2 Linear layers with ReLU activations and Dropout.
- **Output:** A dense representation (`state_embed` of size `hidden_dim`). The model implicitly learns how to weigh the raw facts to understand the semantic meaning of the board state.

### 2. Action Encoder (MLP)
- **Input:** A batch of features for each currently available action (`action_feats`). These explicitly contain the semantic impacts of each move, such as "settling here yields a 6, 5, and 2".
- **Structure:** 2 Linear layers with a ReLU activation in between.
- **Output:** A dense representation of each individual action (`action_embed` of size `hidden_dim`).

### 3. Value Head
- **Purpose:** Evaluates how good the current board position is, completely independently of the specific available actions.
- **Input:** The `state_embed` from the State Encoder.
- **Structure:** 2 Linear layers ending with a **Tanh** activation function.
- **Output:** A single scalar representing the expected game outcome, squeezed into the range `[-1, +1]` (where -1 is a loss and +1 is a win).

### 4. Policy Head
- **Purpose:** Scores each specific valid move so that the MCTS knows which branches to prioritize.
- **Input:** For each available action, the `state_embed` is duplicated to match the number of actions, and then **concatenated** with the specific `action_embed`. The resulting input size is exactly `hidden_dim * 2` (e.g., if `hidden_dim` is 64, the input is 128).
- **Structure:** 2-layer MLP.
- **Output:** A single scalar "logit" (score) for that specific action. 

---

## Handling Variable Input Sizes & Batching

Because the number of legal moves in Catan varies wildly from turn to turn, the network handles data using a combination of **Batching, Padding, and Masking**.

The tensor shapes look like `[B, A_max, ...]`.

### The Batch Dimension (`B`)
The `B` dimension represents multiple independent board states being evaluated at the exact same time. 
- **During Training:** `B` might be 128 or 256, representing a batch of completely different board states from past games. The network evaluates them all simultaneously to update its weights efficiently via gradient descent.
- **During Batched MCTS (Inference):** The MCTS might gather multiple leaf nodes (e.g., 8 different scenarios) from different parts of the search tree and evaluate them all at once (`B = 8`) to maximize GPU efficiency. If evaluating sequentially, `B = 1`.

### Padding to `A_max`
The `A_max` dimension represents the maximum number of valid moves available for any single board state in the current batch. 
- If one state has 10 valid moves and another has 50, both are padded with "fake" zeroes up to `A_max = 50`.
- The PyTorch `nn.Linear` layers in the Policy Head broadcast across the `B` and `A_max` dimensions. This means the Policy Head evaluates all 50 slots (both real and garbage slots) completely independently and simultaneously using the exact same weights.

### Masking Invalid Actions
Because the network calculates scores for the "fake" padded actions, those specific output scores are garbage. 
To prevent the agent from picking fake moves, the model uses an `action_mask` (a boolean array denoting which slots were real and which were padding).

Before returning the final logits, the model does the following:
```python
logits = logits.masked_fill(~action_mask, float("-inf"))
```
This forces the scores of the fake/padded moves to **Negative Infinity** (`-inf`). Later, when these logits are passed through a Softmax function in the MCTS to get probabilities, the `-inf` values evaluate to exactly **0**, completely erasing the invalid moves from consideration.
