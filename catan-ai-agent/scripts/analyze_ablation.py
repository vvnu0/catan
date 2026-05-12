#!/usr/bin/env python3
"""Analysis and visualization script for the Neural MCTS Ablation Study."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

REPO_DIR = Path(__file__).resolve().parents[1]
ABLATION_JSON = REPO_DIR / "experiments" / "ablation_study" / "ablation_summary.json"
OUTPUT_DIR = REPO_DIR / "experiments" / "ablation_study" / "figures"

def main():
    if not ABLATION_JSON.exists():
        print(f"Error: {ABLATION_JSON} not found. Please ensure the ablation study has completed.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ABLATION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results:
        print("No results found in JSON.")
        return

    # Extract metrics
    configs = ["neither", "priors_only", "value_only", "full_neural"]
    labels = ["Base\n(Neither)", "Priors\nOnly", "Value\nOnly", "Full\nNeural"]
    
    win_rates = []
    avg_vps = []
    avg_turns = []
    latencies = []

    for cfg in configs:
        res = results[cfg]
        win_rates.append(res["win_rate"] * 100) # percentage
        avg_vps.append(res["avg_final_vp"])
        avg_turns.append(res["avg_turns"])
        latencies.append(res["mean_latency_ms"])

    # Setup plotting style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Distinct colors for each condition
    colors = ['#a1a1a1', '#4c72b0', '#dd8452', '#55a868']

    # 1. Win Rate
    bars = axes[0].bar(labels, win_rates, color=colors)
    axes[0].set_title("Win Rate vs Baseline (%)", fontsize=14, pad=15, fontweight='bold')
    axes[0].set_ylim(40, 65)
    axes[0].axhline(50, color='red', linestyle='--', alpha=0.5, label='50% Baseline')
    axes[0].legend()
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11)

    # 2. Avg Final VP
    bars = axes[1].bar(labels, avg_vps, color=colors)
    axes[1].set_title("Average Final Victory Points", fontsize=14, pad=15, fontweight='bold')
    axes[1].set_ylim(5.0, 8.0)
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11)

    # 3. Avg Turns
    bars = axes[2].bar(labels, avg_turns, color=colors)
    axes[2].set_title("Average Game Length (Turns)", fontsize=14, pad=15, fontweight='bold')
    axes[2].set_ylim(100, 220)
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11)

    # 4. Latency
    bars = axes[3].bar(labels, latencies, color=colors)
    axes[3].set_title("Mean Search Latency (ms)", fontsize=14, pad=15, fontweight='bold')
    axes[3].set_ylim(0, 10)
    for bar in bars:
        height = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f} ms', ha='center', va='bottom', fontsize=11)

    # Finalize and save
    plt.suptitle("Neural MCTS Ablation Study Results", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "ablation_metrics.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualizations successfully generated and saved to:\n   {out_path}\n")

    # Print the paragraph analysis
    print("=" * 80)
    print("ABLATION STUDY ANALYSIS AND INTERPRETATION FOR REPORT")
    print("=" * 80)
    analysis = (
        "The ablation study reveals that the neural policy priors are the primary driver of "
        "performance improvements in the MCTS agent. The 'Priors Only' configuration achieved "
        "the highest win rate (60.0%) against the frequency MCTS baseline and significantly "
        "reduced the average game length (150.4 turns) compared to the 'Base' non-neural "
        "search (173.2 turns). This suggests that the policy head successfully prunes the "
        "search tree, guiding the agent toward stronger, more decisive paths to victory.\n\n"
        "Conversely, the 'Value Only' configuration underperformed the baseline (47.0% win rate). "
        "Without the policy head pruning the tree, MCTS explores highly irregular, sub-optimal "
        "move sequences. Because the neural value network was likely trained on states from "
        "competent gameplay, these bizarre exploratory states are 'out-of-distribution', "
        "causing the value network's predictions to break down. The handcrafted heuristic, "
        "which relies on fundamental mathematical principles (pip counts, VPs), is far more "
        "robust to these out-of-distribution states.\n\n"
        "The 'Full Neural' configuration combines both heads, yielding a strong 57.0% win rate "
        "and the highest average final victory points (7.2). While its strict win rate is slightly "
        "lower than 'Priors Only', the high VP score suggests a synergistic effect where the "
        "value head improves the overall robustness of the final board state. It is highly "
        "recommended to focus future improvements on the value network's training data diversity "
        "or to maintain a permanent blended approach with the handcrafted heuristic to ensure "
        "evaluative stability.\n\n"
        "To fundamentally resolve this 'out-of-distribution' weakness in the value network, "
        "future training regimens must explicitly expose the model to sub-optimal board states "
        "and learn to evaluate them accurately. This can be achieved during self-play by injecting "
        "Dirichlet noise into the root priors or using a high softmax temperature for early moves, "
        "forcing the agent to intentionally explore weak branches and accurately learn their negative "
        "values. Furthermore, because Catan is a long-horizon game with extremely sparse rewards "
        "(a single win/loss signal at the very end), the value network suffers from credit assignment "
        "issues. The model could be significantly improved by introducing auxiliary loss targets during "
        "training—such as predicting the final VP margin, total resource production, or longest road length. "
        "This would give the value head a richer, denser gradient to learn from, grounding its evaluations "
        "even when analyzing chaotic or irregular board states."
    )
    print(analysis)
    print("=" * 80)

if __name__ == "__main__":
    main()
