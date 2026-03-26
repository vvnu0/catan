"""Run matches comparing BeliefMCTSPlayer against all baselines.

Pairings:
  1. BeliefMCTSPlayer vs DebugPlayer
  2. BeliefMCTSPlayer vs RandomPlayer
  3. BeliefMCTSPlayer vs HeuristicBot
  4. BeliefMCTSPlayer vs single-world MCTSPlayer

Run:
    python scripts/run_belief_mcts_match.py
"""

from __future__ import annotations

import io
import contextlib
import statistics
import time

from catanatron import Color, Game, RandomPlayer

from catan_ai.players import DebugPlayer, HeuristicBot, MCTSPlayer
from catan_ai.players.belief_mcts_player import BeliefMCTSConfig, BeliefMCTSPlayer

GAMES_PER_PAIR = 10
BASE_SEED = 2000

BELIEF_CFG = BeliefMCTSConfig(
    num_worlds=6,
    sims_per_world=10,
    belief_seed=0,
    belief_mode="conservation",
    enable_devcard_sampling=False,
    max_depth=10,
    exploration_c=1.41,
    top_k_roads=3,
    top_k_trades=2,
    top_k_robber=4,
)


def _run_batch(make_red, make_blue, label: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {label}  ({GAMES_PER_PAIR} games)")
    print(f"{'=' * 68}")

    wins = losses = draws = 0
    turn_list: list[int] = []
    ms_list: list[float] = []

    for i in range(GAMES_PER_PAIR):
        red = make_red()
        blue = make_blue()
        game = Game([red, blue], seed=BASE_SEED + i)

        with contextlib.redirect_stdout(io.StringIO()):
            winner = game.play()

        if winner == Color.RED:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

        turn_list.append(game.state.num_turns)
        if isinstance(red, (BeliefMCTSPlayer, MCTSPlayer)):
            ms_list.append(red.avg_move_ms)

    avg_turns = statistics.mean(turn_list)
    avg_ms = statistics.mean(ms_list) if ms_list else 0.0

    print(f"  RED wins:     {wins}")
    print(f"  RED losses:   {losses}")
    print(f"  Draws:        {draws}")
    print(f"  Avg turns:    {avg_turns:.1f}")
    if ms_list:
        print(f"  Avg move ms:  {avg_ms:.1f}")


def main() -> None:
    print(
        f"BeliefMCTS config: {BELIEF_CFG.num_worlds} worlds × "
        f"{BELIEF_CFG.sims_per_world} sims, "
        f"depth={BELIEF_CFG.max_depth}, "
        f"devcard_sampling={BELIEF_CFG.enable_devcard_sampling}"
    )

    mk_belief = lambda: BeliefMCTSPlayer(Color.RED, config=BELIEF_CFG)

    _run_batch(
        mk_belief,
        lambda: DebugPlayer(Color.BLUE),
        "BeliefMCTSPlayer vs DebugPlayer",
    )

    _run_batch(
        mk_belief,
        lambda: RandomPlayer(Color.BLUE),
        "BeliefMCTSPlayer vs RandomPlayer",
    )

    _run_batch(
        mk_belief,
        lambda: HeuristicBot(Color.BLUE),
        "BeliefMCTSPlayer vs HeuristicBot",
    )

    _run_batch(
        mk_belief,
        lambda: MCTSPlayer(Color.BLUE, max_simulations=60, max_depth=10, seed=0),
        "BeliefMCTSPlayer vs MCTSPlayer(60 sims)",
    )


if __name__ == "__main__":
    main()
