"""Run debug matches to validate integration and reproducibility.

Match 1 — DebugPlayer vs DebugPlayer (run twice with the same seed).
  Both players are fully deterministic (sorted action selection), so
  if Catanatron's action generation is stable the two runs must produce
  identical turn counts.  Any difference means the engine itself has
  ordering nondeterminism.

Match 2 — DebugPlayer vs RandomPlayer (smoke test).
  RandomPlayer injects randomness, so this match is NOT expected to be
  reproducible across processes.  It is kept only as a quick integration
  check.

Run:
    python scripts/run_debug_match.py
"""

from catanatron import Color, Game, RandomPlayer

from catan_ai.players import DebugPlayer

SEED = 42
VPS_TO_WIN = 10


def run_game(players, seed, label):
    game = Game(players, seed=seed, vps_to_win=VPS_TO_WIN)
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  seed={seed}  vps_to_win={VPS_TO_WIN}")
    for p in players:
        print(f"    {p.color.value:>6} = {type(p).__name__}")
    print(f"{'=' * 60}")

    winner = game.play()

    winner_str = winner.value if winner else "none (turn limit)"
    print(f"  Winner: {winner_str}")
    print(f"  Turns:  {game.state.num_turns}")
    for p in players:
        if hasattr(p, "_calls"):
            print(f"  {type(p).__name__}({p.color.value}) decide calls: {p._calls}")

    return game.state.num_turns


def main() -> None:
    # -- Match 1a & 1b: deterministic reproducibility check --
    turns_a = run_game(
        [DebugPlayer(Color.RED), DebugPlayer(Color.BLUE)],
        seed=SEED,
        label="Match 1a: DebugPlayer vs DebugPlayer",
    )
    turns_b = run_game(
        [DebugPlayer(Color.RED), DebugPlayer(Color.BLUE)],
        seed=SEED,
        label="Match 1b: DebugPlayer vs DebugPlayer (same seed, second run)",
    )

    print(f"\nReproducibility check: run-a={turns_a} turns, run-b={turns_b} turns", end="  ")
    if turns_a == turns_b:
        print("=> IDENTICAL (deterministic)")
    else:
        print("=> DIFFER (nondeterminism in engine action ordering)")

    # -- Match 2: smoke test against RandomPlayer --
    run_game(
        [DebugPlayer(Color.RED), RandomPlayer(Color.BLUE)],
        seed=SEED,
        label="Match 2: DebugPlayer vs RandomPlayer (smoke test)",
    )


if __name__ == "__main__":
    main()
