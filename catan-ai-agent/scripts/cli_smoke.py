"""Smoke test: print basic environment info and confirm catanatron is usable.

Run:
    python scripts/cli_smoke.py
"""

import platform
import shutil
import sys


def main() -> None:
    print("=== Environment Info ===")
    print(f"Python:   {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Prefix:   {sys.prefix}")
    print()

    # Check catanatron importability
    print("=== Catanatron Import Check ===")
    try:
        import catanatron

        version = getattr(catanatron, "__version__", "(no __version__)")
        print(f"catanatron imported OK  (version: {version})")
        print(f"  location: {catanatron.__file__}")
    except ImportError as exc:
        print(f"FAILED to import catanatron: {exc}")

    print()

    # Check that catanatron-play CLI is on PATH
    print("=== CLI Check ===")
    cli_path = shutil.which("catanatron-play")
    if cli_path:
        print(f"catanatron-play found at: {cli_path}")
    else:
        print("catanatron-play NOT found on PATH")

    print()

    # Quick sanity: can we create a Game?
    print("=== Quick Sanity: Create Game ===")
    try:
        from catanatron import Color, Game, RandomPlayer

        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = Game(players, seed=0)
        print(f"Game created OK  (id={game.id})")
        print(f"  playable_actions count: {len(game.playable_actions)}")
    except Exception as exc:
        print(f"FAILED to create game: {exc}")


if __name__ == "__main__":
    main()
