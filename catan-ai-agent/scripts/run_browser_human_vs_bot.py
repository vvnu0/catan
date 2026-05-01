"""Create a browser game for one human against a custom Catan AI bot."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "http://localhost:5001"
DEFAULT_UI_URL = "http://localhost:3000"
DEFAULT_SEED = 9910


def create_human_vs_bot(
    *,
    bot: str = "MAIN_BOT",
    human_color: str = "RED",
    seed: int = DEFAULT_SEED,
    api_url: str = DEFAULT_API_URL,
    ui_url: str = DEFAULT_UI_URL,
    dry_run: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    bot = _normalize_bot(bot)
    human_color = human_color.upper()
    players = ["HUMAN", bot] if human_color == "RED" else [bot, "HUMAN"]
    payload = {
        "players": players,
        "map_template": "BASE",
        "vps_to_win": 10,
        "discard_limit": 7,
        "friendly_robber": False,
        "seed": seed,
    }
    out = Path(output_dir) if output_dir else REPO / "experiments" / "browser_sessions"
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "dry_run" if dry_run else "planned",
        "api_url": api_url,
        "ui_url": ui_url,
        "bot": bot,
        "human_color": human_color,
        "seed": seed,
        "request_payload": payload,
        "manual_ui_flow": [
            "Open http://localhost:3000",
            f"Set one player to Human and the other to {bot}",
            f"Use seed {seed} via this script for reproducibility if launching through API",
        ],
    }
    if dry_run:
        _write_json(out / "human_vs_bot_plan.json", plan)
        return plan

    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/games",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not create browser game via {api_url}. Is the Catanatron web server running?"
        ) from exc

    game_id = body["game_id"]
    result = {
        **plan,
        "status": "created",
        "game_id": game_id,
        "browser_url": f"{ui_url.rstrip('/')}/games/{game_id}",
    }
    _write_json(out / "human_vs_bot_plan.json", result)
    return result


def _normalize_bot(bot: str) -> str:
    key = bot.upper()
    aliases = {
        "MAIN": "MAIN_BOT",
        "MAIN_BOT": "MAIN_BOT",
        "REFERENCE": "REFERENCE_BOT",
        "REFERENCE_BOT": "REFERENCE_BOT",
    }
    if key not in aliases:
        raise ValueError("bot must be MAIN_BOT or REFERENCE_BOT")
    return aliases[key]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a browser human-vs-custom-bot game")
    parser.add_argument("--bot", choices=("MAIN_BOT", "REFERENCE_BOT", "main", "reference"), default="MAIN_BOT")
    parser.add_argument("--human-color", choices=("RED", "BLUE", "red", "blue"), default="RED")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--ui-url", default=DEFAULT_UI_URL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    print(json.dumps(create_human_vs_bot(
        bot=args.bot,
        human_color=args.human_color,
        seed=args.seed,
        api_url=args.api_url,
        ui_url=args.ui_url,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
