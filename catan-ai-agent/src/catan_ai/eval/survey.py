"""Survey schema helpers for human evaluation sessions."""

from __future__ import annotations

REQUIRED_RESULT_FIELDS = {
    "game_id",
    "bot_faced",
    "bot_role",
    "seed",
    "winner",
    "bot_final_vp",
    "human_final_vp",
    "turns",
}

REQUIRED_SURVEY_FIELDS = {
    "participant_id",
    "skill_group",
    "game_id",
    "bot_faced",
    "seed",
    "winner",
    "final_vp",
    "turns",
    "moves_felt_sensible",
    "gameplay_felt_fair",
    "challenge_level",
}

RATING_FIELDS = (
    "moves_felt_sensible",
    "gameplay_felt_fair",
    "challenge_level",
)

DEFAULT_SURVEY_FIELDS = (
    "participant_id",
    "skill_group",
    "game_id",
    "bot_faced",
    "seed",
    "winner",
    "final_vp",
    "turns",
    "moves_felt_sensible",
    "gameplay_felt_fair",
    "challenge_level",
    "comments",
)


def validate_result(record: dict) -> list[str]:
    """Return missing/invalid field messages for a completed game result."""
    errors = _missing_errors(record, REQUIRED_RESULT_FIELDS)
    if record.get("winner") not in {"human", "bot", "draw"}:
        errors.append("winner must be one of: human, bot, draw")
    return errors


def validate_survey(record: dict) -> list[str]:
    """Return missing/invalid field messages for a survey response."""
    errors = _missing_errors(record, REQUIRED_SURVEY_FIELDS)
    for field in RATING_FIELDS:
        try:
            value = int(record.get(field))
        except (TypeError, ValueError):
            errors.append(f"{field} must be an integer 1-5")
            continue
        if value < 1 or value > 5:
            errors.append(f"{field} must be an integer 1-5")
    return errors


def _missing_errors(record: dict, fields: set[str]) -> list[str]:
    return [f"missing required field: {field}" for field in sorted(fields) if field not in record]
