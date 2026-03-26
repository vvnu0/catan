"""Encode Catanatron Action namedtuples into stable, serialisable keys.

The codec converts each ``Action(color, action_type, value)`` into an
``EncodedAction`` dataclass that contains only strings/primitives.  Encoded
actions are deterministically sortable, hashable, and safe to log or use as
policy-target keys without exposing raw Catanatron objects downstream.
"""

from __future__ import annotations

from catanatron.models.enums import Action

from catan_ai.adapters.public_state import EncodedAction


class ActionCodec:
    """Stateless encoder that converts Catanatron Actions ↔ EncodedActions."""

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    @staticmethod
    def encode(action: Action) -> EncodedAction:
        """Turn a raw Catanatron Action into an EncodedAction."""
        return EncodedAction(
            color=action.color.value,
            action_type=action.action_type.value,
            value=ActionCodec._normalise_value(action.value),
        )

    @staticmethod
    def encode_many(actions) -> list[EncodedAction]:
        """Encode a sequence of raw Actions."""
        return [ActionCodec.encode(a) for a in actions]

    # ------------------------------------------------------------------
    # Deterministic sort
    # ------------------------------------------------------------------
    @staticmethod
    def sort_key(ea: EncodedAction) -> tuple:
        """Return a tuple suitable for deterministic sorting."""
        return (ea.color, ea.action_type, ea.value or "")

    @staticmethod
    def sorted_actions(encoded: list[EncodedAction]) -> list[EncodedAction]:
        """Return a new list sorted deterministically."""
        return sorted(encoded, key=ActionCodec.sort_key)

    # ------------------------------------------------------------------
    # Decoding (best-effort; useful for logging / debugging)
    # ------------------------------------------------------------------
    @staticmethod
    def decode_to_str(ea: EncodedAction) -> str:
        """Human-readable string for logging."""
        if ea.value is None:
            return f"{ea.color}:{ea.action_type}"
        return f"{ea.color}:{ea.action_type}({ea.value})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_value(value) -> str | None:
        """Convert the polymorphic Action.value into a stable string.

        We stringify rather than trying to preserve the original Python types
        so that the key is always a plain string — safe for hashing, JSON
        serialisation, and cross-session determinism.
        """
        if value is None:
            return None
        if isinstance(value, tuple):
            return ActionCodec._tuple_to_str(value)
        return str(value)

    @staticmethod
    def _tuple_to_str(t: tuple) -> str:
        """Recursively render a (possibly nested) tuple as a stable string."""
        parts: list[str] = []
        for item in t:
            if isinstance(item, tuple):
                parts.append(ActionCodec._tuple_to_str(item))
            elif item is None:
                parts.append("None")
            elif hasattr(item, "value"):
                # Enum-like (Color, etc.)
                parts.append(str(item.value))
            else:
                parts.append(str(item))
        return "(" + ",".join(parts) + ")"
