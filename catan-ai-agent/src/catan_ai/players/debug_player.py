"""Minimal integration bot for verifying this repo can plug into Catanatron.

This is NOT a strategic player — it exists only to prove that:
  1. We can subclass Catanatron's Player from an external repo.
  2. Our decide() method is called by the game loop.
  3. We can read game state and return a legal action.

The bot picks the first action from a *sorted* copy of playable_actions so
that the choice is stable regardless of the order the engine yields actions.
This makes DebugPlayer-vs-DebugPlayer games fully reproducible for a given
seed, which is useful for isolating nondeterminism in action generation.

Once the integration is validated, real strategy players will live alongside
this file in the same package.
"""

from catanatron.models.player import Player

_PREVIEW_COUNT = 3


def _action_sort_key(action):
    """Produce a comparable tuple for any Action.

    Action.value is polymorphic (None, int, tuple, str, …), so we
    normalise it to a string to guarantee a total ordering.
    """
    return (action.action_type.value, str(action.value))


class DebugPlayer(Player):
    """A player that prints a short summary each time it is asked to decide,
    then deterministically returns the first action in sorted order."""

    def __init__(self, color, is_bot=True):
        super().__init__(color, is_bot=is_bot)
        self._calls = 0

    def decide(self, game, playable_actions):
        self._calls += 1

        sorted_actions = sorted(playable_actions, key=_action_sort_key)

        preview = sorted_actions[:_PREVIEW_COUNT]
        preview_strs = [
            f"{a.action_type.value}({a.value})" for a in preview
        ]
        remaining = len(sorted_actions) - len(preview)
        if remaining > 0:
            preview_strs.append(f"… +{remaining} more")

        print(
            f"[DebugPlayer {self.color.value}] "
            f"call #{self._calls}  |  "
            f"{len(playable_actions)} actions  |  "
            f"{', '.join(preview_strs)}"
        )

        return sorted_actions[0]

    def reset_state(self):
        self._calls = 0
