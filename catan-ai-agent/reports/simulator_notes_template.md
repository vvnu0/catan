# Catanatron Simulator Notes

Use `scripts/state_probe.py` to inspect raw game state and fill in observations below.

---

## Public Information

_Information visible to all players._

- Board layout (tiles, numbers, ports)
- Robber location
- Building placements (settlements, cities) and their owners
- Road placements and their owners
- Longest Road holder and length
- Largest Army holder
- Visible victory points (settlements + cities + longest road + largest army)
- Number of resource cards each player holds (count only, not which resources)
- Number of development cards each player holds (count only, not which)
- Number of played knights per player
- Dice rolls
- Trade offers and outcomes
- Bank resource supply counts

---

## Hidden Information

_Information private to one player or unknown to all._

- Exact resources in each player's hand (private to that player)
- Development cards in hand (private to that player)
- Remaining development card deck composition
- Actual victory points (hidden VP dev cards are secret until game end)
- `*_OWNED_AT_START` flags — whether a dev card was owned at the start of this turn (affects playability)

---

## Action Categories

_Types of actions a player can take (from `ActionType` enum)._

- **Dice**: `ROLL`
- **Building**: `BUILD_ROAD`, `BUILD_SETTLEMENT`, `BUILD_CITY`
- **Buying**: `BUY_DEVELOPMENT_CARD`
- **Dev card play**: `PLAY_KNIGHT_CARD`, `PLAY_YEAR_OF_PLENTY`, `PLAY_MONOPOLY`, `PLAY_ROAD_BUILDING`
- **Robber**: `MOVE_ROBBER`, `DISCARD`
- **Trading**: `MARITIME_TRADE`, `OFFER_TRADE`, `ACCEPT_TRADE`, `REJECT_TRADE`, `CONFIRM_TRADE`, `CANCEL_TRADE`
- **Turn management**: `END_TURN`

---

## Things That Could Leak Hidden Information

_Observations about information that might be unintentionally exposed._

- `state.player_state` contains *all* players' hands in a flat dict — an agent with access to the state object can read opponents' resources and dev cards directly.
- `state.development_listdeck` reveals the full ordering of the remaining dev card deck.
- `state.resource_freqdeck` (bank supply) combined with visible building yields can let an agent infer what opponents received from rolls.
- `ACTUAL_VICTORY_POINTS` vs `VICTORY_POINTS` — the difference reveals hidden VP dev cards; an omniscient agent could read this for any player.
- `buildings_by_color` and `board.buildings` are public anyway, but code that reads them may accidentally also read adjacent private fields.
- Action records (`state.action_records`) log every action including dev card plays, which is public, but also trade contents which reveal resource information.

---

_Fill in additional observations as you experiment with `state_probe.py` and `sim_probe.py`._
