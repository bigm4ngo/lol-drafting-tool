"""Pure champion-select payload parsing with no GUI dependency.

The parser deliberately separates locked picks from hover intents and records the
local player's role.  The GUI can therefore apply personal pool rules only to
the user's active role while still using allied hovers to improve ban advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from draft_engine import Pick

LCU_ROLE_MAP = {
    "top": "TOP",
    "jungle": "JUNGLE",
    "middle": "MID",
    "mid": "MID",
    "bottom": "ADC",
    "bot": "ADC",
    "utility": "SUPPORT",
    "support": "SUPPORT",
}


@dataclass(frozen=True, slots=True)
class BoardChampion:
    champion_id: int
    role: str | None
    locked: bool
    cell_id: int = -1
    is_local: bool = False


@dataclass(frozen=True, slots=True)
class DraftSnapshot:
    allies: tuple[BoardChampion, ...]
    enemies: tuple[BoardChampion, ...]
    ally_bans: tuple[int, ...]
    enemy_bans: tuple[int, ...]
    phase: str
    active_role: str | None = None
    local_player_cell_id: int = -1
    bans_pending: bool = False

    @property
    def locked_allies(self) -> list[Pick]:
        return [Pick(x.champion_id, x.role) for x in self.allies if x.locked]

    @property
    def locked_enemies(self) -> list[Pick]:
        return [Pick(x.champion_id, x.role) for x in self.enemies if x.locked]

    @property
    def allied_context(self) -> list[dict[str, Any]]:
        """Locked picks and hovers, retaining confidence for ban scoring."""
        return [
            {
                "champion_id": x.champion_id,
                "role": x.role,
                "locked": x.locked,
                "is_local": x.is_local,
            }
            for x in self.allies
        ]

    @property
    def local_champion_id(self) -> int | None:
        for champion in self.allies:
            if champion.is_local:
                return champion.champion_id
        return None

    @property
    def local_locked(self) -> bool:
        return any(champion.is_local and champion.locked for champion in self.allies)

    @property
    def all_bans(self) -> tuple[int, ...]:
        return self.ally_bans + self.enemy_bans

    @property
    def draft_key(self) -> tuple[Any, ...]:
        """State that changes recommendations; excludes the countdown timer."""
        ally_key = tuple(
            sorted(
                (x.champion_id, x.role or "", x.locked, x.is_local, x.cell_id)
                for x in self.allies
            )
        )
        enemy_key = tuple(
            sorted((x.champion_id, x.role or "", x.locked, x.cell_id) for x in self.enemies)
        )
        return (
            ally_key,
            enemy_key,
            tuple(sorted(self.ally_bans)),
            tuple(sorted(self.enemy_bans)),
            self.active_role or "",
            self.local_player_cell_id,
        )

    @property
    def ban_key(self) -> tuple[Any, ...]:
        """State relevant to ban advice, excluding GUI/local-player details.

        Websocket and fallback-poll payloads can disagree briefly about cell
        metadata or the local role while still describing the same champions.
        Ban results should remain usable across those harmless differences.
        """
        # Do not include LCU-assigned role strings here. During Champion Select
        # the websocket payload and the canonical 750-ms poll can briefly disagree
        # about assignedPosition (known vs empty) while describing the exact same
        # champions. v3.0.5 therefore rejected valid completed ban results forever.
        # The ban engine performs whole-draft role inference from the champions, so
        # champion identity + ally hover/lock state is the stable state we need.
        allies = tuple(sorted(
            (x.champion_id, x.locked) for x in self.allies
        ))
        enemies = tuple(sorted(x.champion_id for x in self.enemies))
        return (
            allies,
            enemies,
            tuple(sorted(self.ally_bans)),
            tuple(sorted(self.enemy_bans)),
        )


EMPTY_SNAPSHOT = DraftSnapshot((), (), (), (), "Waiting for Champion Select")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def canonical_lcu_role(value: Any) -> str | None:
    return LCU_ROLE_MAP.get(str(value or "").strip().casefold())


def _iter_actions(raw: Any):
    """Yield only mapping actions from all LCU action payload shapes."""
    if not isinstance(raw, list):
        return
    for group in raw:
        if isinstance(group, Mapping):
            yield group
        elif isinstance(group, list):
            for action in group:
                if isinstance(action, Mapping):
                    yield action


def _parse_team(
    raw: Any,
    *,
    allow_hover: bool,
    local_player_cell_id: int = -1,
    completed_pick_cells: set[int] | None = None,
    active_pick_cells: set[int] | None = None,
    pick_actions_available: bool = False,
) -> tuple[BoardChampion, ...]:
    output: list[BoardChampion] = []
    if not isinstance(raw, list):
        return ()
    for player in raw:
        if not isinstance(player, Mapping):
            continue
        cell_id = _safe_int(player.get("cellId"), -1)
        champion_id = _safe_int(player.get("championId"), 0)
        if pick_actions_available:
            # A positive championId is committed unless this exact cell currently
            # owns an in-progress, incomplete pick action. LCU exposes future pick
            # actions as incomplete too, so only ``isInProgress`` actions may turn
            # a champion into a hover when that field is available.
            locked = champion_id > 0 and (
                cell_id in (completed_pick_cells or set())
                or cell_id not in (active_pick_cells or set())
            )
        else:
            # Some test payloads and older client states omit actions entirely.
            locked = champion_id > 0
        if not locked and allow_hover:
            champion_id = champion_id or _safe_int(player.get("championPickIntent"), 0)
        if champion_id > 0:
            output.append(
                BoardChampion(
                    champion_id=champion_id,
                    role=canonical_lcu_role(player.get("assignedPosition")),
                    locked=locked,
                    cell_id=cell_id,
                    is_local=cell_id == local_player_cell_id,
                )
            )
    return tuple(output)


def parse_lcu_session(
    session: Mapping[str, Any], *, include_hover_intents: bool,
    previous_snapshot: DraftSnapshot | None = None,
) -> DraftSnapshot:
    local_player_cell_id = _safe_int(session.get("localPlayerCellId"), -1)
    completed_pick_cells: set[int] = set()
    incomplete_pick_cells: set[int] = set()
    in_progress_pick_cells: set[int] = set()
    has_progress_field = False
    pick_actions_available = False
    raw_actions = session.get("actions", [])
    for action in _iter_actions(raw_actions):
        if str(action.get("type", "")).casefold() != "pick":
            continue
        pick_actions_available = True
        actor_cell_id = _safe_int(action.get("actorCellId"), -1)
        if "isInProgress" in action:
            has_progress_field = True
        if bool(action.get("completed")) and _safe_int(action.get("championId"), 0) > 0:
            completed_pick_cells.add(actor_cell_id)
        elif not bool(action.get("completed")):
            incomplete_pick_cells.add(actor_cell_id)
            if bool(action.get("isInProgress")):
                in_progress_pick_cells.add(actor_cell_id)

    active_pick_cells = (
        in_progress_pick_cells if has_progress_field else incomplete_pick_cells
    )

    raw_my_team = session.get("myTeam", [])
    allies = _parse_team(
        raw_my_team,
        allow_hover=include_hover_intents,
        local_player_cell_id=local_player_cell_id,
        completed_pick_cells=completed_pick_cells,
        active_pick_cells=active_pick_cells,
        pick_actions_available=pick_actions_available,
    )
    enemies = _parse_team(
        session.get("theirTeam", []),
        allow_hover=False,
        completed_pick_cells=completed_pick_cells,
        active_pick_cells=active_pick_cells,
        pick_actions_available=pick_actions_available,
    )

    # LCU action history is not perfectly stable across every phase/client
    # build. A completed pick action can disappear while ``myTeam`` /
    # ``theirTeam`` still contain the locked champion. Preserve lock state by
    # cell for the lifetime of the Champion Select session so a locked card
    # cannot regress into a hover or reopen its role suggestions. This also
    # handles post-draft champion trades: a previously locked cell remains
    # locked even if its champion ID changes.
    if previous_snapshot is not None and previous_snapshot is not EMPTY_SNAPSHOT:
        def preserve_locks(
            current: tuple[BoardChampion, ...],
            previous: tuple[BoardChampion, ...],
        ) -> tuple[BoardChampion, ...]:
            previous_by_cell = {item.cell_id: item for item in previous if item.cell_id >= 0}
            stabilized: list[BoardChampion] = []
            for item in current:
                before = previous_by_cell.get(item.cell_id)
                if before is not None and before.locked and not item.locked:
                    item = BoardChampion(
                        champion_id=item.champion_id,
                        role=item.role or before.role,
                        locked=True,
                        cell_id=item.cell_id,
                        is_local=item.is_local,
                    )
                elif before is not None and item.role is None and before.role is not None:
                    item = BoardChampion(
                        champion_id=item.champion_id,
                        role=before.role,
                        locked=item.locked,
                        cell_id=item.cell_id,
                        is_local=item.is_local,
                    )
                stabilized.append(item)
            return tuple(stabilized)

        allies = preserve_locks(allies, previous_snapshot.allies)
        enemies = preserve_locks(enemies, previous_snapshot.enemies)

    active_role: str | None = None
    if isinstance(raw_my_team, list):
        for player in raw_my_team:
            if not isinstance(player, Mapping):
                continue
            if _safe_int(player.get("cellId"), -1) == local_player_cell_id:
                active_role = canonical_lcu_role(player.get("assignedPosition"))
                break

    bans_pending = False
    for action in _iter_actions(raw_actions):
        if (
            str(action.get("type", "")).casefold() == "ban"
            and not bool(action.get("completed"))
        ):
            bans_pending = True
            break
    raw_bans = session.get("bans", {})
    bans = raw_bans if isinstance(raw_bans, Mapping) else {}
    ally_bans = tuple(
        _safe_int(x) for x in bans.get("myTeamBans", []) or [] if _safe_int(x) > 0
    )
    enemy_bans = tuple(
        _safe_int(x) for x in bans.get("theirTeamBans", []) or [] if _safe_int(x) > 0
    )
    if not ally_bans and not enemy_bans:
        ally: list[int] = []
        enemy: list[int] = []
        for action in _iter_actions(raw_actions):
            if (
                str(action.get("type", "")).casefold() == "ban"
                and bool(action.get("completed"))
            ):
                champion_id = _safe_int(action.get("championId"), 0)
                if champion_id > 0:
                    (ally if bool(action.get("isAllyAction")) else enemy).append(
                        champion_id
                    )
        ally_bans, enemy_bans = tuple(ally), tuple(enemy)

    raw_timer = session.get("timer", {})
    timer = raw_timer if isinstance(raw_timer, Mapping) else {}
    phase = str(timer.get("phase", "Champion Select")).replace("_", " ").title()
    remaining = int(timer.get("adjustedTimeLeftInPhase", 0) or 0)
    if remaining > 0:
        phase += f" · {remaining / 1000:.1f}s"
    return DraftSnapshot(
        allies=allies,
        enemies=enemies,
        ally_bans=ally_bans,
        enemy_bans=enemy_bans,
        phase=phase,
        active_role=active_role,
        local_player_cell_id=local_player_cell_id,
        bans_pending=bans_pending,
    )
