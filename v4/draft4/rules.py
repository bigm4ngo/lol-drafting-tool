"""Draft formats, immutable draft state and deterministic legality."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

import numpy as np

BLUE, RED = 0, 1
SIDES = ("blue", "red")
EMPTY, SKIPPED = -1, -2  # SKIPPED = a ban slot deliberately left empty


@dataclass(frozen=True, slots=True)
class Step:
    side: int
    kind: str  # "ban" | "pick"
    slot: int


def _seq(spec: str, kind: str, start: dict[int, int]) -> list[Step]:
    out = []
    for ch in spec:
        side = BLUE if ch == "B" else RED
        out.append(Step(side, kind, start[side]))
        start[side] += 1
    return out


def _pro() -> list[Step]:
    bans, picks = {0: 0, 1: 0}, {0: 0, 1: 0}
    return (_seq("BRBRBR", "ban", bans) + _seq("BRRBBR", "pick", picks)
            + _seq("RBRB", "ban", bans) + _seq("RBBR", "pick", picks))


def _ranked() -> list[Step]:
    # Ranked bans are simultaneous; the order here is a simulation convenience only.
    return _seq("BRBRBRBRBR", "ban", {0: 0, 1: 0}) + _seq("BRRBBRRBBR", "pick", {0: 0, 1: 0})


FORMATS: dict[str, dict] = {
    "pro": {"label": "Pro (tournament draft)", "steps": _pro(), "fearless": False, "shared_bans": False},
    "fearless": {"label": "Pro — Fearless series", "steps": _pro(), "fearless": True, "shared_bans": False},
    "ranked": {"label": "Ranked solo/duo", "steps": _ranked(), "fearless": False, "shared_bans": True},
}
for _f in FORMATS.values():  # same order with the first-pick team on red
    _f["steps_red_first"] = [Step(1 - st.side, st.kind, st.slot) for st in _f["steps"]]


@dataclass(frozen=True, slots=True)
class DraftState:
    fmt: str = "pro"
    bans: tuple[tuple[int, ...], tuple[int, ...]] = ((EMPTY,) * 5, (EMPTY,) * 5)
    picks: tuple[tuple[int, ...], tuple[int, ...]] = ((EMPTY,) * 5, (EMPTY,) * 5)
    roles: tuple[tuple[int, ...], tuple[int, ...]] = ((EMPTY,) * 5, (EMPTY,) * 5)  # locked role per pick slot
    unavailable: frozenset[int] = field(default_factory=frozenset)  # fearless series locks
    first: int = BLUE  # side holding first pick; since 2026 pro teams may take first pick on red

    def __post_init__(self) -> None:
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown format {self.fmt!r}")
        if self.fmt == "ranked" and self.first != BLUE:
            raise ValueError("ranked always gives blue first pick")

    @property
    def steps(self) -> list[Step]:
        return FORMATS[self.fmt]["steps" if self.first == BLUE else "steps_red_first"]

    def slot_value(self, step: Step) -> int:
        return (self.bans if step.kind == "ban" else self.picks)[step.side][step.slot]

    def next_index(self) -> int | None:
        for i, step in enumerate(self.steps):
            if self.slot_value(step) == EMPTY:
                return i
        return None

    def next_step(self) -> Step | None:
        i = self.next_index()
        return None if i is None else self.steps[i]

    def remaining_steps(self) -> list[Step]:
        return [s for s in self.steps if self.slot_value(s) == EMPTY]

    def team(self, side: int) -> list[int]:
        return [c for c in self.picks[side] if c >= 0]

    def all_bans(self) -> list[int]:
        return [c for side in self.bans for c in side if c >= 0]

    def taken(self) -> set[int]:
        return set(self.team(BLUE)) | set(self.team(RED)) | set(self.all_bans()) | set(self.unavailable)

    def is_complete(self) -> bool:
        return self.next_index() is None

    def legal_mask(self, n_champions: int) -> np.ndarray:
        mask = np.ones(n_champions, dtype=bool)
        taken = list(self.taken())
        if taken:
            mask[taken] = False
        return mask

    def is_legal(self, step: Step, champ: int) -> bool:
        if champ == SKIPPED:
            return step.kind == "ban"
        if champ in set(self.team(BLUE)) | set(self.team(RED)) | self.unavailable:
            return False
        if champ in self.all_bans():
            # ranked bans are simultaneous, so both teams may ban the same champion
            return step.kind == "ban" and FORMATS[self.fmt]["shared_bans"] and champ not in self.bans[step.side]
        return True

    def apply(self, step: Step, champ: int, role: int = EMPTY, *, check: bool = True) -> "DraftState":
        if check and not self.is_legal(step, champ):
            raise ValueError(f"illegal {step.kind} of champion {champ} for {SIDES[step.side]}")
        if step.kind == "ban":
            side_bans = list(self.bans[step.side])
            side_bans[step.slot] = champ
            bans = tuple(tuple(side_bans) if s == step.side else self.bans[s] for s in (0, 1))
            return replace(self, bans=bans)
        side_picks, side_roles = list(self.picks[step.side]), list(self.roles[step.side])
        side_picks[step.slot], side_roles[step.slot] = champ, role
        return replace(
            self,
            picks=tuple(tuple(side_picks) if s == step.side else self.picks[s] for s in (0, 1)),
            roles=tuple(tuple(side_roles) if s == step.side else self.roles[s] for s in (0, 1)),
        )

    def fingerprint(self) -> str:
        raw = repr((self.fmt, self.first, self.bans, self.picks, self.roles, sorted(self.unavailable)))
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    # JSON boundary (champion Riot ids <-> engine idx handled by the caller's maps)
    def to_json(self, idx_to_id: list[int]) -> dict:
        conv = lambda c: c if c < 0 else idx_to_id[c]
        return {
            "fmt": self.fmt,
            "first": SIDES[self.first],
            "bans": [[conv(c) for c in side] for side in self.bans],
            "picks": [[conv(c) for c in side] for side in self.picks],
            "roles": [list(side) for side in self.roles],
            "unavailable": sorted(idx_to_id[c] for c in self.unavailable),
        }

    @classmethod
    def from_json(cls, data: dict, id_to_idx: dict[int, int]) -> "DraftState":
        def conv(c) -> int:
            c = int(c)
            if c < 0:
                return EMPTY if c != SKIPPED else SKIPPED
            if c not in id_to_idx:
                raise ValueError(f"unknown champion id {c}")
            return id_to_idx[c]

        def five(rows, f) -> tuple[tuple[int, ...], tuple[int, ...]]:
            rows = rows or [[EMPTY] * 5, [EMPTY] * 5]
            if len(rows) != 2 or any(len(r) != 5 for r in rows):
                raise ValueError("expected two rows of five slots")
            return tuple(tuple(f(c) for c in r) for r in rows)  # type: ignore[return-value]

        roles = five(data.get("roles"), lambda r: int(r) if 0 <= int(r) < 5 else EMPTY)
        state = cls(
            fmt=str(data.get("fmt", "pro")),
            first=SIDES.index(data.get("first", "blue")) if data.get("first", "blue") in SIDES else BLUE,
            bans=five(data.get("bans"), conv),
            picks=five(data.get("picks"), lambda c: EMPTY if int(c) == SKIPPED else conv(c)),
            roles=roles,
            unavailable=frozenset(conv(c) for c in data.get("unavailable", [])),
        )
        picked = state.team(BLUE) + state.team(RED)
        if len(picked) != len(set(picked)):
            raise ValueError("a champion is picked twice")
        if set(picked) & state.unavailable:
            raise ValueError("a fearless-locked champion is picked")
        return state
