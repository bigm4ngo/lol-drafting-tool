"""Read-only adapter over the collector's draft_data.sqlite3, plus frozen snapshots."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .catalog import ROLES, Catalog
from .rules import SKIPPED

ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}


@dataclass
class Matches:
    """Role-ordered lineups: blue[i, r] is the engine idx of blue's champion in role r."""
    match_id: np.ndarray  # str
    creation: np.ndarray  # int64 ms
    patch: np.ndarray  # str "16.23"
    blue: np.ndarray  # (N,5) int
    red: np.ndarray  # (N,5) int
    blue_win: np.ndarray  # (N,) int8
    bans: np.ndarray  # (N,2,5) int, SKIPPED where no ban
    champ_ids: np.ndarray  # engine idx -> Riot id at snapshot time

    def __len__(self) -> int:
        return len(self.blue)

    def subset(self, mask: np.ndarray) -> "Matches":
        return Matches(self.match_id[mask], self.creation[mask], self.patch[mask], self.blue[mask],
                       self.red[mask], self.blue_win[mask], self.bans[mask], self.champ_ids)

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})
        digest = hashlib.sha1(self.match_id.astype(str).tobytes()).hexdigest()[:12]
        return digest

    @classmethod
    def load(cls, path: Path) -> "Matches":
        with np.load(path, allow_pickle=False) as data:
            return cls(**{k: data[k] for k in cls.__dataclass_fields__})


def load_matches(db_path: Path, catalog: Catalog, *, queue_ids=(420,), since_ms: int | None = None,
                 until_ms: int | None = None) -> Matches:
    """Return complete, role-valid matches in chronological order. Never writes to the DB."""
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        where, args = [f"m.queue_id IN ({','.join('?' * len(queue_ids))})", "m.winning_team_id IN (100,200)"], list(queue_ids)
        if since_ms is not None:
            where.append("m.game_creation >= ?"); args.append(since_ms)
        if until_ms is not None:
            where.append("m.game_creation < ?"); args.append(until_ms)
        rows = conn.execute(
            "SELECT m.match_id, m.game_creation, m.game_version, m.winning_team_id, p.team_id, p.champion_id, p.role "
            f"FROM matches m JOIN participants p USING(match_id) WHERE {' AND '.join(where)} "
            "ORDER BY m.game_creation, m.match_id", args).fetchall()
        ban_rows = conn.execute("SELECT match_id, team_id, champion_id, pick_turn FROM bans ORDER BY match_id, team_id, pick_turn").fetchall()
    finally:
        conn.close()

    bans_by_match: dict[str, list[list[int]]] = {}
    for mid, team, cid, _turn in ban_rows:
        side = 0 if team == 100 else 1
        champ = catalog.by_id.get(int(cid))
        bans_by_match.setdefault(mid, [[], []])[side].append(champ.idx if champ else SKIPPED)

    out: dict[str, list] = {k: [] for k in ("match_id", "creation", "patch", "blue", "red", "blue_win", "bans")}

    def flush(mid, creation, version, winner, lineup) -> None:
        if any(slot is None for team in lineup for slot in team):
            return  # incomplete or duplicated roles, or champion unknown to this catalog
        out["match_id"].append(mid); out["creation"].append(creation)
        out["patch"].append(".".join(str(version).split(".")[:2]))
        out["blue"].append(lineup[0]); out["red"].append(lineup[1]); out["blue_win"].append(int(winner == 100))
        b = bans_by_match.get(mid, [[], []])
        out["bans"].append([(side + [SKIPPED] * 5)[:5] for side in b])

    current, meta, lineup, bad = None, None, None, False
    for mid, creation, version, winner, team, cid, role in rows:
        if mid != current:
            if current is not None and not bad:
                flush(current, *meta, lineup)
            current, meta, lineup, bad = mid, (creation, version, winner), [[None] * 5, [None] * 5], False
        side, r, champ = (0 if team == 100 else 1), ROLE_INDEX.get(role), catalog.by_id.get(int(cid))
        if r is None or champ is None or lineup[side][r] is not None:
            bad = True
            continue
        lineup[side][r] = champ.idx
    if current is not None and not bad:
        flush(current, *meta, lineup)

    n = len(out["blue"])
    return Matches(
        match_id=np.array(out["match_id"], dtype=str), creation=np.array(out["creation"], dtype=np.int64),
        patch=np.array(out["patch"], dtype=str), blue=np.array(out["blue"], dtype=np.int64).reshape(n, 5),
        red=np.array(out["red"], dtype=np.int64).reshape(n, 5), blue_win=np.array(out["blue_win"], dtype=np.int8),
        bans=np.array(out["bans"], dtype=np.int64).reshape(n, 2, 5),
        champ_ids=np.array([c.id for c in catalog.champions], dtype=np.int64),
    )


def time_split(m: Matches, fractions=(0.8, 0.1, 0.1)) -> tuple[Matches, Matches, Matches]:
    """Chronological train / calibration / test split by match (never by masked slot)."""
    n = len(m)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    idx = np.arange(n)
    return m.subset(idx < a), m.subset((idx >= a) & (idx < b)), m.subset(idx >= b)
