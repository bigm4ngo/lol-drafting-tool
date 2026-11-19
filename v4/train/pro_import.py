"""Oracle's Elixir CSVs -> ordered pro drafts (read-only on the CSVs).

    python -m train.pro_import --years 2024 2025 2026

OE gives, per team row, ban1..ban5 and pick1..pick5 in the order that team made them, plus
firstPick and side. Interleaved with the tournament sequence, that is the full ordered draft.
Games whose draft can't be rebuilt exactly (missing picks, pick list != played champions,
unknown champion names, unknown first pick) are dropped and counted, never guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from draft4.catalog import Catalog, load_catalog
from draft4.rules import BLUE, RED, SKIPPED, DraftState

ROOT = Path(__file__).resolve().parent.parent
PRO_DIR = ROOT / "data" / "pro"
POSITION = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}


@dataclass
class ProDrafts:
    game_id: np.ndarray  # str
    series: np.ndarray  # str: games of one series share it (split unit)
    game_no: np.ndarray  # int
    league: np.ndarray  # str
    patch: np.ndarray  # str
    time: np.ndarray  # int64 ms
    first: np.ndarray  # side holding first pick
    blue_win: np.ndarray  # int8
    bans: np.ndarray  # (N,2,5) in each team's ban order, SKIPPED if none
    picks: np.ndarray  # (N,2,5) in each team's pick order
    roles: np.ndarray  # (N,2,5) role of each pick
    champ_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.game_id)

    def subset(self, mask) -> "ProDrafts":
        return ProDrafts(*(getattr(self, k)[mask] if k != "champ_ids" else self.champ_ids for k in self.__dataclass_fields__))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> "ProDrafts":
        with np.load(path, allow_pickle=False) as d:
            return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def actions(self, i: int) -> list[tuple[int, str, int, int]]:
        """Ordered (side, kind, champion, role) for game i, following the tournament sequence."""
        state = DraftState("pro", first=int(self.first[i]))
        out = []
        for step in state.steps:
            if step.kind == "ban":
                out.append((step.side, "ban", int(self.bans[i, step.side, step.slot]), -1))
            else:
                out.append((step.side, "pick", int(self.picks[i, step.side, step.slot]), int(self.roles[i, step.side, step.slot])))
        return out


def parse_rows(rows: list[dict], catalog: Catalog, stats: Counter) -> dict | None:
    teams = {r["side"]: r for r in rows if r["position"] == "team"}
    players = [r for r in rows if r["position"] in POSITION]
    if set(teams) != {"Blue", "Red"} or len(players) != 10:
        stats["bad_rows"] += 1
        return None
    fp = teams["Blue"]["firstPick"]
    if fp not in ("0", "1"):
        stats["unknown_first_pick"] += 1
        return None

    def champ(name: str) -> int | None:
        c = catalog.resolve(name) if name else None
        if name and c is None:
            stats[f"unknown_champion:{name}"] += 1
        return None if c is None else c.idx

    bans, picks, roles = [[], []], [[], []], [[], []]
    for side_name, side in (("Blue", BLUE), ("Red", RED)):
        t = teams[side_name]
        played = {}
        for p in players:
            if p["side"] == side_name:
                played[champ(p["champion"])] = POSITION[p["position"]]
        for k in range(1, 6):
            b = t[f"ban{k}"]
            bans[side].append(SKIPPED if not b else champ(b))
            pk = champ(t[f"pick{k}"])
            if pk is None or pk not in played:
                stats["pick_order_mismatch"] += 1
                return None
            picks[side].append(pk); roles[side].append(played[pk])
        if None in bans[side] or len(set(picks[side])) != 5:
            stats["bad_draft"] += 1
            return None
    used = [c for side in (0, 1) for c in bans[side] + picks[side] if c >= 0]
    if len(used) != len(set(used)):  # same champion banned and picked (or banned twice): data error
        stats["duplicate_champion"] += 1
        return None
    gid, b = teams["Blue"]["gameid"], teams["Blue"]
    m = re.match(r"^(\d+-\d+)_game_\d+$", gid)
    series = m.group(1) if m else f"{b['league']}|{b['date'][:10]}|{'|'.join(sorted(t['teamname'] for t in teams.values()))}"
    return {
        "game_id": gid, "series": series, "game_no": int(b["game"] or 1), "league": b["league"], "patch": b["patch"],
        "time": int(datetime.fromisoformat(b["date"]).timestamp() * 1000), "first": BLUE if fp == "1" else RED,
        "blue_win": int(b["result"] == "1"), "bans": bans, "picks": picks, "roles": roles,
    }


def import_years(years: list[int], catalog: Catalog, pro_dir: Path = PRO_DIR) -> tuple[ProDrafts, dict]:
    stats, out = Counter(), defaultdict(list)
    for y in years:
        path = pro_dir / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv"
        games = defaultdict(list)
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                games[r["gameid"]].append(r)
        for rows in games.values():
            stats["games_seen"] += 1
            g = parse_rows(rows, catalog, stats)
            if g is None:
                continue
            stats["games_kept"] += 1
            for k, v in g.items():
                out[k].append(v)
    order = np.argsort(np.array(out["time"]), kind="stable")
    arr = lambda k, dt=None: np.array(out[k], dtype=dt)[order]
    drafts = ProDrafts(arr("game_id", str), arr("series", str), arr("game_no", np.int64), arr("league", str), arr("patch", str),
                       arr("time", np.int64), arr("first", np.int64), arr("blue_win", np.int8), arr("bans", np.int64),
                       arr("picks", np.int64), arr("roles", np.int64), np.array([c.id for c in catalog.champions], dtype=np.int64))
    report = {"years": years, **{k: v for k, v in stats.items() if not k.startswith("unknown_champion")},
              "unknown_champions": {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("unknown_champion")},
              "red_first_pick_share": float((drafts.first == RED).mean()) if len(drafts) else 0.0,
              "leagues": dict(Counter(drafts.league.tolist()).most_common(15))}
    return drafts, report


def save_pro_priors(d: ProDrafts, path: Path, prior_games: float = 2.0) -> None:
    """P(role | champion) in pro play, lightly smoothed toward the coach seed for rarely seen champions."""
    from draft4.priors import Priors
    seed = Priors.from_seed(load_catalog()).role_rate
    counts = np.zeros_like(seed)
    np.add.at(counts, (d.picks.ravel(), d.roles.ravel()), 1)
    rr = (counts + prior_games * seed) / (counts.sum(1, keepdims=True) + prior_games)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, role_rate=rr, games=counts, champ_ids=d.champ_ids)


def series_time_split(d: ProDrafts, fractions=(0.8, 0.1, 0.1)):
    """Chronological split by series start, so no series straddles train and test."""
    start = {}
    for s, t in zip(d.series, d.time):
        start[s] = min(start.get(s, t), t)
    keys = sorted(start, key=start.get)
    a, b = int(len(keys) * fractions[0]), int(len(keys) * (fractions[0] + fractions[1]))
    rank = {k: i for i, k in enumerate(keys)}
    r = np.array([rank[s] for s in d.series])
    return d.subset(r < a), d.subset((r >= a) & (r < b)), d.subset(r >= b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "snapshots" / "pro_latest.npz")
    a = ap.parse_args()
    drafts, report = import_years(a.years, load_catalog())
    drafts.save(a.out)
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    save_pro_priors(drafts, ROOT / "models" / "candidates" / "pro_priors.npz")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
