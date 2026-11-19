"""Write a fake collector-schema SQLite DB with planted structure (for smoke tests only).

    python -m train.synthetic --out data/synthetic.sqlite3 --matches 20000
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from draft4.catalog import ROLES, load_catalog

SCHEMA = """
CREATE TABLE matches (match_id TEXT PRIMARY KEY, game_creation INTEGER NOT NULL, game_duration INTEGER NOT NULL,
  game_version TEXT NOT NULL, queue_id INTEGER NOT NULL, target_tier TEXT NOT NULL, winning_team_id INTEGER, fetched_at TEXT NOT NULL);
CREATE TABLE participants (match_id TEXT NOT NULL, participant_id INTEGER NOT NULL, puuid TEXT, team_id INTEGER NOT NULL,
  champion_id INTEGER NOT NULL, champion_name TEXT NOT NULL, role TEXT NOT NULL, win INTEGER NOT NULL, PRIMARY KEY(match_id, participant_id));
CREATE TABLE bans (match_id TEXT NOT NULL, team_id INTEGER NOT NULL, champion_id INTEGER NOT NULL, pick_turn INTEGER NOT NULL,
  PRIMARY KEY(match_id, team_id, pick_turn));
"""


def generate(out: Path, n_matches: int, seed: int = 7) -> dict:
    cat = load_catalog()
    rng = np.random.default_rng(seed)
    C = len(cat)
    strength = rng.normal(0, 0.25, C)
    pop = rng.gamma(0.8, 1.0, C)
    by_role = [[c.idx for c in cat.champions if r in c.roles] for r in range(5)]
    syn_pairs = {tuple(sorted(rng.choice(C, 2, replace=False))): 0.6 for _ in range(60)}
    ctr = {(int(a), int(b)): 0.5 for a, b in rng.choice(C, (60, 2))}
    out.unlink(missing_ok=True)
    conn = sqlite3.connect(out)
    conn.executescript(SCHEMA)
    t0 = 1_760_000_000_000
    for m in range(n_matches):
        used: set[int] = set()
        teams = []
        for _side in range(2):
            lineup = []
            for r in range(5):
                pool = [c for c in by_role[r] if c not in used]
                w = pop[pool] / pop[pool].sum()
                c = int(rng.choice(pool, p=w))
                used.add(c); lineup.append(c)
            teams.append(lineup)
        logit = 0.05 + strength[teams[0]].sum() - strength[teams[1]].sum()
        for s, sign in ((0, 1), (1, -1)):
            for i in range(5):
                for j in range(i + 1, 5):
                    logit += sign * syn_pairs.get(tuple(sorted((teams[s][i], teams[s][j]))), 0)
        for a in teams[0]:
            for b in teams[1]:
                logit += ctr.get((a, b), 0) - ctr.get((b, a), 0)
        blue_win = rng.random() < 1 / (1 + np.exp(-logit))
        mid = f"SYN_{m:07d}"
        patch = f"16.{20 + m * 4 // n_matches}.1"
        conn.execute("INSERT INTO matches VALUES (?,?,?,?,?,?,?,?)",
                     (mid, t0 + m * 60_000, 1800, patch, 420, "EMERALD", 100 if blue_win else 200, "synthetic"))
        rows = []
        for s, team_id in ((0, 100), (1, 200)):
            for r, c in enumerate(teams[s]):
                ch = cat.champions[c]
                rows.append((mid, s * 5 + r + 1, None, team_id, ch.id, ch.key, ROLES[r], int(blue_win == (s == 0))))
        conn.executemany("INSERT INTO participants VALUES (?,?,?,?,?,?,?,?)", rows)
        free = [c for c in np.argsort(-pop) if c not in used][:20]
        bans = rng.choice(free, 10, replace=False)
        conn.executemany("INSERT INTO bans VALUES (?,?,?,?)",
                         [(mid, 100 if k < 5 else 200, cat.champions[int(c)].id, k + 1) for k, c in enumerate(bans)])
    conn.commit()
    conn.close()
    return {"matches": n_matches, "path": str(out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic.sqlite3"))
    ap.add_argument("--matches", type=int, default=20000)
    a = ap.parse_args()
    print(generate(a.out, a.matches))
