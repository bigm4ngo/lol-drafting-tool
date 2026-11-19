"""Smoothed statistical priors: the baseline policy/value evidence and the fallback for neural models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .catalog import Catalog
from .dataset import Matches


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Priors:
    role_rate: np.ndarray  # (C,5) P(role | champion played)
    pick_rate: np.ndarray  # (C,5) P(champion | role slot)
    win_logit: np.ndarray  # (C,5) shrunk win log-odds when played in role
    games: np.ndarray  # (C,5) observed games (coverage)
    syn: np.ndarray  # (C,C) same-team residual log-odds
    ctr: np.ndarray  # (C,C) ctr[a,b]: residual log-odds of a facing b (antisymmetric)
    lane: np.ndarray  # (C,C) same-role lane residual
    ban_rate: np.ndarray  # (C,)
    side_logit: float = 0.0
    pair_weight: float = 0.25  # scale on syn/ctr in the additive value; fitted on the calibration window
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in ("role_rate", "pick_rate", "win_logit", "games", "syn", "ctr", "lane", "ban_rate")}
        np.savez_compressed(path, side_logit=self.side_logit, pair_weight=self.pair_weight, meta=json.dumps(self.meta), **arrays)

    @classmethod
    def load(cls, path: Path, catalog: Catalog) -> "Priors":
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"]))
            p = cls(**{k: d[k] for k in ("role_rate", "pick_rate", "win_logit", "games", "syn", "ctr", "lane", "ban_rate")},
                    side_logit=float(d["side_logit"]), pair_weight=float(d["pair_weight"]), meta=meta)
        return p.remap(meta["champ_ids"], catalog)

    def remap(self, champ_ids: list[int], catalog: Catalog) -> "Priors":
        """Re-index to the current catalog; champions new since the build fall back to seed rows."""
        if list(champ_ids) == [c.id for c in catalog.champions]:
            return self
        seed = Priors.from_seed(catalog)
        src = {cid: i for i, cid in enumerate(champ_ids)}
        rows = np.array([src.get(c.id, -1) for c in catalog.champions])
        ok = rows >= 0
        for name in ("role_rate", "pick_rate", "win_logit", "games", "ban_rate"):
            arr = getattr(seed, name).copy()
            arr[ok] = getattr(self, name)[rows[ok]]
            setattr(seed, name, arr)
        for name in ("syn", "ctr", "lane"):
            arr = np.zeros_like(getattr(seed, name))
            ix = np.where(ok)[0]
            arr[np.ix_(ix, ix)] = getattr(self, name)[np.ix_(rows[ix], rows[ix])]
            setattr(seed, name, arr)
        seed.side_logit, seed.pair_weight, seed.meta = self.side_logit, self.pair_weight, {**self.meta, "champ_ids": [c.id for c in catalog.champions]}
        return seed

    @classmethod
    def from_seed(cls, catalog: Catalog) -> "Priors":
        """Uniform-evidence priors from the coach seed file: role knowledge only, no win evidence."""
        C = len(catalog)
        role_rate = np.full((C, 5), 0.01)
        for c in catalog.champions:
            for rank, r in enumerate(c.roles):
                role_rate[c.idx, r] += 0.75 if rank == 0 else 0.25 / max(1, len(c.roles) - 1)
        role_rate /= role_rate.sum(1, keepdims=True)
        pick_rate = role_rate / role_rate.sum(0, keepdims=True)
        z = np.zeros((C, C))
        return cls(role_rate, pick_rate, np.zeros((C, 5)), np.zeros((C, 5)), z, z.copy(), z.copy(),
                   np.full(C, 10 / C), 0.0, 0.25, {"source": "seed", "matches": 0, "champ_ids": [c.id for c in catalog.champions]})

    @classmethod
    def build(cls, m: Matches, catalog: Catalog, *, m_champ=100.0, m_pair=400.0, min_pair_games=50) -> "Priors":
        """Fit from matches that are all strictly before the evaluation cutoff (caller's responsibility)."""
        if list(m.champ_ids) != [c.id for c in catalog.champions]:
            raise ValueError("snapshot champion ids differ from catalog; rebuild the snapshot")
        C, N = len(catalog), len(m)
        bw = m.blue_win.astype(float)
        games, wins = np.zeros((C, 5)), np.zeros((C, 5))
        for team, won in ((m.blue, bw), (m.red, 1 - bw)):
            for r in range(5):
                np.add.at(games, (team[:, r], r), 1)
                np.add.at(wins, (team[:, r], r), won)
        role_rate = (games + 0.2 * Priors.from_seed(catalog).role_rate) / (games.sum(1, keepdims=True) + 0.2)
        pick_rate = (games + 0.5) / (games + 0.5).sum(0, keepdims=True)
        win_logit = _logit((wins + m_champ * 0.5) / (games + m_champ))
        champ_logit = _logit((wins.sum(1) + m_champ * 0.5) / (games.sum(1) + m_champ))

        def pair_table(pairs_a, pairs_b, won):
            g, w = np.zeros((C, C)), np.zeros((C, C))
            np.add.at(g, (pairs_a, pairs_b), 1)
            np.add.at(w, (pairs_a, pairs_b), won)
            return g, w

        # same-team synergy
        ia, ib = np.triu_indices(5, 1)
        sa, sb, sw = [], [], []
        for team, won in ((m.blue, bw), (m.red, 1 - bw)):
            sa.append(team[:, ia].ravel()); sb.append(team[:, ib].ravel()); sw.append(np.repeat(won, len(ia)))
        a, b, w_ = np.concatenate(sa), np.concatenate(sb), np.concatenate(sw)
        g, w = pair_table(np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([w_, w_]))
        exp_p = _sigmoid(champ_logit[:, None] + champ_logit[None, :])
        syn = np.where(g >= min_pair_games, _logit((w + m_pair * exp_p) / (g + m_pair)) - _logit(exp_p), 0.0)
        np.fill_diagonal(syn, 0.0)

        def cross(blue_cols, red_cols):
            a = np.concatenate([m.blue[:, blue_cols].ravel(), m.red[:, red_cols].ravel()])
            b = np.concatenate([m.red[:, red_cols].ravel(), m.blue[:, blue_cols].ravel()])
            reps = len(blue_cols)
            won = np.concatenate([np.repeat(bw, reps), np.repeat(1 - bw, reps)])
            g, w = pair_table(a, b, won)
            exp_p = _sigmoid(champ_logit[:, None] - champ_logit[None, :])
            out = np.where(g >= min_pair_games, _logit((w + m_pair * exp_p) / (g + m_pair)) - _logit(exp_p), 0.0)
            return (out - out.T) / 2  # enforce antisymmetry

        bi, ri = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
        ctr = cross(bi.ravel(), ri.ravel())
        lane = cross(np.arange(5), np.arange(5))

        ban_counts = np.zeros(C)
        valid = m.bans[m.bans >= 0]
        np.add.at(ban_counts, valid, 1)
        return cls(role_rate, pick_rate, win_logit, games, syn, ctr, lane, ban_counts / max(N, 1),
                   float(_logit((bw.sum() + 50) / (N + 100))), 0.25, {
                       "source": "matches", "matches": int(N),
                       "patches": sorted(set(m.patch.tolist()))[-5:],
                       "until_ms": int(m.creation.max()) if N else 0,
                       "champ_ids": [c.id for c in catalog.champions],
                   })
