"""Transparent, coach-editable game-plan scoring: win condition + requirements + failure modes.

Plan fit is reported separately from win probability and is never added to it.
Traits come from data/champion_seed.txt (coach-authored); lane edges come from the priors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import TRAITS, Catalog
from .priors import Priors

TOP, JG, MID, ADC, SUP = range(5)
ROLE_NAME = ("top", "jungle", "mid", "bot", "support")


@dataclass
class Ctx:
    us: np.ndarray  # role-ordered idx
    them: np.ndarray
    t_us: np.ndarray  # (5, n_traits) 0/1
    t_them: np.ndarray
    edge: np.ndarray  # (5,) our lane edge in log-odds (+ = we are favoured)
    dmg_us: list[str]

    def has(self, role: int, *traits: str, side: str = "us") -> bool:
        t = self.t_us if side == "us" else self.t_them
        return any(t[role, TRAITS.index(x)] for x in traits)

    def count(self, *traits: str, side: str = "us") -> int:
        t = self.t_us if side == "us" else self.t_them
        return int(t[:, [TRAITS.index(x) for x in traits]].max(1).sum())  # champions with any of the traits


# Each requirement: (label, weight, test). Each risk: (label, test) evaluated against the enemy.
PLANS: dict[str, dict] = {
    "bot_strong": {
        "label": "Bot strong side",
        "win": "Funnel early pressure and resources bot; convert bot leads into dragons.",
        "req": [
            ("Bot lane projects ahead", 2.0, lambda c: c.edge[ADC] + c.edge[SUP] > 0.0),
            ("Jungler can play toward bot", 1.0, lambda c: c.has(JG, "early", "dive", "engage", "pick")),
            ("Top can hold with little help", 1.5, lambda c: c.edge[TOP] > -0.1 or c.has(TOP, "frontline", "scaling")),
            ("Bot/support has engage or kill pressure", 1.0, lambda c: c.has(SUP, "engage", "pick") or c.has(ADC, "early")),
        ],
        "risks": [
            ("Enemy top+jungle can punish an isolated top", lambda c: c.has(TOP, "early", "dive", side="them") and c.has(JG, "early", "dive", "pick", side="them") and c.edge[TOP] < 0.05),
            ("Enemy bot outscales if the lane stalls", lambda c: c.has(ADC, "scaling", side="them") and not c.has(ADC, "scaling")),
        ],
    },
    "top_strong": {
        "label": "Top strong side",
        "win": "Play through top: jungle covers top, top creates pressure or a split.",
        "req": [
            ("Top lane projects ahead", 2.0, lambda c: c.edge[TOP] > 0.0),
            ("Top has a carry or pressure pattern", 1.0, lambda c: c.has(TOP, "early", "splitpush", "dive")),
            ("Jungler can play toward top", 1.0, lambda c: c.has(JG, "early", "dive", "engage", "pick")),
            ("Bot can survive on low resources", 1.5, lambda c: c.edge[ADC] + c.edge[SUP] > -0.15 or c.has(SUP, "peel", "disengage")),
        ],
        "risks": [
            ("Enemy bot+jungle can snowball the weak side", lambda c: c.has(ADC, "early", side="them") and c.has(SUP, "engage", "pick", side="them")),
            ("Enemy top can absorb pressure and scale", lambda c: c.has(TOP, "frontline", "scaling", side="them")),
        ],
    },
    "front_to_back": {
        "label": "Front-to-back teamfight",
        "win": "Win grouped fights with a front line protecting a scaling carry.",
        "req": [
            ("Two or more front-line champions", 1.5, lambda c: c.count("frontline") >= 2),
            ("A carry that wants long fights", 1.0, lambda c: c.has(ADC, "scaling", "teamfight") or c.has(MID, "scaling", "teamfight")),
            ("Peel or disengage for the carry", 1.0, lambda c: c.count("peel", "disengage") >= 1),
            ("Teamfight tools (engage/teamfight) on 2+ champions", 1.0, lambda c: c.count("engage", "teamfight") >= 2),
        ],
        "risks": [
            ("Enemy poke can chip you before the fight starts", lambda c: c.count("poke", side="them") >= 2),
            ("Enemy catch tools can split the carry from the front line", lambda c: c.count("pick", side="them") >= 3),
        ],
    },
    "pick_dive": {
        "label": "Pick / dive",
        "win": "Catch or dive isolated targets before the enemy can group.",
        "req": [
            ("Three or more pick/dive/engage threats", 2.0, lambda c: c.count("pick", "dive", "engage") >= 3),
            ("Jungler starts plays early", 1.0, lambda c: c.has(JG, "early", "dive", "pick")),
            ("Solo-lane follow-up", 1.0, lambda c: c.has(MID, "dive", "pick", "engage") or c.has(TOP, "dive", "engage")),
        ],
        "risks": [
            ("Enemy peel/disengage blunts the dive", lambda c: c.count("peel", "disengage", side="them") >= 3),
            ("Enemy front line absorbs the first engage", lambda c: c.count("frontline", side="them") >= 3),
        ],
    },
    "poke_siege": {
        "label": "Poke / siege",
        "win": "Win sieges with range and wave clear; avoid hard fights until the enemy is low.",
        "req": [
            ("Two or more poke threats", 2.0, lambda c: c.count("poke") >= 2),
            ("Wave clear to set up sieges", 1.0, lambda c: c.count("waveclear") >= 2),
            ("Disengage or peel to deny hard engage", 1.0, lambda c: c.count("disengage", "peel") >= 1),
        ],
        "risks": [
            ("Hard engage can skip the poke phase", lambda c: c.count("engage", "dive", side="them") >= 3),
            ("Enemy sustain/peel neutralises chip damage", lambda c: c.count("peel", side="them") >= 2),
        ],
    },
    "side_lane": {
        "label": "Side-lane pressure",
        "win": "Split pressure on a side lane while four players hold or trade objectives.",
        "req": [
            ("A credible split-pusher", 2.0, lambda c: c.count("splitpush") >= 1),
            ("Wave clear so four can hold", 1.0, lambda c: c.count("waveclear") >= 2),
            ("Disengage or a fast engage to punish over-commits", 1.0, lambda c: c.count("disengage", "engage") >= 1),
        ],
        "risks": [
            ("Enemy can force 5v4 fights faster than you split", lambda c: c.count("engage", "teamfight", side="them") >= 4),
            ("Enemy has a side-laner that matches the split", lambda c: c.count("splitpush", side="them") >= 1),
        ],
    },
}

GENERAL = [
    ("Damage is heavily one type (easy to itemise against)", lambda c: max(c.dmg_us.count("AD"), c.dmg_us.count("AP")) >= 4),
    ("No front line", lambda c: c.count("frontline") == 0),
    ("No reliable engage", lambda c: c.count("engage") == 0),
]


class GamePlanScorer:
    def __init__(self, catalog: Catalog, priors: Priors) -> None:
        self.cat, self.p = catalog, priors
        self.traits = np.array([[t in ch.traits for t in TRAITS] for ch in catalog.champions], dtype=np.int8)
        early = self.traits[:, TRAITS.index("early")].astype(float)
        self._early = early

    def ctx(self, us: np.ndarray, them: np.ndarray) -> Ctx:
        r = np.arange(5)
        edge = self.p.win_logit[us, r] - self.p.win_logit[them, r] + self.p.lane[us, them]
        edge = edge + 0.1 * (self._early[us] - self._early[them])  # coach-trait tilt when data is thin
        return Ctx(us, them, self.traits[us], self.traits[them], edge, [self.cat.champions[i].damage for i in us])

    def score(self, plan: str, us: np.ndarray, them: np.ndarray) -> dict:
        c, spec = self.ctx(us, them), PLANS[plan]
        met, unmet, total, got = [], [], 0.0, 0.0
        for label, weight, test in spec["req"]:
            total += weight
            if test(c):
                got += weight
                met.append(label)
            else:
                unmet.append(label)
        risks = [label for label, test in spec["risks"] if test(c)] + [label for label, test in GENERAL if test(c)]
        return {"plan": plan, "label": spec["label"], "fit": round(got / total, 3), "met": met, "unmet": unmet,
                "risks": risks, "lane_edge": {ROLE_NAME[i]: round(float(e), 3) for i, e in enumerate(c.edge)}}

    def best(self, us: np.ndarray, them: np.ndarray) -> list[dict]:
        return sorted((self.score(p, us, them) for p in PLANS), key=lambda d: (-d["fit"], len(d["risks"])))

    def fit_batch(self, plan: str, us: np.ndarray, them: np.ndarray) -> np.ndarray:
        return np.array([self.score(plan, u, t)["fit"] for u, t in zip(us, them)])
