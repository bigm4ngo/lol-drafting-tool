"""Draft Planner: budgeted, paired-rollout search over legal lines. The planner is the decision-maker;
the policy only proposes plausible actions and the value model only scores complete lineups."""

from __future__ import annotations

import hashlib
import itertools
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .catalog import ROLES, TRAITS, Catalog
from .gameplan import PLANS, GamePlanScorer
from .priors import Priors
from .rules import BLUE, EMPTY, SIDES, DraftState, Step

PERMS = {n: np.array(list(itertools.permutations(range(5), n))) for n in range(6)}
PLAN_TRAITS = {
    "bot_strong": ("early", "engage", "pick"), "top_strong": ("early", "splitpush", "dive"),
    "front_to_back": ("frontline", "peel", "teamfight", "scaling"), "pick_dive": ("pick", "dive", "engage"),
    "poke_siege": ("poke", "waveclear", "disengage"), "side_lane": ("splitpush", "waveclear", "disengage"),
}


@dataclass
class Prefs:
    our_side: int | None = None  # None: advise whichever side acts next
    plan: str | None = None
    pools: dict[int, list[int]] = field(default_factory=dict)  # role -> comfort champion idx
    pool_only: bool = False
    budget_ms: int = 1500
    n_lines: int = 3


def assign_roles(champs: list[int], locked: list[int], role_rate: np.ndarray, rng=None) -> np.ndarray:
    """Role-ordered lineup (5,) for current picks. Best assignment, or sampled when rng is given."""
    out = np.full(5, EMPTY)
    if not champs:
        return out
    perms = PERMS[len(champs)]
    lr = np.log(role_rate[champs] + 1e-9)  # (n,5)
    score = lr[np.arange(len(champs)), perms].sum(1)
    for i, r in enumerate(locked):
        if r >= 0:
            score = np.where(perms[:, i] == r, score, -np.inf)
    if not np.isfinite(score).any():  # contradictory locks: honour them in slot order
        score = np.zeros(len(perms))
    if rng is None:
        k = int(np.argmax(score))
    else:
        p = np.exp(score - score.max())
        k = int(rng.choice(len(p), p=p / p.sum()))
    out[perms[k]] = champs
    return out


def role_distribution(champs: list[int], locked: list[int], role_rate: np.ndarray) -> list[dict[int, float]]:
    """P(role) for each current pick, marginalised over feasible assignments (flex visibility)."""
    if not champs:
        return []
    perms = PERMS[len(champs)]
    score = np.log(role_rate[champs] + 1e-9)[np.arange(len(champs)), perms].sum(1)
    for i, r in enumerate(locked):
        if r >= 0:
            score = np.where(perms[:, i] == r, score, -np.inf)
    p = np.exp(score - score.max()); p /= p.sum()
    out = []
    for i in range(len(champs)):
        d = Counter()
        for k, pr in enumerate(p):
            d[int(perms[k, i])] += pr
        out.append({r: round(v, 3) for r, v in d.items() if v >= 0.05})
    return out


class Sim:
    """Mutable rollout state: role-ordered teams, bans and the legal mask."""
    __slots__ = ("team", "bans", "legal", "policy", "step", "rr")

    def __init__(self, team: np.ndarray, bans: list[int], legal: np.ndarray, policy, step: int | None, rr: np.ndarray) -> None:
        self.team, self.bans, self.legal = team, bans, legal
        self.policy, self.step = policy, step  # format-appropriate policy; current ordered-draft position
        self.rr = rr  # format-appropriate P(role | champion): pro role habits differ from solo queue

    def ban(self, c: int) -> None:
        if c >= 0:
            self.bans.append(c); self.legal[c] = False

    def pick(self, side: int, c: int, r: int) -> None:
        self.team[side, r] = c; self.legal[c] = False


class Engine:
    def __init__(self, catalog: Catalog, priors: Priors, policy, value, status: dict | None = None, pro_policy=None,
                 pro_role_rate: np.ndarray | None = None, pro_games: np.ndarray | None = None) -> None:
        self.cat, self.p, self.policy, self.value, self.pro_policy = catalog, priors, policy, value, pro_policy
        self.pro_role_rate, self.pro_games = pro_role_rate, pro_games
        self.status = status or {}
        self.plans = GamePlanScorer(catalog, priors)
        self.C = len(catalog)
        traits = self.plans.traits
        self.plan_affinity = {k: traits[:, [TRAITS.index(t) for t in v]].sum(1) for k, v in PLAN_TRAITS.items()}

    # ---------------------------------------------------------------- policy helpers
    def _joint(self, sim: Sim, side: int, our_side: int, prefs: Prefs) -> tuple[np.ndarray, np.ndarray]:
        """Joint (role, champion) probabilities for a pick by `side` -> (open_roles, P[len(open), C])."""
        open_roles = np.where(sim.team[side] < 0)[0]
        L = sim.policy.role_logits(sim.team[side], sim.team[1 - side], np.array(sim.bans), sim.step)[open_roles]
        if side == our_side:  # decision-maker: tilt toward value and comfort, sharper than imitation
            L = L + 1.0 * self.p.win_logit[:, open_roles].T
            for i, r in enumerate(open_roles):
                pool = prefs.pools.get(int(r))
                if pool:
                    if prefs.pool_only and sim.legal[pool].any():  # exhausted pool: fall back, don't dead-end
                        mask = np.full(self.C, -np.inf); mask[pool] = 0.0
                        L[i] = L[i] + mask
                    else:
                        L[i, pool] += 1.0
            L = L / 0.7
        L = np.where(sim.legal[None, :], L, -np.inf)
        L = L - np.max(L, axis=1, keepdims=True, where=np.isfinite(L), initial=-1e9)
        P = np.exp(L)
        P = np.nan_to_num(P / P.sum(1, keepdims=True).clip(1e-300)) / len(open_roles)
        return open_roles, P

    def _ban_probs(self, sim: Sim, side: int) -> np.ndarray:
        L = sim.policy.ban_logits(sim.team[1 - side], sim.team[side], np.array(sim.bans), sim.step)
        L = np.where(sim.legal, L, -np.inf)
        P = np.exp(L - L[np.isfinite(L)].max())
        return P / P.sum()

    def _sample_step(self, sim: Sim, step: Step, our_side: int, prefs: Prefs, rng) -> tuple[int, int]:
        if step.kind == "ban":
            P = self._ban_probs(sim, step.side)
            c = int(rng.choice(self.C, p=P))
            sim.ban(c)
            return c, EMPTY
        open_roles, P = self._joint(sim, step.side, our_side, prefs)
        flat = P.ravel()
        k = int(rng.choice(flat.size, p=flat / flat.sum()))
        r, c = int(open_roles[k // self.C]), k % self.C
        sim.pick(step.side, c, r)
        return c, r

    # ---------------------------------------------------------------- rollouts
    def role_rate(self, fmt: str) -> np.ndarray:
        return self.pro_role_rate if self.pro_role_rate is not None and fmt != "ranked" else self.p.role_rate

    def _root_sim(self, state: DraftState, rng=None, sample_side: int | None = None) -> Sim:
        rr = self.role_rate(state.fmt)
        team = np.full((2, 5), EMPTY)
        for s in (0, 1):
            champs = [c for c in state.picks[s] if c >= 0]
            locked = [state.roles[s][i] for i, c in enumerate(state.picks[s]) if c >= 0]
            team[s] = assign_roles(champs, locked, rr, rng if s == sample_side else None)
        policy = self.pro_policy if self.pro_policy is not None and state.fmt != "ranked" else self.policy
        return Sim(team, state.all_bans(), state.legal_mask(self.C), policy, state.next_index(), rr)

    def _rollouts(self, state: DraftState, first: tuple[Step, int, int] | None, our_side: int, prefs: Prefs,
                  seeds: list[int], force_reply: tuple[Step, int] | None = None) -> dict:
        """Play the draft out from `state` after applying `first` (step, champ, role)."""
        steps = state.remaining_steps()
        lineups, replies, follow = [], [], []
        index = {s: i for i, s in enumerate(state.steps)}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sim = self._root_sim(state, rng, sample_side=1 - our_side)  # opponent roles uncertain
            todo = steps
            if first is not None:
                step, c, r = first
                if step.kind == "ban":
                    sim.ban(c)
                else:
                    if r < 0 or sim.team[step.side, r] >= 0:
                        r = int(np.where(sim.team[step.side] < 0)[0][np.argmax(sim.rr[c, sim.team[step.side] < 0])])
                    sim.pick(step.side, c, r)
                todo = steps[1:]
            reply = None
            mine = None
            for j, step in enumerate(todo):
                sim.step = index[step]
                if j == 0 and force_reply is not None and force_reply[0] == step:
                    c = force_reply[1]
                    if step.kind == "ban":
                        sim.ban(c)
                    else:
                        open_roles = np.where(sim.team[step.side] < 0)[0]
                        sim.pick(step.side, c, int(open_roles[np.argmax(sim.rr[c, open_roles])]))
                else:
                    c, _ = self._sample_step(sim, step, our_side, prefs, rng)
                if reply is None and step.side != our_side:
                    reply = (step.kind, c)
                if mine is None and step.side == our_side and step.kind == "pick" and first is not None:
                    mine = c
            lineups.append(sim.team.copy())
            replies.append(reply)
            follow.append(mine)
        teams = np.array(lineups)  # (n,2,5)
        p_blue = self.value.win_prob(teams[:, 0], teams[:, 1])
        ours = p_blue if our_side == BLUE else 1 - p_blue
        cov = self.value.coverage(teams[:, 0], teams[:, 1]) if hasattr(self.value, "coverage") else np.ones(len(ours))
        return {"win": ours, "teams": teams, "replies": replies, "follow": follow, "coverage": cov}

    # ---------------------------------------------------------------- candidates
    def _static_pick_scores(self, sim: Sim, side: int) -> tuple[np.ndarray, np.ndarray]:
        open_roles = np.where(sim.team[side] < 0)[0]
        allies, enemies = sim.team[side][sim.team[side] >= 0], sim.team[1 - side][sim.team[1 - side] >= 0]
        rr = sim.rr[:, open_roles]
        per_role = np.where(rr >= 0.03, self.p.win_logit[:, open_roles] + np.log(rr + 1e-3) * 0.3, -np.inf)  # off-role: not a candidate
        best_role = open_roles[np.argmax(per_role, 1)]
        s = per_role.max(1) + (self.p.syn[allies].sum(0) if len(allies) else 0) + (self.p.ctr[:, enemies].sum(1) if len(enemies) else 0)
        return np.where(sim.legal, s, -np.inf), best_role

    def _pick_candidates(self, state: DraftState, step: Step, prefs: Prefs) -> dict[int, dict]:
        sim = self._root_sim(state)
        open_roles, P = self._joint(sim, step.side, -1, prefs)  # imitation view: what people pick here
        marginal = P.sum(0)
        static, best_role = self._static_pick_scores(sim, step.side)
        static = self._pro_gate(state, static)  # player-pool picks are exempt below
        cands: dict[int, dict] = {}

        def add(c: int, reason: str, role: int | None = None) -> None:
            if not sim.legal[c]:
                return
            if prefs.pool_only and prefs.pools and not any(c in prefs.pools.get(int(r), []) for r in open_roles):
                return
            entry = cands.setdefault(int(c), {"role": int(role if role is not None else best_role[c]), "reasons": []})
            if reason not in entry["reasons"]:
                entry["reasons"].append(reason)

        for c in np.argsort(-marginal)[:6]:
            add(c, "likely here", int(open_roles[np.argmax(P[:, c])]))
        for c in np.argsort(-static)[:6]:
            add(c, "strong on current evidence")
        plan = prefs.plan
        if plan in PLAN_TRAITS:
            fit = np.where(sim.legal, self.plan_affinity[plan] + 0.5 * np.tanh(static), -np.inf)
            for c in np.argsort(-fit)[:4]:
                add(c, f"fits {PLANS[plan]['label']}")
        if len(open_roles) >= 2:
            flexy = (sim.rr[:, open_roles] >= 0.2).sum(1) >= 2
            for c in [c for c in np.argsort(-static) if flexy[c]][:3]:
                add(c, "flex: hides role assignment")
        for r in open_roles:
            pool = [c for c in prefs.pools.get(int(r), []) if sim.legal[c]]
            for c in sorted(pool, key=lambda c: -static[c])[:3]:
                add(c, f"comfort pick ({ROLES[r].lower()})", int(r))
        return cands

    def _pro_gate(self, state: DraftState, scores: np.ndarray) -> np.ndarray:
        """Solo-queue win rates alone don't make a pro pick: in pro formats require >= 0.5% pro presence."""
        if state.fmt == "ranked" or self.pro_games is None:
            return scores
        return np.where(self.pro_games >= 0.005 * self.pro_games.sum() / 10, scores, -np.inf)

    def _ban_candidates(self, state: DraftState, step: Step, prefs: Prefs) -> dict[int, dict]:
        sim = self._root_sim(state)
        opp = 1 - step.side
        cands: dict[int, dict] = {}
        if (sim.team[opp] < 0).any():
            sim.step = next((i for i, s in enumerate(state.steps) if s.side == opp and s.kind == "pick"
                             and state.slot_value(s) == EMPTY), sim.step)  # their next pick, for ordered policies
            _, P = self._joint(sim, opp, -1, prefs)
            for c in np.argsort(-P.sum(0))[:7]:
                cands.setdefault(int(c), {"role": EMPTY, "reasons": []})["reasons"].append("likely enemy pick")
            static, _ = self._pro_gate(state, self._static_pick_scores(sim, opp)[0]), None
            for c in np.argsort(-static)[:5]:
                e = cands.setdefault(int(c), {"role": EMPTY, "reasons": []})
                e["reasons"].append("dangerous for the enemy to get")
        P = self._ban_probs(sim, step.side)
        for c in np.argsort(-P)[:4]:
            e = cands.setdefault(int(c), {"role": EMPTY, "reasons": []})
            e["reasons"].append("commonly banned")
        # never suggest banning our own comfort picks
        ours = {c for pool in prefs.pools.values() for c in pool}
        return {c: v for c, v in cands.items() if sim.legal[c] and c not in ours}

    # ---------------------------------------------------------------- analysis
    def analyze(self, state: DraftState, prefs: Prefs) -> dict:
        t0 = time.perf_counter()
        step = state.next_step()
        base = {"state": state.fingerprint(), "versions": self.versions()}
        if step is None:
            return {**base, "kind": "complete", **self._final(state, prefs), "elapsed_ms": self._ms(t0)}
        our_side = step.side if prefs.our_side is None else prefs.our_side
        seed0 = int(hashlib.sha1(f"{state.fingerprint()}|{our_side}|{prefs.plan}".encode()).hexdigest()[:8], 16)
        if step.side != our_side:
            return {**base, "kind": "forecast", "step": self._step_json(step), "our_side": SIDES[our_side],
                    **self._forecast(state, step, our_side, prefs, seed0, t0), "elapsed_ms": self._ms(t0)}
        cands = (self._pick_candidates if step.kind == "pick" else self._ban_candidates)(state, step, prefs)
        if not cands:
            return {**base, "kind": "no_options", "step": self._step_json(step), "our_side": SIDES[our_side],
                    "message": "No legal candidate satisfies the constraints (check pools / pool-only).",
                    "elapsed_ms": self._ms(t0)}
        # Deterministic schedule from the budget tier so identical inputs give identical advice;
        # wall-clock is only a hard safety cap (2x budget), which can make an overloaded run shallower.
        max_n = 16 if prefs.budget_ms < 1000 else 32 if prefs.budget_ms < 2500 else 64
        hard_stop = t0 + 2 * prefs.budget_ms / 1000
        n, alive, results = 8, list(cands), {}
        baseline = None
        while True:
            seeds = [seed0 + k for k in range(n)]
            for c in alive:
                results[c] = self._rollouts(state, (step, c, cands[c]["role"]), our_side, prefs, seeds)
            if step.kind == "ban":
                baseline = self._rollouts(state, None, our_side, prefs, seeds)
            if n >= max_n or time.perf_counter() > hard_stop:
                break
            keep = max(prefs.n_lines + 2, len(alive) // 2)
            alive = sorted(alive, key=lambda c: -results[c]["win"].mean())[:keep]
            n *= 2
        lines = self._rank(alive, results, cands, prefs, step, our_side, state, baseline)
        chosen = lines[: prefs.n_lines]
        for line in chosen:  # credible punish + pivot only for shown lines (bounded extra work)
            if time.perf_counter() < hard_stop:
                self._punish(line, state, step, our_side, prefs, seed0, n)
            line.setdefault("punish", None)
            line["explain"] = self._explain(line)
        return {**base, "kind": "recommend", "step": self._step_json(step), "our_side": SIDES[our_side],
                "plan": prefs.plan, "lines": chosen, "also": [self._brief(l) for l in lines[prefs.n_lines:prefs.n_lines + 5]],
                "rollouts_per_line": n, "candidates": len(cands), "elapsed_ms": self._ms(t0),
                "notes": self._notes(state, results, chosen)}

    def _rank(self, alive, results, cands, prefs, step, our_side, state, baseline) -> list[dict]:
        lines = []
        for c in alive:
            res = results[c]
            w = res["win"]
            ours = res["teams"][:, our_side]; theirs = res["teams"][:, 1 - our_side]
            k = min(16, len(w))
            plan = prefs.plan
            if plan in PLANS:
                plan_rows = [self.plans.score(plan, ours[i], theirs[i]) for i in range(k)]
            else:
                plan_rows = [self.plans.best(ours[i], theirs[i])[0] for i in range(k)]
                plan = Counter(r["plan"] for r in plan_rows).most_common(1)[0][0]
                plan_rows = [self.plans.score(plan, ours[i], theirs[i]) for i in range(k)]
            unmet = Counter(u for r in plan_rows for u in r["unmet"])
            risks = Counter(u for r in plan_rows for u in r["risks"])
            reply = Counter(r for r in res["replies"] if r is not None).most_common(1)
            top_half = np.argsort(-w)[: max(1, len(w) // 2)]
            follow = Counter(res["follow"][i] for i in top_half if res["follow"][i] is not None).most_common(1)
            ch = self.cat.champions[c]
            line = {
                "champion": ch.id, "name": ch.name, "action": step.kind,
                "role": ROLES[cands[c]["role"]] if cands[c]["role"] >= 0 else None,
                "reasons": cands[c]["reasons"],
                "win": round(float(w.mean()), 4), "se": round(float(w.std() / np.sqrt(len(w))), 4),
                "downside": round(float(np.percentile(w, 10)), 4), "spread": round(float(w.std()), 4),
                "coverage": round(float(res["coverage"].mean()), 3),
                "plan": {"plan": plan, "label": PLANS[plan]["label"], "fit": round(float(np.mean([r["fit"] for r in plan_rows])), 3),
                         "often_unmet": [u for u, _ in unmet.most_common(2)], "risks": [u for u, n_ in risks.most_common(3) if n_ >= k / 3]},
                "likely_reply": self._action_json(reply[0][0], reply[0][1] / len(w)) if reply else None,
                "follow_up": self._champ_json(follow[0][0]) if follow else None,
            }
            if step.kind == "pick":
                line["preserves"] = self._preserves(state, step, c, cands[c]["role"])
            if baseline is not None:
                line["delta_vs_typical_ban"] = round(float((w - baseline["win"]).mean()), 4)
                line["substitute"] = self._substitute(c, res, our_side, self.role_rate(state.fmt))
            lines.append(line)
        best_c = max(alive, key=lambda c: results[c]["win"].mean())
        for c, l in zip(alive, lines):  # "too close to call": paired difference within noise or 1 pt
            diff = results[best_c]["win"] - results[c]["win"]  # same seeds -> paired comparison
            l["close_to_best"] = bool(diff.mean() <= max(0.01, 2 * diff.std() / np.sqrt(len(diff))))
        lines.sort(key=lambda l: (not l["close_to_best"], -l["plan"]["fit"] if l["close_to_best"] else 0, -l["win"], -l["downside"]))
        for i, l in enumerate(lines):
            l["rank"] = i + 1
        return lines

    def _punish(self, line: dict, state: DraftState, step: Step, our_side: int, prefs: Prefs, seed0: int, n: int) -> None:
        c = self.cat.by_id[line["champion"]].idx
        role = ROLES.index(line["role"]) if line["role"] else EMPTY
        after = state.apply(step, c, role if role >= 0 else EMPTY, check=False)
        nxt = after.next_step()
        if nxt is None or nxt.side == our_side:  # shortcut: no punish search across our own double pick
            return
        sim = self._root_sim(after)
        if nxt.kind == "pick":
            _, P = self._joint(sim, nxt.side, -1, prefs)
            options = np.argsort(-P.sum(0))[:6]
        else:
            options = np.argsort(-self._ban_probs(sim, nxt.side))[:6]
        seeds = [seed0 + 1000 + k for k in range(max(8, n // 2))]
        worst = None
        for o in options:
            res = self._rollouts(after, None, our_side, prefs, seeds, force_reply=(nxt, int(o)))
            m = float(res["win"].mean())
            if worst is None or m < worst[1]:
                worst = (int(o), m, res)
        o, m, res = worst
        if m - line["win"] > -0.005:  # nothing meaningfully worse than the average reply
            line["punish"] = None
            return
        top_half = np.argsort(-res["win"])[: max(1, len(seeds) // 2)]
        ours = [int(res["teams"][i, our_side][r]) for i in top_half for r in range(5)
                if res["teams"][i, our_side][r] >= 0 and res["teams"][i, our_side][r] not in after.picks[our_side]]
        pivot = Counter(ours).most_common(1)
        line["punish"] = {**self._action_json((nxt.kind, o), None), "win_if": round(m, 4),
                          "delta": round(m - line["win"], 4),
                          "pivot_to": self._champ_json(pivot[0][0]) if pivot else None}

    def _forecast(self, state, step, our_side, prefs, seed0, t0) -> dict:
        sim = self._root_sim(state)
        if step.kind == "pick":
            open_roles, P = self._joint(sim, step.side, -1, prefs)
            marginal = P.sum(0)
        else:
            marginal = self._ban_probs(sim, step.side)
        top = np.argsort(-marginal)[:8]
        seeds = [seed0 + k for k in range(16)]
        rows = []
        for c in top:
            res = self._rollouts(state, (step, int(c), EMPTY), our_side, prefs, seeds)
            rows.append({**self._champ_json(int(c)), "probability": round(float(marginal[c]), 4),
                         "our_win_if": round(float(res["win"].mean()), 4)})
            if time.perf_counter() - t0 > 2 * prefs.budget_ms / 1000:
                break
        rows.sort(key=lambda r: -r["probability"])
        threat = min(rows, key=lambda r: r["our_win_if"]) if rows else None
        return {"forecast": rows, "biggest_threat": threat,
                "message": f"{SIDES[step.side].title()} to {step.kind}. Forecast of their action and what each does to your outlook."}

    def _final(self, state: DraftState, prefs: Prefs) -> dict:
        sim = self._root_sim(state)
        p_blue = float(self.value.win_prob(sim.team[None, 0], sim.team[None, 1])[0])
        out = {"win_blue": round(p_blue, 4), "teams": {}}
        for s in (0, 1):
            champs = [c for c in state.picks[s] if c >= 0]
            locked = [state.roles[s][i] for i, c in enumerate(state.picks[s]) if c >= 0]
            out["teams"][SIDES[s]] = {
                "lineup": [self._champ_json(int(c)) | {"role": ROLES[r]} for r, c in enumerate(sim.team[s])],
                "plans": self.plans.best(sim.team[s], sim.team[1 - s])[:3],
                "role_uncertainty": [{"champion": self.cat.champions[c].name, "roles": {ROLES[r]: p for r, p in d.items()}}
                                     for c, d in zip(champs, role_distribution(champs, locked, sim.rr)) if len(d) > 1],
            }
        out["coverage"] = round(float(self.value.coverage(sim.team[None, 0], sim.team[None, 1])[0]), 3)
        return out

    # ---------------------------------------------------------------- formatting
    def versions(self) -> dict:
        return {"policy": self.policy.version, "value": self.value.version,
                "priors": self.p.meta.get("source"), "matches": self.p.meta.get("matches", 0), "catalog": self.cat.version}

    def _champ_json(self, c: int) -> dict:
        ch = self.cat.champions[int(c)]
        return {"champion": ch.id, "name": ch.name}

    def _action_json(self, action, freq) -> dict:
        kind, c = action
        return {**self._champ_json(c), "kind": kind, **({"frequency": round(freq, 3)} if freq is not None else {})}

    def _step_json(self, step: Step) -> dict:
        return {"side": SIDES[step.side], "kind": step.kind, "slot": step.slot}

    def _preserves(self, state: DraftState, step: Step, c: int, role: int) -> dict:
        sim = self._root_sim(state)
        open_after = [ROLES[r] for r in range(5) if sim.team[step.side, r] < 0 and r != role]
        flex = [ROLES[r] for r in np.where(sim.rr[c] >= 0.2)[0]]
        return {"open_roles": open_after, "flex_roles": flex if len(flex) > 1 else []}

    def _substitute(self, banned: int, res: dict, our_side: int, rr: np.ndarray):
        r = int(np.argmax(rr[banned]))
        picks = Counter(int(t[1 - our_side, r]) for t in res["teams"] if t[1 - our_side, r] >= 0).most_common(1)
        return self._champ_json(picks[0][0]) | {"role": ROLES[r]} if picks else None

    def _brief(self, l: dict) -> dict:
        return {k: l[k] for k in ("champion", "name", "action", "role", "win", "downside", "reasons")}

    def _explain(self, l: dict) -> str:
        verb = "Ban" if l["action"] == "ban" else "Lock"
        role = f" ({l['role'].lower()})" if l.get("role") else ""
        s = [f"{verb} {l['name']}{role}: {l['win'] * 100:.1f}% average across plausible replies, "
             f"{l['downside'] * 100:.1f}% in the worst tenth of branches."]
        if l.get("delta_vs_typical_ban") is not None:
            s.append(f"{l['delta_vs_typical_ban'] * 100:+.1f} pts versus a typical ban here.")
            if l.get("substitute"):
                s.append(f"Likely substitute: {l['substitute']['name']}.")
        pres = l.get("preserves") or {}
        if pres.get("flex_roles"):
            s.append(f"Keeps {'/'.join(r.lower() for r in pres['flex_roles'])} assignment hidden.")
        if pres.get("open_roles"):
            s.append(f"Still to fill: {', '.join(r.lower() for r in pres['open_roles'])}.")
        if l.get("likely_reply"):
            r = l["likely_reply"]
            s.append(f"Most likely reply: {r['kind']} {r['name']} ({r['frequency'] * 100:.0f}% of branches).")
        p = l.get("punish")
        if p:
            s.append(f"Credible punish: {p['kind']} {p['name']} ({p['delta'] * 100:+.1f} pts)"
                     + (f"; if that happens, pivot toward {p['pivot_to']['name']}." if p.get("pivot_to") else "."))
        plan = l["plan"]
        s.append(f"{plan['label']} fit {plan['fit'] * 100:.0f}%"
                 + (f"; often missing: {plan['often_unmet'][0].lower()}" if plan["often_unmet"] else "") + ".")
        if plan["risks"]:
            s.append(f"Watch: {plan['risks'][0].lower()}.")
        return " ".join(s)

    def _notes(self, state, results, chosen) -> list[str]:
        notes = []
        if self.p.meta.get("source") == "seed":
            notes.append("No match data loaded: win estimates are flat; ranking leans on roles, coach traits and plan fit.")
        cov = np.mean([l["coverage"] for l in chosen]) if chosen else 1
        if cov < 0.5 and self.p.meta.get("source") != "seed":
            notes.append("Low data coverage for these lineups: treat small differences as ties.")
        if chosen and all(l["close_to_best"] for l in chosen):
            notes.append("Top lines are within noise of each other: choose on plan fit and comfort.")
        if state.fmt != "ranked" and self.p.meta.get("source") != "seed":
            notes.append("Win estimates come from ranked solo-queue games, not pro results; "
                         + ("opponent forecasts come from the pro draft policy." if self.pro_policy is not None else "no pro policy is loaded."))
        if state.fmt == "ranked":
            notes.append("Ranked simulation fills slots in pick order for convenience; it does not predict true pick timing.")
        return notes

    @staticmethod
    def _ms(t0: float) -> int:
        return int((time.perf_counter() - t0) * 1000)
