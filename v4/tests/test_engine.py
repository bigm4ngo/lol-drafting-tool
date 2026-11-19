import time
from pathlib import Path

import numpy as np
import pytest

from draft4.catalog import load_catalog
from draft4.dataset import load_matches
from draft4.lcu import session_to_state
from draft4.models import AdditiveValue, PriorPolicy
from draft4.planner import Engine, Prefs
from draft4.priors import Priors
from draft4.rules import BLUE, EMPTY, RED, SKIPPED, DraftState
from train.synthetic import generate

CAT = load_catalog()


@pytest.fixture(scope="module")
def synthetic_priors(tmp_path_factory) -> Priors:
    db = tmp_path_factory.mktemp("db") / "syn.sqlite3"
    generate(db, 4000)
    return Priors.build(load_matches(db, CAT), CAT)


def engine(priors: Priors) -> Engine:
    return Engine(CAT, priors, PriorPolicy(priors), AdditiveValue(priors))


# ------------------------------------------------------------------ rules
def test_pro_order():
    steps = DraftState("pro").steps
    code = "".join(("B" if s.side == BLUE else "R") + s.kind[0] for s in steps)
    assert code == "BbRbBbRbBbRb" "BpRpRpBpBpRp" "RbBbRbBb" "RpBpBpRp"
    assert len(steps) == 20


def test_ranked_order_and_shared_bans():
    s = DraftState("ranked")
    picks = [x for x in s.steps if x.kind == "pick"]
    assert "".join("BR"[x.side] for x in picks) == "BRRBBRRBBR"
    s = s.apply(s.next_step(), 5)
    assert s.is_legal(s.next_step(), 5)  # red may also ban it in ranked
    assert not DraftState("pro").apply(DraftState("pro").next_step(), 5).is_legal(DraftState("pro").steps[1], 5)


def test_illegal_actions_rejected():
    s = DraftState("pro")
    s = s.apply(s.next_step(), 10)
    with pytest.raises(ValueError):
        s.apply(s.next_step(), 10)
    f = DraftState("fearless", unavailable=frozenset({3}))
    assert not f.is_legal(f.next_step(), 3)
    with pytest.raises(ValueError):
        DraftState.from_json({"fmt": "pro", "picks": [[CAT.champions[0].id] + [-1] * 4, [CAT.champions[0].id] + [-1] * 4]},
                             {c.id: c.idx for c in CAT.champions})


def test_json_round_trip():
    ids = [c.id for c in CAT.champions]
    s = DraftState("pro").apply(DraftState("pro").next_step(), SKIPPED)
    s = s.apply(s.next_step(), 7)
    assert DraftState.from_json(s.to_json(ids), {c: i for i, c in enumerate(ids)}) == s


# ------------------------------------------------------------------ planner
def random_state(rng, fmt) -> DraftState:
    s = DraftState(fmt)
    for _ in range(rng.integers(0, len(s.steps))):
        step = s.next_step()
        legal = [c for c in range(len(CAT)) if s.is_legal(step, c)]
        s = s.apply(step, int(rng.choice(legal)))
    return s


@pytest.mark.parametrize("fmt", ["pro", "ranked", "fearless"])
def test_never_suggests_illegal(fmt, synthetic_priors):
    e, rng = engine(synthetic_priors), np.random.default_rng(1)
    for _ in range(12):
        s = random_state(rng, fmt)
        res = e.analyze(s, Prefs(budget_ms=300))
        for line in res.get("lines", []) + res.get("forecast", []):
            idx = CAT.by_id[line["champion"]].idx
            assert s.is_legal(s.next_step(), idx), (fmt, line["name"])


def test_deterministic(synthetic_priors):
    e = engine(synthetic_priors)
    s = DraftState("pro")
    for c in (1, 2, 3, 4, 5, 6):
        s = s.apply(s.next_step(), c)
    a, b = e.analyze(s, Prefs(budget_ms=500)), e.analyze(s, Prefs(budget_ms=500))
    assert [l["champion"] for l in a["lines"]] == [l["champion"] for l in b["lines"]]
    assert [l["win"] for l in a["lines"]] == [l["win"] for l in b["lines"]]


def test_pool_only_and_plan_respected(synthetic_priors):
    e = engine(synthetic_priors)
    s = DraftState("pro")
    for c in range(6):
        s = s.apply(s.next_step(), c + 20)
    pool = {r: [c.idx for c in CAT.champions if r in c.roles][:3] for r in range(5)}
    res = e.analyze(s, Prefs(pools=pool, pool_only=True, plan="poke_siege", budget_ms=500))
    allowed = {c for v in pool.values() for c in v}
    assert res["lines"] and all(CAT.by_id[l["champion"]].idx in allowed for l in res["lines"])
    assert all(l["plan"]["plan"] == "poke_siege" for l in res["lines"])


def test_bans_never_target_own_pool(synthetic_priors):
    e = engine(synthetic_priors)
    s = DraftState("pro")
    top = [int(c) for c in np.argsort(-synthetic_priors.pick_rate.max(1))[:15]]
    res = e.analyze(s, Prefs(pools={0: top}, budget_ms=400))
    assert all(CAT.by_id[l["champion"]].idx not in top for l in res["lines"])


def test_forecast_and_complete(synthetic_priors):
    e = engine(synthetic_priors)
    s = DraftState("pro")
    res = e.analyze(s, Prefs(our_side=RED, budget_ms=400))
    assert res["kind"] == "forecast" and res["forecast"]
    rng = np.random.default_rng(3)
    while s.next_step():
        step = s.next_step()
        s = s.apply(step, int(rng.choice([c for c in range(len(CAT)) if s.is_legal(step, c)])))
    res = e.analyze(s, Prefs())
    assert res["kind"] == "complete" and 0 < res["win_blue"] < 1


def test_latency_seed_mode():
    e = engine(Priors.from_seed(CAT))
    rng, times = np.random.default_rng(5), []
    for _ in range(10):
        s = random_state(rng, "pro")
        t = time.perf_counter()
        e.analyze(s, Prefs(budget_ms=800))
        times.append(time.perf_counter() - t)
    assert np.percentile(times, 95) < 2.0  # hard cap is 2x budget plus formatting


def test_bot_strong_reasoning_changes_with_lane(synthetic_priors):
    """Proposal before/after check: making our bot lane favoured must raise bot-strong-side fit."""
    from draft4.gameplan import GamePlanScorer
    p = Priors.from_seed(CAT)
    us, them = np.array([10, 20, 30, 40, 50]), np.array([11, 21, 31, 41, 51])
    before = GamePlanScorer(CAT, p).score("bot_strong", us, them)["fit"]
    p.win_logit[40, 3] = p.win_logit[50, 4] = 1.0
    after = GamePlanScorer(CAT, p).score("bot_strong", us, them)
    assert after["fit"] > before and "Bot lane projects ahead" in after["met"]


# ------------------------------------------------------------------ live client parsing
def test_lcu_session_to_state():
    a, b, c = CAT.champions[0].id, CAT.champions[1].id, CAT.champions[2].id
    session = {
        "myTeam": [{"cellId": i, "team": 2, "assignedPosition": "utility" if i == 0 else ""} for i in range(5)],
        "theirTeam": [{"cellId": 5 + i, "team": 1} for i in range(5)],
        "actions": [[{"type": "ban", "completed": True, "championId": a, "isAllyAction": True, "actorCellId": 0}],
                    [{"type": "pick", "completed": True, "championId": b, "isAllyAction": True, "actorCellId": 0},
                     {"type": "pick", "completed": False, "championId": c, "isAllyAction": False, "actorCellId": 5}]],
    }
    out = session_to_state(session, CAT)
    assert out["our_side"] == "red"
    st = out["state"]
    assert st["bans"][RED][0] == a and st["picks"][RED][0] == b and st["roles"][RED][0] == 4
    assert st["picks"][BLUE][0] == EMPTY  # hovers are not locks
