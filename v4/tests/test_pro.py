from collections import Counter

import pytest

from draft4.catalog import load_catalog
from draft4.rules import BLUE, RED, DraftState
from train.pro_import import parse_rows

CAT = load_catalog()


def test_red_first_pick_order():
    s = DraftState("pro", first=RED)
    code = "".join("BR"[x.side] + x.kind[0] for x in s.steps[:12])
    assert code == "RbBbRbBbRbBb" "RpBpBpRpRpBp"
    with pytest.raises(ValueError):
        DraftState("ranked", first=RED)
    ids = [c.id for c in CAT.champions]
    assert DraftState.from_json(s.to_json(ids), {c: i for i, c in enumerate(ids)}).first == RED


def oe_game(first_pick_blue: str, red_picks_order=None, red_ban1="Orianna"):
    blue = ["Sion", "Wukong", "Ahri", "Yunara", "Braum"]
    red = ["Mordekaiser", "Ambessa", "Taliyah", "Varus", "Nautilus"]
    pos = ["top", "jng", "mid", "bot", "sup"]
    rows = [{"side": s, "position": p, "champion": c} for s, team in (("Blue", blue), ("Red", red)) for p, c in zip(pos, team)]
    order = {"Blue": ["Wukong", "Sion", "Ahri", "Yunara", "Braum"], "Red": red_picks_order or red}
    for side, bans in (("Blue", ["Qiyana", "Neeko", "Xin Zhao", "", "Alistar"]), ("Red", [red_ban1, "Zed", "Lux", "Jinx", "Ezreal"])):
        t = {"side": side, "position": "team", "gameid": "13057-13057_game_1", "firstPick": first_pick_blue if side == "Blue" else str(1 - int(first_pick_blue)),
             "league": "LCK", "date": "2026-01-01 09:14:49", "game": "1", "patch": "16.1", "teamname": side, "result": "1" if side == "Blue" else "0"}
        t.update({f"ban{i + 1}": b for i, b in enumerate(bans)})
        t.update({f"pick{i + 1}": c for i, c in enumerate(order[side])})
        rows.append(t)
    return rows


def test_parse_oe_game():
    stats = Counter()
    g = parse_rows(oe_game("0"), CAT, stats)
    assert g["first"] == RED and g["series"] == "13057-13057" and g["blue_win"] == 1
    assert CAT.champions[g["picks"][BLUE][0]].name == "Wukong" and g["roles"][BLUE][0] == 1  # pick order kept, role from player row
    assert g["bans"][BLUE][3] < 0  # missing ban -> skipped, not dropped
    assert parse_rows(oe_game("1", ["Zed", "Ambessa", "Taliyah", "Varus", "Nautilus"]), CAT, stats) is None
    assert stats["pick_order_mismatch"] == 1
    assert parse_rows(oe_game("1", red_ban1="Ahri"), CAT, stats) is None  # banned and picked: data error
    assert stats["duplicate_champion"] == 1
