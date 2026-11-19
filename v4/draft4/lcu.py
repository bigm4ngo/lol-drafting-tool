"""Read-only League client (LCU) champ-select reader. Optional: manual mode never needs it."""

from __future__ import annotations

import os
from pathlib import Path

import requests
import urllib3

from .catalog import Catalog
from .rules import EMPTY

LOCKFILES = [
    Path(os.environ["LOL_LOCKFILE"]) if os.environ.get("LOL_LOCKFILE") else None,
    Path("C:/Riot Games/League of Legends/lockfile"),
    Path("/Applications/League of Legends.app/Contents/LoL/lockfile"),
]
POSITIONS = {"top": 0, "jungle": 1, "middle": 2, "bottom": 3, "utility": 4}


def fetch_session(timeout: float = 1.5) -> tuple[dict | None, str]:
    lock = next((p for p in LOCKFILES if p and p.exists()), None)
    if lock is None:
        return None, "League client not found (set LOL_LOCKFILE to the client's lockfile path)."
    try:
        _name, _pid, port, password, _proto = lock.read_text().split(":")
    except ValueError:
        return None, "League client lockfile is unreadable."
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # loopback-only; the LCU serves a Riot self-signed cert
    try:
        r = requests.get(f"https://127.0.0.1:{port}/lol-champ-select/v1/session", auth=("riot", password),
                         verify=False, timeout=timeout)
    except requests.RequestException as exc:
        return None, f"League client not reachable: {exc.__class__.__name__}"
    if r.status_code == 404:
        return None, "Not in champion select."
    if not r.ok:
        return None, f"League client returned HTTP {r.status_code}."
    return r.json(), "ok"


def session_to_state(session: dict, catalog: Catalog) -> dict:
    """Convert a champ-select session to the /api/analyze state JSON (Riot champion ids). Locks only;
    hovers are ignored because they are intentions, not commitments."""
    my_team, their_team = session.get("myTeam", []), session.get("theirTeam", [])
    our_side = 0 if (my_team and my_team[0].get("team") == 1) else 1
    cells = {0: [], 1: []}
    for side, team in ((our_side, my_team), (1 - our_side, their_team)):
        cells[side] = sorted(m["cellId"] for m in team)
    locked: dict[int, int] = {}
    bans = {0: [], 1: []}
    for group in session.get("actions", []):
        for act in group:
            cid = int(act.get("championId") or 0)
            if not act.get("completed") or cid <= 0 or catalog.by_id.get(cid) is None:
                continue
            side = our_side if act.get("isAllyAction") else 1 - our_side
            if act.get("type") == "ban":
                bans[side].append(cid)
            elif act.get("type") == "pick":
                locked[int(act["actorCellId"])] = cid
    picks, roles = [[EMPTY] * 5, [EMPTY] * 5], [[EMPTY] * 5, [EMPTY] * 5]
    for side, team in ((our_side, my_team), (1 - our_side, their_team)):
        for member in team:
            slot = cells[side].index(member["cellId"])
            if slot < 5 and member["cellId"] in locked:
                picks[side][slot] = locked[member["cellId"]]
                roles[side][slot] = POSITIONS.get(str(member.get("assignedPosition", "")).lower(), EMPTY)
    return {"state": {"fmt": "ranked", "bans": [(b + [EMPTY] * 5)[:5] for b in (bans[0], bans[1])],
                      "picks": picks, "roles": roles, "unavailable": []},
            "our_side": "blue" if our_side == 0 else "red"}
