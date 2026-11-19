"""Local HTTP server: JSON engine API + the web UI. Standard library only.

    python -m draft4.server                 # http://127.0.0.1:8765
    python -m draft4.server --host 0.0.0.0  # share with coaches on the LAN (no auth: trusted networks only)
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .catalog import ROLES, load_catalog
from .gameplan import PLANS
from .lcu import fetch_session, session_to_state
from .models import MODELS_DIR, load_models
from .planner import Engine, Prefs
from .rules import FORMATS, SIDES, DraftState

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
MAX_BODY = 64 * 1024
LOG = logging.getLogger("draft4.server")


def gpu_status() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda": False}
    cuda = torch.cuda.is_available()
    return {"torch": torch.__version__, "cuda": cuda, "device": torch.cuda.get_device_name(0) if cuda else None}


class App:
    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        catalog = load_catalog()
        models = load_models(catalog, self.models_dir)
        with self.lock:
            self.catalog, self.status = catalog, models["status"]
            self.engine = Engine(catalog, **models)
            self.id_to_idx = {c.id: c.idx for c in catalog.champions}

    def meta(self) -> dict:
        cat, p = self.catalog, self.engine.p
        return {
            "roles": ROLES,
            "champions": [{"id": c.id, "key": c.key, "name": c.name, "image": c.image, "roles": [ROLES[r] for r in c.roles],
                           "traits": sorted(c.traits), "damage": c.damage,
                           "role_rate": [round(float(x), 3) for x in p.role_rate[c.idx]]} for c in cat.champions],
            "formats": {k: {"label": v["label"], "fearless": v["fearless"],
                            **{key: [{"side": SIDES[s.side], "kind": s.kind, "slot": s.slot} for s in v[key]]
                               for key in ("steps", "steps_red_first")}}
                        for k, v in FORMATS.items()},
            "plans": {k: {"label": v["label"], "win": v["win"], "requirements": [r[0] for r in v["req"]],
                          "risks": [r[0] for r in v["risks"]]} for k, v in PLANS.items()},
            "status": {**self.status, "versions": self.engine.versions(), "gpu": gpu_status()},
        }

    def analyze(self, body: dict) -> dict:
        state = DraftState.from_json(body.get("state") or {}, self.id_to_idx)
        prefs_in = body.get("prefs") or {}
        side = prefs_in.get("our_side")
        if side not in (None, "", "auto", "blue", "red"):
            raise ValueError("our_side must be blue, red or auto")
        plan = prefs_in.get("plan") or None
        if plan is not None and plan not in PLANS:
            raise ValueError(f"unknown plan {plan!r}")
        pools = {}
        for role, ids in (prefs_in.get("pools") or {}).items():
            if role not in ROLES:
                raise ValueError(f"unknown role {role!r}")
            pools[ROLES.index(role)] = [self.id_to_idx[int(i)] for i in ids if int(i) in self.id_to_idx]
        prefs = Prefs(our_side=SIDES.index(side) if side in SIDES else None, plan=plan, pools=pools,
                      pool_only=bool(prefs_in.get("pool_only")),
                      budget_ms=max(200, min(10000, int(prefs_in.get("budget_ms", 1500)))),
                      n_lines=max(1, min(6, int(prefs_in.get("n_lines", 3)))))
        return {"rev": body.get("rev"), **self.engine.analyze(state, prefs)}

    def live(self) -> dict:
        session, message = fetch_session()
        if session is None:
            return {"available": False, "message": message}
        return {"available": True, **session_to_state(session, self.catalog)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:
            LOG.debug(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/api/meta":
                return self._json(app.meta())
            if path == "/api/live":
                return self._json(app.live())
            rel = "index.html" if path in ("/", "") else path.lstrip("/").removeprefix("ui/")
            target = (UI_DIR / rel).resolve()
            if not target.is_file() or UI_DIR not in target.parents:
                return self._json({"error": "not found"}, 404)
            self._send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._json({"error": "request too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/analyze":
                    return self._json(app.analyze(body))
                if self.path == "/api/reload":
                    app.reload()
                    return self._json({"ok": True, "status": app.status})
                return self._json({"error": "not found"}, 404)
            except (ValueError, KeyError, TypeError) as exc:
                return self._json({"error": str(exc)}, 400)
            except Exception as exc:  # never leave the UI hanging
                LOG.exception("request failed")
                return self._json({"error": f"internal error: {exc}"}, 500)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--models", type=Path, default=MODELS_DIR, help="folder holding active/ checkpoints")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = App(a.models)
    server = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    LOG.info("Draft Lab v4 on http://%s:%d  (policy=%s, value=%s)", a.host, a.port, app.status["policy"], app.status["value"])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
