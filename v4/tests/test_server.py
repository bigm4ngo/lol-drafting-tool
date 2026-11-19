import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from draft4.server import App, make_handler


@pytest.fixture(scope="module")
def port(tmp_path_factory):
    app = App(tmp_path_factory.mktemp("models"))  # empty models dir -> seed mode
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    p = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(app, {f"127.0.0.1:{p}"})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield p
    srv.shutdown()


def request(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    return r.status, r.read()


def test_analyze_ok(port):
    status, body = request(port, "POST", "/api/analyze", json.dumps({"state": {"fmt": "pro"}, "prefs": {"budget_ms": 200}}),
                           {"Content-Type": "application/json"})
    assert status == 200 and json.loads(body)["kind"] == "recommend"


def test_rebinding_host_rejected(port):
    assert request(port, "GET", "/api/live", headers={"Host": "evil.example:8765"})[0] == 403


def test_non_json_post_rejected(port):
    assert request(port, "POST", "/api/reload", "{}", {"Content-Type": "text/plain"})[0] == 415


@pytest.mark.parametrize("length", ["-1", "abc", "999999999"])
def test_bad_content_length(port, length):
    assert request(port, "POST", "/api/analyze", None, {"Content-Type": "application/json", "Content-Length": length})[0] in (400, 413)
