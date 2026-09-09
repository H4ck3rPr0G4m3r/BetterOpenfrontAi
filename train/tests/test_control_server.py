"""Tests for the side-quest control server. Fast and bounded by design.

No node worker boots here: the live path runs against the in-process
fake worker (predictor "simulated"), and every wait carries a deadline
so a regression fails in seconds instead of hanging the suite. That
rule is load-bearing: this server once wedged a terminal with an
unbounded poll loop, and these tests must never do the same.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import control_server
from control_server import Handler


def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture()
def base_url():
    server = _serve()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    try:
        control_server.LIVE.stop()
    except Exception:
        pass
    server.shutdown()
    server.server_close()


def _request(method, url, body=None, timeout=10):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _wait_until(predicate, deadline_s=60):
    deadline = time.monotonic() + deadline_s
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.5)
    raise AssertionError("deadline exceeded waiting for condition")


def test_state_lists_valid_presets(base_url):
    status, data = _request("GET", f"{base_url}/api/state")
    assert status == 200
    assert isinstance(data["jobs"], list)
    names = {p.get("name") for p in data["presets"]}
    assert "no-factories" in names
    assert "berserker" in names
    assert "turtle" in names
    assert all("error" not in p for p in data["presets"])


def test_preset_validation_and_round_trip(base_url):
    status, _ = _request("POST", f"{base_url}/api/presets", {"disabled_intents": [0]})
    assert status == 400
    body = {"name": "srv-test-roundtrip", "aggression": 70}
    status, saved = _request("POST", f"{base_url}/api/presets", body)
    assert status == 201
    assert saved["name"] == "srv-test-roundtrip"
    try:
        status, data = _request("GET", f"{base_url}/api/presets")
        assert any(p.get("name") == "srv-test-roundtrip" for p in data["presets"])
    finally:
        status, _ = _request("DELETE", f"{base_url}/api/presets/srv-test-roundtrip")
        assert status == 200


def test_unknown_job_is_404(base_url):
    status, _ = _request("GET", f"{base_url}/api/jobs/nope-000")
    assert status == 404
    status, _ = _request("DELETE", f"{base_url}/api/jobs/nope-000")
    assert status == 404


def test_checkpoints_and_metrics_empty_paths(base_url):
    status, data = _request("GET", f"{base_url}/api/checkpoints?dir=control")
    assert status == 200
    assert data["checkpoints"] == []
    status, data = _request("GET", f"{base_url}/api/metrics?tb_dir=control&run=x")
    assert status == 200
    assert data["note"] == "no event files found"


def test_dashboard_index_served(base_url):
    req = urllib.request.Request(f"{base_url}/", method="GET")
    with urllib.request.urlopen(req, timeout=10) as res:
        assert res.status == 200
        assert "AI control" in res.read().decode("utf-8")


def test_live_rejects_bad_predictor(base_url):
    status, _ = _request("POST", f"{base_url}/api/live/start", {"predictor": "nope"})
    assert status == 400


def test_live_simulated_streams_steps(base_url):
    status, state = _request(
        "POST",
        f"{base_url}/api/live/start",
        {"predictor": "simulated", "behavior": {"name": "t", "aggression": 90}},
    )
    assert status == 200
    assert state["running"] is True
    try:
        status, _ = _request(
            "POST",
            f"{base_url}/api/live/start",
            {"predictor": "simulated"},
        )
        assert status == 400

        def stepped():
            _, st = _request("GET", f"{base_url}/api/live/state")
            return st if st["step_count"] >= 5 else None

        grown = _wait_until(stepped)
        latest = grown["latest"]
        assert latest["action_text"]
        assert latest["masks"]["intent_type"]["total"] == 14
        assert grown["intent_counts"]
    finally:
        status, stopped = _request("POST", f"{base_url}/api/live/stop", {})
        assert status == 200
        assert stopped["running"] is False
