"""Control server: routes, input validation, and job plumbing. Fast and
bounded: no game, browser or trainer starts here (job launches are
captured), and every wait has a deadline."""

import json
import shutil
import threading
import time
import urllib.error
import urllib.request

import pytest

import control_server
from control_server import TRAIN_DIR, JobManager


@pytest.fixture()
def base_url(monkeypatch):
    launched = []

    def fake_start(kind, title, argv, result_path=""):
        launched.append({"kind": kind, "title": title, "argv": argv})
        return control_server.Job(f"{kind}-x", kind, title, argv, "missing.log")

    monkeypatch.setattr(control_server.JOBS, "start", fake_start)
    server = control_server.make_server(port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", launched
    server.shutdown()
    server.server_close()


def _request(method, url, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, json.loads(res.read().decode("utf-8")), dict(res.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8")), dict(exc.headers)


def test_serves_dashboard_and_catalog(base_url):
    url, _ = base_url
    with urllib.request.urlopen(f"{url}/", timeout=10) as res:
        assert b"Play in a lobby" in res.read()
    status, catalog, _ = _request("GET", f"{url}/api/catalog")
    assert status == 200
    assert "scripted" in catalog["agents"]
    assert "selfplay.yaml" in catalog["configs"]
    assert all(c["path"].endswith(".pt") and "/pool/" not in c["path"] for c in catalog["checkpoints"])
    status, replays, headers = _request("GET", f"{url}/api/replays")
    assert status == 200 and isinstance(replays["replays"], list)
    assert headers.get("Access-Control-Allow-Origin") == "*"


def test_dashboard_files_cannot_escape(base_url):
    url, _ = base_url
    status, body, _ = _request("GET", f"{url}/..%2Ftrain%2Fcontrol_server.py")
    assert status == 400


def test_lobby_job_runs_play_lobby(base_url):
    url, launched = base_url
    status, _, _ = _request("POST", f"{url}/api/jobs/lobby",
                            {"lobby": "https://openfront.io/w2/game/AbCd1234", "host": "local", "model": "scripted",
                             "name": "Bot <script>"})
    assert status == 201
    argv = launched[-1]["argv"]
    assert argv[:2] == ["play_lobby.py", "AbCd1234"]
    assert argv[argv.index("--host") + 1] == "https://openfront.io"  # the link's origin wins
    assert argv[argv.index("--name") + 1] == "Bot script"


def test_delete_replays(base_url):
    url, _ = base_url
    folder = TRAIN_DIR / "replays" / "_delete_test" / "sub"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        for name in ("a.json", "b.json"):
            (folder / name).write_text("{}", encoding="utf-8")
        # Anything outside replays/ or not a replay is refused, and nothing goes.
        for bad in (["../train.py"], ["_delete_test/sub/a.json", "../control_server.py"], ["_delete_test/sub"], []):
            status, payload, _ = _request("POST", f"{url}/api/replays/delete", {"paths": bad})
            assert status == 400 and payload["error"]
        assert (folder / "a.json").is_file()
        status, payload, _ = _request("POST", f"{url}/api/replays/delete",
                                      {"paths": ["_delete_test/sub/a.json", "_delete_test/sub/b.json"]})
        assert status == 200 and payload["deleted"] == 2
        assert not (TRAIN_DIR / "replays" / "_delete_test").exists()  # emptied folders go too
    finally:
        shutil.rmtree(TRAIN_DIR / "replays" / "_delete_test", ignore_errors=True)


def test_quick_match_records_one_game(base_url):
    url, launched = base_url
    status, _, _ = _request("POST", f"{url}/api/jobs/match", {"model": "scripted", "mode": "selfplay", "minutes": 5})
    assert status == 201
    argv = launched[-1]["argv"]
    assert argv[argv.index("--seats") + 1: argv.index("--games")] == ["scripted"] * 4
    assert argv[argv.index("--games") + 1] == "1" and argv[argv.index("--tick-cap") + 1] == "3000"
    assert argv[argv.index("--replay-dir") + 1].startswith("replays/matches/")
    assert argv[argv.index("--nations") + 1] == "disabled"  # nations only when asked for
    status, _, _ = _request("POST", f"{url}/api/jobs/match", {"model": "scripted", "mode": "bots", "nations": True,
                                                              "minutes": ""})
    argv = launched[-1]["argv"]
    assert argv[argv.index("--seats") + 1: argv.index("--games")] == ["scripted"]
    assert argv[argv.index("--nations") + 1] == "default"
    assert argv[argv.index("--tick-cap") + 1] == "12000"  # blank minutes: the training length
    status, _, _ = _request("POST", f"{url}/api/jobs/match", {"model": "scripted", "mode": "chaos"})
    assert status == 400


def test_match_env_follows_the_training_stage():
    from openfront.config import match_env

    cfg = {"env": {"map": "World", "bots": 20, "tick_cap": 12000, "close_spawn": 0.3, "seats": 4},
           "curriculum": [{"name": "warmup", "bots": 10, "nations": "disabled"}, {"name": "full", "bots": 20}]}
    assert match_env(cfg, 0, "selfplay") == {"map": "World", "bots": 10, "tick_cap": 12000, "close_spawn": 0.3, "seats": 4}
    assert match_env(cfg, 1, "selfplay")["bots"] == 20
    assert match_env(cfg, 0, "bots")["seats"] == 1
    mixed = {"env": {"map": "World", "bots": 20, "tick_cap": 12000, "close_spawn": 0.0, "seats": 4},
             "scenarios": [{"name": "vs_bots", "seats": 1}, {"name": "crowded", "seats": 6, "bots": 30, "weight": 1},
                           {"name": "selfplay_ffa", "seats": 4, "weight": 3}]}
    assert match_env(mixed, 0, "selfplay")["seats"] == 4


def test_public_lobby_job(base_url):
    url, launched = base_url
    status, _, _ = _request("POST", f"{url}/api/jobs/lobby", {"public": "ffa", "model": "scripted"})
    assert status == 201
    argv = launched[-1]["argv"]
    assert argv[1:5] == ["--public", "ffa", "--host", "openfront"]
    status, _, _ = _request("POST", f"{url}/api/jobs/lobby", {"public": "ranked", "model": "scripted"})
    assert status == 400


@pytest.mark.parametrize("body", [
    {"lobby": "nope", "model": "scripted"},
    {"lobby": "AbCd1234", "model": "../../secrets.pt"},
    {"lobby": "AbCd1234", "model": "configs/selfplay.yaml"},
    {"lobby": "AbCd1234", "model": "scripted", "host": "evil.example"},
])
def test_lobby_job_rejects_bad_input(base_url, body):
    url, launched = base_url
    status, payload, _ = _request("POST", f"{url}/api/jobs/lobby", body)
    assert status == 400 and payload["error"]
    assert not launched


def test_train_job_defaults_to_100k_steps(base_url):
    url, launched = base_url
    status, _, _ = _request("POST", f"{url}/api/jobs/train", {"config": "smoke.yaml", "run_name": "ui test!"})
    assert status == 201
    argv = launched[-1]["argv"]
    assert argv[:3] == ["train.py", "--config", "configs\\smoke.yaml"] or argv[:3] == ["train.py", "--config", "configs/smoke.yaml"]
    assert "run.total_steps=100000" in argv and "run.name=ui_test" in argv
    status, _, _ = _request("POST", f"{url}/api/jobs/train", {"config": "../train.py"})
    assert status == 400


def test_eval_job_passes_seats(base_url):
    url, launched = base_url
    status, _, _ = _request("POST", f"{url}/api/jobs/eval", {"seats": ["scripted", "", "random"], "games": 3, "record": True})
    assert status == 201
    argv = launched[-1]["argv"]
    assert argv[argv.index("--seats") + 1 : argv.index("--seats") + 3] == ["scripted", "random"]
    assert "--replay-dir" in argv and "--json" in argv
    status, _, _ = _request("POST", f"{url}/api/jobs/eval", {"seats": []})
    assert status == 400


def test_metrics_read_from_metrics_jsonl(base_url):
    url, _ = base_url
    run = TRAIN_DIR / "runs" / "zz_control_test"
    run.mkdir(parents=True, exist_ok=True)
    try:
        rows = [{"step": i * 100, "update": i, "rollout/survival_rate": i / 10, "intents/attack": 0.3} for i in range(5)]
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        status, data, _ = _request("GET", f"{url}/api/runstats?run=zz_control_test")
        assert status == 200
        assert [p["value"] for p in data["series"]["rollout/survival_rate"]] == [0, 0.1, 0.2, 0.3, 0.4]
        assert data["intents"] == {"attack": 0.3} and data["last_step"] == 400
    finally:
        shutil.rmtree(run)


def test_job_manager_runs_and_reaps_a_process(tmp_path):
    jobs = JobManager(tmp_path)
    job = jobs.start("train", "echo", ["-c", "print('hello from the job')"])
    deadline = time.monotonic() + 30
    while jobs.get(job.id).status == "running" and time.monotonic() < deadline:
        time.sleep(0.1)
    described = jobs.describe(jobs.get(job.id))
    assert described["status"] == "done" and described["returncode"] == 0
    assert "hello from the job" in described["log_tail"]
    jobs.remove(job.id)
    assert jobs.get(job.id) is None
