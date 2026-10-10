"""Local control server for the dashboard (dashboard/), stdlib HTTP only.

Everything the dashboard starts is an ordinary subprocess job running the
same CLI you would run by hand, with its output kept in a log:

* lobby  play_lobby.py <lobby> | --public KIND ...   (AI vs Human locally, or a real lobby)
* train  train.py --config ... --set run.total_steps=...
* eval   evaluate.py --seats ... --games ... --json ...
* watch  watch_replay.py <file>      (opens the game client on a replay)

Run from ``train/``::

    python control_server.py          # http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
JOBS_DIR = TRAIN_DIR / "control" / "jobs"
DASHBOARD_DIR = TRAIN_DIR.parent / "dashboard"
AGENTS = ("scripted", "random", "noop")
LOG_TAIL = 200
METRIC_POINTS = 400
METRIC_KEYS = (
    "rollout/ep_rew_mean", "rollout/survival_rate", "rollout/final_tiles", "rollout/ep_len_mean",
    "selfplay/learner_beats_pool", "train/approx_kl", "train/explained_variance", "time/steps_per_second",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_under(root: Path, *parts: str) -> Path:
    """``root/parts`` resolved, refusing anything that escapes ``root``."""
    path = root.joinpath(*parts).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"path escapes {root.name}/: {'/'.join(parts)}")
    return path


def read_json(handler: BaseHTTPRequestHandler, limit: int = 1_000_000) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if length > limit:
        raise ValueError("request body too large")
    raw = handler.rfile.read(length) if length else b""
    body = json.loads(raw or b"{}")
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    return body


def model_spec(raw) -> str:
    """A predictor spec from the client: an agent kind or a .pt inside train/."""
    spec = str(raw or "").strip()
    if spec in AGENTS:
        return spec
    # Checkpoints load through pickle, so only files under train/ are allowed.
    path = resolve_under(TRAIN_DIR, spec)
    if path.suffix != ".pt" or not path.is_file():
        raise ValueError(f"not a checkpoint: {spec}")
    return str(path.relative_to(TRAIN_DIR))


def int_in(body: dict, key: str, default: int, lo: int, hi: int) -> int:
    value = int(body.get(key, default))
    if not lo <= value <= hi:
        raise ValueError(f"{key} must be {lo}..{hi}")
    return value


def safe_name(raw, default: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw or "")).strip("_")
    return name[:40] or default


# --------------------------------------------------------------------------
# jobs


@dataclass
class Job:
    id: str
    kind: str
    title: str
    argv: list
    log_path: str
    status: str = "running"
    started: str = field(default_factory=now_iso)
    ended: str = ""
    returncode: int | None = None
    result_path: str = ""
    proc: subprocess.Popen | None = field(default=None, repr=False)


class JobManager:
    def __init__(self, jobs_dir: Path = JOBS_DIR):
        self.jobs_dir = jobs_dir
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._counter = 0

    def start(self, kind: str, title: str, argv: list, result_path: str = "") -> Job:
        with self._lock:
            self._counter += 1
            job_id = f"{kind}-{self._counter:03d}"
        run_dir = self.jobs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        job = Job(job_id, kind, title, [sys.executable, "-u", *argv], str(run_dir / "output.log"),
                  result_path=result_path)
        (run_dir / "command.json").write_text(json.dumps(job.argv, indent=2), encoding="utf-8")
        log = open(job.log_path, "w", encoding="utf-8")
        job.proc = subprocess.Popen(job.argv, cwd=str(TRAIN_DIR), stdout=log, stderr=subprocess.STDOUT)
        with self._lock:
            self._jobs[job_id] = job
            self._evict_locked()
        threading.Thread(target=self._reap, args=(job, log), daemon=True).start()
        return job

    def _reap(self, job: Job, log) -> None:
        try:
            code = job.proc.wait()
        finally:
            log.close()
        with self._lock:
            if job.status == "running":
                job.status = "done" if code == 0 else "error"
            job.returncode = code
            job.ended = now_iso()

    def _evict_locked(self, keep: int = 50) -> None:
        finished = [jid for jid, job in self._jobs.items() if job.status != "running"]
        for jid in finished[: max(0, len(self._jobs) - keep)]:
            del self._jobs[jid]

    def stop(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.proc is not None and job.proc.poll() is None:
            job.proc.terminate()
            try:
                job.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                job.proc.kill()
            with self._lock:
                job.status = "stopped"
                job.ended = now_iso()
        return job

    def remove(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status == "running":
                raise ValueError("job still running, stop it first")
            del self._jobs[job_id]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def summaries(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [self.describe(job) for job in reversed(jobs)]

    @staticmethod
    def describe(job: Job, tail: int = 12) -> dict:
        lines: list[str] = []
        try:
            with open(job.log_path, encoding="utf-8", errors="replace") as handle:
                lines = [line.rstrip("\n") for line in handle.readlines()[-tail:]]
        except OSError:
            pass
        out = {k: getattr(job, k) for k in ("id", "kind", "title", "status", "started", "ended", "returncode")}
        out["log_tail"] = lines
        if job.result_path and Path(job.result_path).is_file():
            try:
                out["result"] = json.loads(Path(job.result_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return out

    # ----- the four job kinds ------------------------------------------------

    def start_lobby(self, body: dict) -> Job:
        from openfront.lobby import PUBLIC_KINDS, resolve_lobby

        model = model_spec(body.get("model", "scripted"))
        name = re.sub(r"[^A-Za-z0-9 _-]+", "", str(body.get("name") or "OpenFrontAI")).strip()[:24] or "OpenFrontAI"
        profile = safe_name(body.get("profile"), "default")
        public = body.get("public")
        if public:
            # The next public openfront.io lobby of this kind, picked at join time.
            if public not in (*PUBLIC_KINDS, "any"):
                raise ValueError(f"public must be one of {', '.join((*PUBLIC_KINDS, 'any'))}")
            target, where = ["--public", public, "--host", "openfront"], f"next public {public} lobby"
        else:
            host = str(body.get("host", "local"))
            if host not in ("local", "openfront"):
                raise ValueError("host must be local or openfront")
            base, lobby = resolve_lobby(str(body.get("lobby", "")), host)
            target, where = [lobby, "--host", base], lobby
        argv = ["play_lobby.py", *target, "--model", model, "--name", name, "--profile", profile]
        if body.get("deterministic"):
            argv.append("--deterministic")
        return self.start("lobby", f"{name} in {where} ({Path(model).stem})", argv)

    def profile_in_use(self, profile: str) -> bool:
        with self._lock:
            return any(j.kind == "lobby" and j.status == "running" and
                       f"--profile\x00{profile}" in "\x00".join(j.argv) for j in self._jobs.values())

    def start_train(self, body: dict) -> Job:
        config = resolve_under(TRAIN_DIR, "configs", str(body.get("config", "selfplay.yaml")))
        if config.suffix != ".yaml" or not config.is_file():
            raise ValueError(f"no such config: {config.name}")
        name = safe_name(body.get("run_name"), config.stem)
        steps = int_in(body, "total_steps", 100_000, 512, 100_000_000)
        argv = ["train.py", "--config", str(config.relative_to(TRAIN_DIR)),
                "--set", f"run.name={name}", "--set", f"run.total_steps={steps}"]
        if body.get("workers"):
            argv += ["--set", f"env.workers={int_in(body, 'workers', 8, 1, 64)}"]
        if body.get("resume"):
            argv.append("--resume")
        return self.start("train", f"{name}: {steps:,} steps from {config.name}", argv)

    def start_eval(self, body: dict) -> Job:
        seats = [model_spec(s) for s in body.get("seats", []) if str(s).strip()]
        if not 1 <= len(seats) <= 8:
            raise ValueError("pick 1..8 seats")
        games = int_in(body, "games", 8, 1, 500)
        bots = int_in(body, "bots", 20, 0, 200)
        minutes = int_in(body, "minutes", 60, 1, 240)
        with self._lock:
            n = self._counter + 1
        result = self.jobs_dir / f"eval-{n:03d}" / "result.json"
        argv = ["evaluate.py", "--seats", *seats, "--games", str(games), "--bots", str(bots),
                "--tick-cap", str(minutes * 600), "--json", str(result)]
        if body.get("record"):
            argv += ["--replay-dir", f"replays/eval-{n:03d}"]
        title = " vs ".join(Path(s).stem for s in seats) + f", {games} games"
        return self.start("eval", title, argv, result_path=str(result))

    def start_watch(self, body: dict) -> Job:
        replay = resolve_under(TRAIN_DIR / "replays", str(body.get("replay", "")))
        if replay.suffix != ".json" or not replay.is_file():
            raise ValueError("no such replay")
        rel = str(replay.relative_to(TRAIN_DIR))
        return self.start("watch", f"replay {replay.name}", ["watch_replay.py", rel])


JOBS = JobManager()
PUBLIC_LOCK = threading.Lock()


def public_lobbies() -> dict:
    """openfront.io's public lobbies, read by ``play_lobby.py --list`` in a
    short-lived visible window. Uses the joining profile (so one bot check
    covers both) unless a lobby job has it open."""
    if not PUBLIC_LOCK.acquire(blocking=False):
        raise ValueError("already looking for lobbies; finish the check in the browser window")
    try:
        profile = "browse" if JOBS.profile_in_use("default") else "default"
        try:
            proc = subprocess.run([sys.executable, "-u", "play_lobby.py", "--list", "--profile", profile],
                                  cwd=str(TRAIN_DIR), capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            raise ValueError("openfront.io did not load in time (was the Cloudflare check answered?)") from None
        if proc.returncode != 0:
            last = (proc.stderr.strip().splitlines() or ["lobby listing failed"])[-1]
            raise ValueError(last)
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        out["fetched"] = now_iso()
        return out
    finally:
        PUBLIC_LOCK.release()


# --------------------------------------------------------------------------
# listings and metrics


def _stat(path: Path, root: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "size": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }


_LAYOUT_CACHE: dict[tuple[str, float], bool] = {}


def _compatible(path: Path) -> bool:
    """Whether a checkpoint matches the current observation layout (cached by mtime)."""
    key = (str(path), path.stat().st_mtime)
    if key not in _LAYOUT_CACHE:
        try:
            import torch

            from openfront import layout as L

            payload = torch.load(str(path), map_location="cpu", weights_only=False)
            L.check_layout(payload["layout"])
            _LAYOUT_CACHE[key] = True
        except Exception:  # noqa: BLE001, any unreadable or older checkpoint is hidden
            _LAYOUT_CACHE[key] = False
    return _LAYOUT_CACHE[key]


def list_checkpoints() -> list[dict]:
    """Checkpoints the current code can load: older observation layouts are hidden."""
    root = TRAIN_DIR / "checkpoints"
    if not root.is_dir():
        return []
    found = [p for p in root.glob("**/*.pt") if "pool" not in p.parts and _compatible(p)]
    return [_stat(p, TRAIN_DIR) for p in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)]


def list_replays() -> list[dict]:
    root = TRAIN_DIR / "replays"
    if not root.is_dir():
        return []
    found = sorted(root.glob("**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [_stat(p, root) for p in found]


def list_runs() -> list[dict]:
    root = TRAIN_DIR / "runs"
    if not root.is_dir():
        return []
    runs = [p.parent for p in root.glob("*/metrics.jsonl")]
    return [{"name": p.name, **_stat(p / "metrics.jsonl", TRAIN_DIR)}
            for p in sorted(runs, key=lambda p: (p / "metrics.jsonl").stat().st_mtime, reverse=True)]


def list_configs() -> list[str]:
    return sorted(p.name for p in (TRAIN_DIR / "configs").glob("*.yaml"))


def read_metrics(run: str) -> dict:
    path = resolve_under(TRAIN_DIR, "runs", safe_name(run, "x"), "metrics.jsonl")
    if not path.is_file():
        return {"series": {}, "note": "no metrics.jsonl for this run"}
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    stride = max(1, len(rows) // METRIC_POINTS)
    series = {}
    scenario_keys = sorted({k for r in rows for k in r if k.startswith("scenario/") and k.endswith("/survival_rate")})
    for key in (*METRIC_KEYS, *scenario_keys):
        points = [{"step": r["step"], "value": r[key]} for r in rows[::stride]
                  if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
        if points:
            series[key] = points
    last = rows[-1] if rows else {}
    intents = {k[len("intents/"):]: v for k, v in last.items() if k.startswith("intents/")}
    latest = {k: v for k, v in last.items() if isinstance(v, (int, float)) and v == v and not k.startswith("intents/")}
    return {"series": series, "intents": intents, "last": latest, "last_step": last.get("step"), "updates": len(rows)}


# --------------------------------------------------------------------------
# HTTP


CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json"}


class Handler(BaseHTTPRequestHandler):
    server_version = "OpenFrontControl/2"

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path, query = url.path, urllib.parse.parse_qs(url.query)
        self._dispatch(lambda: self._get(path, query))

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        self._dispatch(lambda: self._post(path, read_json(self)))

    def do_DELETE(self):
        path = urllib.parse.urlparse(self.path).path
        self._dispatch(lambda: self._delete(path))

    def _dispatch(self, fn):
        try:
            fn()
        except KeyError as exc:
            self.send_error_json(404, f"unknown: {exc}")
        except (ValueError, FileNotFoundError) as exc:
            self.send_error_json(400, str(exc))
        except Exception as exc:  # noqa: BLE001, the dashboard always gets JSON
            self.send_error_json(500, f"{type(exc).__name__}: {exc}")

    def _get(self, path: str, query: dict) -> None:
        if path.startswith("/replays/"):
            return self.serve_file(resolve_under(TRAIN_DIR / "replays", urllib.parse.unquote(path[len("/replays/"):])),
                                   cors=True)
        if not path.startswith("/api/"):
            name = "index.html" if path in ("", "/") else path.lstrip("/")
            return self.serve_file(resolve_under(DASHBOARD_DIR, name))
        if path == "/api/state":
            return self.send_json({"jobs": JOBS.summaries()})
        if path == "/api/catalog":
            return self.send_json({"checkpoints": list_checkpoints(), "agents": list(AGENTS),
                                   "configs": list_configs(), "runs": list_runs()})
        if path == "/api/replays":
            return self.send_json({"replays": list_replays()}, cors=True)
        if path == "/api/runstats":
            return self.send_json(read_metrics(query.get("run", [""])[0]))
        if path.startswith("/api/jobs/") and path.endswith("/logs"):
            job = JOBS.get(path[len("/api/jobs/"):-len("/logs")])
            if job is None:
                raise KeyError("job")
            return self.send_json(JOBS.describe(job, tail=LOG_TAIL))
        raise KeyError(path)

    def _post(self, path: str, body: dict) -> None:
        if path == "/api/public-lobbies":
            return self.send_json(public_lobbies())
        starters = {"/api/jobs/lobby": JOBS.start_lobby, "/api/jobs/train": JOBS.start_train,
                    "/api/jobs/eval": JOBS.start_eval, "/api/jobs/watch": JOBS.start_watch}
        if path not in starters:
            raise KeyError(path)
        self.send_json(JOBS.describe(starters[path](body)), status=201)

    def _delete(self, path: str) -> None:
        if not path.startswith("/api/jobs/"):
            raise KeyError(path)
        job_id = path[len("/api/jobs/"):]
        job = JOBS.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == "running":
            return self.send_json(JOBS.describe(JOBS.stop(job_id)))
        JOBS.remove(job_id)
        self.send_json({"deleted": job_id})

    def serve_file(self, path: Path, cors: bool = False) -> None:
        if not path.is_file():
            raise FileNotFoundError(path.name)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict, status: int = 200, cors: bool = False) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status=status)


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.host, args.port)
    print(f"dashboard on http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in JOBS.summaries():
            if job["status"] == "running":
                JOBS.stop(job["id"])


if __name__ == "__main__":
    main()
