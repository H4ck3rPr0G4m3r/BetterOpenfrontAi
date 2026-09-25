"""Local control server for the AI control UI (side quest).

Stdlib only. It spawns the existing ``train_ppo.py`` / ``eval.py`` scripts
as subprocesses with a generated config, tails their logs over SSE, reads
TensorBoard scalars for charts, manages behavior presets, and runs a live
1x1 bot session whose every step streams to the dashboard.

Run from ``train/``::

    python control_server.py --port 8765

Then open http://127.0.0.1:8765/ in a browser.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from action_filter import (
    BehaviorPreset,
    apply_aggression_batch,
    describe_action,
    mask_stats,
)

TRAIN_DIR = Path(__file__).resolve().parent
CONTROL_DIR = TRAIN_DIR / "control"
RUNS_DIR = CONTROL_DIR / "runs"
REPLAYS_DIR = TRAIN_DIR / "replays"
PRESETS_DIR = CONTROL_DIR / "presets"
DASHBOARD_DIR = TRAIN_DIR.parent / "dashboard"

MAX_LOG_TAIL = 200
LIVE_RING = 200
METRIC_POINTS = 400


# --------------------------------------------------------------------------
# small helpers


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_json(handler: BaseHTTPRequestHandler, limit: int = 1_000_000):
    length = int(handler.headers.get("Content-Length", 0))
    if length <= 0 or length > limit:
        raise ValueError("missing or oversized JSON body")
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def resolve_under(root: Path, *parts: str) -> Path:
    """Join untrusted path parts onto root, rejecting escapes.

    A string-prefix check is not enough: siblings such as
    ``train/configs-evil`` share the ``train/configs`` prefix, and an
    absolute part discards the root entirely. Resolving first and then
    requiring containment is exact.
    """
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes {root}")
    return target


# Client-supplied run config may only touch these keys. Paths in
# particular (maps_dir, checkpoint_dir, tensorboard_log) stay
# server-side: letting them point anywhere would drive writes and
# checkpoint loads outside train/.
OVERRIDE_ALLOW: dict[str, set[str] | None] = {
    "env": {
        "map",
        "bots",
        "ticks_per_step",
        "tick_cap",
        "num_workers",
        "envs_per_worker",
        "strict_illegal_actions",
        "socket_timeout",
        "seed",
    },
    "ppo": None,
    "policy": None,
    "run": {"total_timesteps", "checkpoint_freq", "run_name", "device"},
}


def sanitize_override(raw: dict) -> dict:
    """Keep only whitelisted config-override keys (see OVERRIDE_ALLOW)."""
    if not isinstance(raw, dict):
        raise ValueError("config_override must be a mapping")
    unknown_sections = set(raw) - set(OVERRIDE_ALLOW)
    if unknown_sections:
        raise ValueError(f"override sections not allowed: {sorted(unknown_sections)}")
    out: dict = {}
    for section, allowed in OVERRIDE_ALLOW.items():
        part = raw.get(section, {}) or {}
        if not isinstance(part, dict):
            raise ValueError(f"override section {section} must be a mapping")
        if allowed is None:
            out[section] = copy.deepcopy(part)
        else:
            unknown = set(part) - allowed
            if unknown:
                raise ValueError(
                    f"override keys not allowed in {section}: {sorted(unknown)}"
                )
            out[section] = copy.deepcopy(part)
    return out


# --------------------------------------------------------------------------
# jobs: train / eval subprocesses


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"
    config_path: str = ""
    log_path: str = ""
    started: str = ""
    ended: str = ""
    returncode: int | None = None
    error: str = ""
    proc: subprocess.Popen | None = field(default=None, repr=False)


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._counter = 0

    def start_train(self, body: dict) -> Job:
        base_name = str(body.get("base", "default.yaml"))
        base_path = resolve_under(TRAIN_DIR, "configs", base_name)
        with open(base_path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        preset = BehaviorPreset.from_dict(body.get("behavior", {"name": "custom"}))
        behavior_cfg = {
            "aggression": preset.aggression,
            "min_attack_ratio": preset.min_attack_ratio,
            "disabled_intents": list(preset.disabled_intents),
            "disabled_builds": list(preset.disabled_builds),
            "reward_weights": list(preset.reward_weights)
            if preset.reward_weights
            else None,
            "troop_loss_scale": preset.troop_loss_scale,
        }
        cfg = deep_merge(cfg, sanitize_override(body.get("config_override", {}) or {}))
        cfg["behavior"] = behavior_cfg
        run = cfg.setdefault("run", {})
        if body.get("run_name"):
            run["run_name"] = str(body["run_name"])
        if body.get("total_timesteps") is not None:
            run["total_timesteps"] = int(body["total_timesteps"])
        return self._spawn("train", [sys.executable, "train_ppo.py"], cfg)

    def start_eval(self, body: dict) -> Job:
        model = str(body.get("model", ""))
        if not model:
            raise ValueError("eval needs a model checkpoint path")
        # Checkpoints load through pickle, so the path stays contained in
        # train/ rather than trusting client input.
        model_path = str(resolve_under(TRAIN_DIR, model))
        base_name = str(body.get("base", "default.yaml"))
        base_path = resolve_under(TRAIN_DIR, "configs", base_name)
        with open(base_path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        preset = BehaviorPreset.from_dict(body.get("behavior", {"name": "custom"}))
        cfg["behavior"] = {
            "aggression": preset.aggression,
            "min_attack_ratio": preset.min_attack_ratio,
            "disabled_intents": list(preset.disabled_intents),
            "disabled_builds": list(preset.disabled_builds),
            "reward_weights": list(preset.reward_weights)
            if preset.reward_weights
            else None,
            "troop_loss_scale": preset.troop_loss_scale,
        }
        job = self._spawn("eval", [], cfg, write_config_only=True)
        episodes = int(body.get("episodes", 10))
        seed = str(body.get("seed", "eval"))
        use_baseline = bool(body.get("baseline"))
        stochastic = bool(body.get("stochastic"))
        # Everything needed to reproduce this eval from the terminal lives
        # next to the generated config, not just in CLI flags.
        eval_spec = {
            "model": model_path,
            "episodes": episodes,
            "seed": seed,
            "min_attack_ratio": preset.min_attack_ratio,
            "baseline": use_baseline,
            "stochastic": stochastic,
            "behavior": cfg["behavior"],
        }
        with open(RUNS_DIR / job.id / "eval.json", "w", encoding="utf-8") as handle:
            json.dump(eval_spec, handle, indent=2)
        argv = [
            sys.executable,
            "eval.py",
            "--model",
            model_path,
            "--config",
            job.config_path,
            "--episodes",
            str(episodes),
            "--seed",
            seed,
            "--min-attack-ratio",
            str(preset.min_attack_ratio),
        ]
        if use_baseline:
            argv.append("--baseline")
        if stochastic:
            argv.append("--stochastic")
        self._launch(job, argv)
        return job

    def _spawn(self, kind: str, argv: list, cfg: dict, write_config_only=False) -> Job:
        with self._lock:
            self._counter += 1
            job_id = f"{kind}-{self._counter:03d}"
        run_dir = RUNS_DIR / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        preset_path = run_dir / "behavior.json"
        with open(preset_path, "w", encoding="utf-8") as handle:
            json.dump(cfg.get("behavior", {}), handle, indent=2)
        job = Job(
            id=job_id,
            kind=kind,
            config_path=str(config_path),
            log_path=str(run_dir / "stdout.log"),
            started=now_iso(),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._evict_locked()
        if not write_config_only:
            argv = [*argv, "--config", str(config_path)]
            self._launch(job, argv)
        return job

    def _evict_locked(self, keep: int = 50) -> None:
        """Drop oldest terminal records past the cap. Running jobs stay."""
        if len(self._jobs) <= keep:
            return
        terminal = [jid for jid, job in self._jobs.items() if job.status != "running"]
        for jid in terminal[: len(self._jobs) - keep]:
            del self._jobs[jid]

    def _launch(self, job: Job, argv: list) -> None:
        log = open(job.log_path, "w", encoding="utf-8")
        job.proc = subprocess.Popen(
            argv, cwd=str(TRAIN_DIR), stdout=log, stderr=subprocess.STDOUT
        )
        threading.Thread(target=self._reap, args=(job, log), daemon=True).start()

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

    def stop(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
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
        """Forget a terminal job record. Run files stay on disk."""
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
        return [self.describe(job) for job in jobs]

    @staticmethod
    def describe(job: Job) -> dict:
        tail: list[str] = []
        try:
            with open(job.log_path, encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
            tail = [line.rstrip("\n") for line in lines[-MAX_LOG_TAIL:]]
        except OSError:
            pass
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "config_path": job.config_path,
            "started": job.started,
            "ended": job.ended,
            "returncode": job.returncode,
            "error": job.error,
            "log_tail": tail,
        }


# --------------------------------------------------------------------------
# presets


def list_presets() -> list[dict]:
    out = []
    if not PRESETS_DIR.is_dir():
        return out
    for path in sorted(PRESETS_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            preset = BehaviorPreset.from_dict(raw)
            out.append({**preset.to_dict(), "file": path.name})
        except (ValueError, OSError) as exc:
            out.append({"file": path.name, "error": str(exc)})
    return out


def save_preset(body: dict) -> dict:
    preset = BehaviorPreset.from_dict(body)
    safe = re.sub(r"[^a-z0-9-]+", "-", preset.name.lower()).strip("-") or "custom"
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    path = PRESETS_DIR / f"{safe}.json"
    raw = {
        "name": preset.name,
        "description": preset.description,
        "aggression": preset.aggression,
        "min_attack_ratio": preset.min_attack_ratio,
        "disabled_intents": list(preset.disabled_intents),
        "disabled_builds": list(preset.disabled_builds),
        "reward_weights": list(preset.reward_weights)
        if preset.reward_weights
        else None,
        "troop_loss_scale": preset.troop_loss_scale,
        "env_overrides": dict(preset.env_overrides),
        "ppo_overrides": dict(preset.ppo_overrides),
    }
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return {**preset.to_dict(), "file": path.name}


# --------------------------------------------------------------------------
# checkpoints and TensorBoard metrics


def list_checkpoints(subdir: str) -> list[dict]:
    root = resolve_under(TRAIN_DIR, subdir)
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.glob("*.zip")):
        stat = path.stat()
        out.append(
            {
                "path": str(path.relative_to(TRAIN_DIR)),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
            }
        )
    return out


def list_replays(subdir: str = "replays") -> list[dict]:
    """Recorded episode files for the phase-8 replay viewer.

    Mirrors list_checkpoints but recursive: training writes one small JSON
    per recorded episode under replays/<run>/, and the game client lists
    them without touching the filesystem.
    """
    root = resolve_under(TRAIN_DIR, subdir)
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.glob("**/*.json")):
        stat = path.stat()
        out.append(
            {
                "name": str(path.relative_to(root)),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
            }
        )
    return out


def read_metrics(tb_dir: str, run: str) -> dict:
    from tensorboard.backend.event_processing import event_accumulator

    root = resolve_under(TRAIN_DIR, tb_dir)
    matches = sorted(root.glob(f"**/{run}/events*")) or sorted(root.glob("**/events*"))
    if not matches:
        return {"scalars": {}, "note": "no event files found"}
    # Newest event file wins; TensorBoard appends to one file per run.
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    acc = event_accumulator.EventAccumulator(
        str(latest.parent),
        # Zero keeps every point; charts downsample after load. Fine for
        # local runs, but watch memory on very long ones.
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    scalars = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        step = max(1, len(events) // METRIC_POINTS)
        scalars[tag] = [
            {"step": e.step, "value": e.value} for e in events[::step]
        ]
    return {"scalars": scalars, "event_file": str(latest.relative_to(TRAIN_DIR))}


# --------------------------------------------------------------------------
# live bot session: a real 1x1 env stepped in a background thread


class LiveSession:
    """One bot playing live, every step pushed to SSE subscribers.

    Worker boot is the slow, failure-prone part (first boot compiles the
    engine under tsx, then the map loads), so it runs in phases with a
    deadline and the worker stderr tail lands in state on failure instead
    of hanging silently. ``predictor="simulated"`` skips the engine
    entirely and steps the in-process fake worker: instant, and it
    exercises the whole live path including a real policy net.
    """

    BOOT_SOCKET_TIMEOUT = 45.0

    def __init__(self):
        # RLock, not Lock: state() is called from inside other locked
        # sections (start/stop return it). A plain Lock deadlocked the
        # whole live API the first time start() ran.
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.running = False
        self.phase = "idle"
        self.phase_detail = ""
        self.error = ""
        self.error_detail = ""
        self.spec: dict = {}
        self.steps: list[dict] = []
        self.intent_counts: Counter = Counter()
        self.episodes = 0
        self._subscribers: list[queue.Queue] = []
        self._worker_logs: list[str | None] = []

    def start(self, body: dict) -> dict:
        with self._lock:
            if self.running or self.phase == "booting":
                raise ValueError("a live session is already running")
            predictor = str(body.get("predictor", "baseline"))
            if predictor not in ("baseline", "checkpoint", "simulated"):
                raise ValueError(f"unknown predictor {predictor!r}")
            model = body.get("model")
            if predictor == "checkpoint" and not model:
                raise ValueError("checkpoint predictor needs a model path")
            preset = BehaviorPreset.from_dict(body.get("behavior", {"name": "live"}))
            spec = {
                "predictor": predictor,
                "model": model,
                "seed": str(body.get("seed", "live")),
                "map": str(body.get("map", "World")),
                "bots": int(body.get("bots", 8)),
                "boot_timeout": float(body.get("boot_timeout", 300)),
                "behavior": preset.to_dict(),
            }
            self.spec = spec
            self.steps = []
            self.intent_counts = Counter()
            self.episodes = 0
            self.error = ""
            self.error_detail = ""
            self._worker_logs = []
            self._stop.clear()
            self.running = True
            self.phase = "booting"
            self.phase_detail = "starting"
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return self.state()

    def stop(self) -> dict:
        with self._lock:
            self._stop.set()
            self.running = False
            if self.phase == "booting":
                self.phase = "idle"
                self.phase_detail = "stopped during boot"
        return self.state()

    def state(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "phase": self.phase,
                "phase_detail": self.phase_detail,
                "error": self.error,
                "error_detail": self.error_detail,
                "spec": self.spec,
                "step_count": len(self.steps),
                "latest": self.steps[-1] if self.steps else None,
                "recent": self.steps[-20:],
                "intent_counts": dict(self.intent_counts),
                "episodes": self.episodes,
            }

    def subscribe(self) -> queue.Queue:
        inbox: queue.Queue = queue.Queue(maxsize=64)
        with self._lock:
            self._subscribers.append(inbox)
        return inbox

    def unsubscribe(self, inbox: queue.Queue) -> None:
        with self._lock:
            if inbox in self._subscribers:
                self._subscribers.remove(inbox)

    def _publish(self, step: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for inbox in subscribers:
            try:
                inbox.put_nowait(step)
            except queue.Full:
                pass

    def _record(self, step: dict) -> None:
        with self._lock:
            self.steps.append(step)
            del self.steps[: -LIVE_RING]
            self.intent_counts[step["action_name"]] += 1
            if step.get("terminal"):
                self.episodes += 1

    def _set_phase(self, phase: str, detail: str = "") -> None:
        with self._lock:
            self.phase = phase
            self.phase_detail = detail

    def _tail_worker_logs(self, limit: int = 30) -> str:
        tails = []
        with self._lock:
            paths = list(self._worker_logs)
        for path in paths:
            if not path:
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    lines = handle.readlines()
                tails.append(f"--- {path} ---\n" + "".join(lines[-limit:]))
            except OSError as exc:
                tails.append(f"--- {path} unreadable: {exc} ---")
        return "\n".join(tails)

    def _fail(self, message: str) -> None:
        detail = self._tail_worker_logs()
        with self._lock:
            self.error = message
            self.error_detail = detail
            self.running = False
            self.phase = "failed"
            self.phase_detail = message

    def _loop(self) -> None:
        try:
            self._run()
        except Exception as exc:  # surfaced in state, never kills the server
            self._fail(str(exc))

    def _run(self) -> None:
        import numpy as np

        from agents.scripted_agent import ScriptedBaselineAgent
        from openfront_gym.vec import OpenFrontVecEnv

        spec = self.spec
        behavior = spec["behavior"]
        boot_timeout = float(spec.get("boot_timeout", 300))
        boot_start = time.monotonic()
        simulated = spec["predictor"] == "simulated"

        factory = None
        fake = None
        if simulated:
            from openfront_gym.protocol import OBSERVATION_SIZE
            from openfront_gym.worker import WorkerClient, WorkerConfig
            from tests.fake_worker import FakeWorker

            obs_size = (
                OBSERVATION_SIZE if spec.get("model") else 8
            )
            fake = FakeWorker(num_envs=1, obs_size=obs_size)
            fake.start()
            fake_config = WorkerConfig(
                num_envs=1, socket_timeout=10.0, spawn_timeout=10.0
            )

            def factory(worker_index: int) -> WorkerClient:
                return WorkerClient.connect("127.0.0.1", fake.port, fake_config)

        self._set_phase("booting", "simulator ready" if simulated else "spawning node worker")
        try:
            vec = OpenFrontVecEnv(
                1,
                1,
                map=spec["map"],
                bots=spec["bots"],
                ticks_per_step=10,
                tick_cap=6000,
                seeds=None,
                disabled_intents=tuple(behavior["disabled_intents"]),
                disabled_builds=tuple(behavior["disabled_build_names"]),
                reward_weights=tuple(behavior["reward_weights"])
                if behavior["reward_weights"]
                else None,
                troop_loss_scale=float(behavior.get("troop_loss_scale", 1.0)),
                socket_timeout=10.0 if simulated else self.BOOT_SOCKET_TIMEOUT,
                spawn_timeout=10.0 if simulated else min(boot_timeout, 600),
                worker_factory=factory,
            )
        except Exception as exc:
            log_path = getattr(exc, "worker_log_path", None)
            if log_path:
                with self._lock:
                    self._worker_logs = [log_path]
            raise
        with self._lock:
            try:
                self._worker_logs = vec.worker_log_paths()
            except Exception:
                pass
        try:
            if time.monotonic() - boot_start > boot_timeout:
                raise TimeoutError(
                    f"worker boot took longer than {boot_timeout:.0f}s, "
                    "see error_detail for the worker log tail"
                )
            self._run_steps(vec, boot_start)
        finally:
            try:
                vec.close()
            finally:
                if fake is not None:
                    fake.stop()
            with self._lock:
                self.running = False
                if self.phase == "running":
                    self.phase = "idle"
                    self.phase_detail = "stopped"

    def _run_steps(self, vec, boot_start: float) -> None:
        import numpy as np

        from agents.scripted_agent import ScriptedBaselineAgent

        spec = self.spec
        behavior = spec["behavior"]
        model = None
        if spec["predictor"] in ("checkpoint", "simulated") and spec.get("model"):
            from sb3_contrib import MaskablePPO

            self._set_phase("booting", "loading checkpoint")
            # Same containment as eval jobs: checkpoints unpickle.
            model = MaskablePPO.load(
                str(resolve_under(TRAIN_DIR, str(spec["model"]))), device="cpu"
            )
        baseline = ScriptedBaselineAgent()
        min_ratio = int(behavior["min_attack_ratio"])
        episode = 0
        self._set_phase("booting", "resetting env")
        obs, infos = vec.reset(seed=[f"{spec['seed']}-{episode}"])
        masks = infos["action_masks"]
        self._set_phase(
            "running",
            f"boot took {time.monotonic() - boot_start:.1f}s",
        )
        total = 0.0
        step_no = 0
        while not self._stop.is_set():
            if model is not None:
                actions, _ = model.predict(
                    np.asarray(obs),
                    deterministic=False,
                    action_masks=np.asarray(masks),
                )
            else:
                actions = baseline.act_batch(np.asarray(obs), np.asarray(masks))
            actions = apply_aggression_batch(np.asarray(actions), min_ratio)
            obs, rewards, terminated, truncated, infos = vec.step(actions)
            masks = infos["action_masks"]
            step_no += 1
            total += float(rewards[0])
            done = bool(terminated[0] or truncated[0])
            terminal = str(infos["terminal_reason"][0]) if done else None
            action = [int(v) for v in np.asarray(actions)[0]]
            text = describe_action(action)
            step = {
                "n": step_no,
                "episode": episode,
                "action": action,
                "action_name": text.split(" ")[0],
                "action_text": text,
                "reward": float(rewards[0]),
                "total_reward": total,
                "ticks": int(infos["ticks"][0]),
                "terminal": terminal,
                "masks": mask_stats(np.asarray(masks)[0]),
                "ts": now_iso(),
            }
            self._record(step)
            self._publish(step)
            # Simulated steps answer in microseconds; pace them so the
            # loop does not spin. Against the real engine one step costs
            # ~1s, so 50ms changes nothing there.
            time.sleep(0.05)
            if done:
                episode += 1
                total = 0.0
                obs, infos = vec.reset(seed=[f"{spec['seed']}-{episode}"])
                masks = infos["action_masks"]


JOBS = JobManager()
LIVE = LiveSession()


# --------------------------------------------------------------------------
# Local versus play: one human-vs-AI match on a two-seat sim.


class VersusManager:
    """Owns a single PlaySession; the HTTP layer only passes dicts through."""

    def __init__(self):
        self._lock = threading.Lock()
        self._session = None

    def start(self, body: dict) -> dict:
        from action_filter import describe_action
        from play_session import PlaySession, PlaySpec

        predictor = str(body.get("predictor", "scripted"))
        model = body.get("model", None)
        if predictor == "checkpoint":
            if not model:
                raise ValueError("checkpoint predictor needs a model path")
            # Same containment as eval jobs: checkpoints unpickle.
            model = str(resolve_under(TRAIN_DIR, str(model)))
        spec = PlaySpec(
            seed=str(body.get("seed", "versus")),
            bots=int(body.get("bots", 5)),
            nations=str(body.get("nations", "default")),
            map=str(body.get("map", "World")),
            ticks_per_step=int(body.get("ticks_per_step", 10)),
            tick_cap=int(body.get("tick_cap", 6000)),
            predictor=predictor,
            model=model,
            min_attack_ratio=int(body.get("min_attack_ratio", 0)),
        )
        if spec.nations not in ("default", "disabled"):
            raise ValueError(f"nations must be default or disabled, got {spec.nations}")
        with self._lock:
            if self._session is not None:
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = None
            session = PlaySession(spec)
            snapshot = session.start()
            self._session = session
        return {"active": True, "snapshot": snapshot, "moves": 0}

    def move(self, body: dict) -> dict:
        from action_filter import describe_action

        with self._lock:
            session = self._session
            if session is None:
                raise ValueError("no active versus game, start one first")
            result = session.move(body.get("action", {}))
            moves = session.moves
        return {
            **result,
            "moves": moves,
            "ai_action_text": describe_action(result["ai_action"]),
        }

    def state(self) -> dict:
        from action_filter import describe_action

        with self._lock:
            session = self._session
            if session is None:
                return {"active": False}
            try:
                result = session.state()
            except Exception as exc:
                return {"active": True, "error": str(exc)}
        return {
            "active": True,
            "moves": result["moves"],
            "ai_action": result["ai_action"],
            "ai_action_text": describe_action(result["ai_action"]),
            **{k: result[k] for k in ("done", "ticks", "snapshot") if k in result},
        }

    def stop(self) -> dict:
        with self._lock:
            session, self._session = self._session, None
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        return {"active": False}


PLAY = VersusManager()


# --------------------------------------------------------------------------
# HTTP layer


class Handler(BaseHTTPRequestHandler):
    server_version = "OpenFrontControl/1"

    def log_message(self, *args):
        pass

    # -- dispatch ------------------------------------------------------

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        query = urllib.parse.parse_qs(url.query)
        try:
            if path.startswith("/replays/"):
                self.serve_replay_file(path[len("/replays/"):])
            elif not path.startswith("/api/"):
                self.serve_dashboard(path)
            elif path == "/api/state":
                self.send_json(
                    {
                        "jobs": JOBS.summaries(),
                        "presets": list_presets(),
                        "live": LIVE.state(),
                    }
                )
            elif path == "/api/presets":
                self.send_json({"presets": list_presets()})
            elif path == "/api/checkpoints":
                subdir = query.get("dir", ["checkpoints/openfront_ppo"])[0]
                self.send_json({"checkpoints": list_checkpoints(subdir)})
            elif path == "/api/replays":
                subdir = query.get("dir", ["replays"])[0]
                self.send_json({"replays": list_replays(subdir)}, cors=True)
            elif path == "/api/metrics":
                tb_dir = query.get("tb_dir", ["runs/openfront_ppo"])[0]
                run = query.get("run", ["ppo_default"])[0]
                self.send_json(read_metrics(tb_dir, run))
            elif path == "/api/live/state":
                self.send_json(LIVE.state())
            elif path == "/api/live/stream":
                self.serve_live_stream()
            elif path == "/api/play/state":
                self.send_json(PLAY.state())
            elif path.startswith("/api/jobs/") and path.endswith("/logs"):
                job_id = path[len("/api/jobs/") : -len("/logs")]
                self.serve_job_logs(job_id, query)
            elif path.startswith("/api/jobs/"):
                job = JOBS.get(path[len("/api/jobs/") :])
                if job is None:
                    self.send_error_json(404, "unknown job")
                else:
                    self.send_json(JOBS.describe(job))
            elif path == "/api/jobs":
                self.send_json({"jobs": JOBS.summaries()})
            else:
                self.send_error_json(404, "not found")
        except (ValueError, KeyError) as exc:
            self.send_error_json(400, str(exc))
        except Exception as exc:  # noqa: BLE001, last resort stays JSON
            self.send_error_json(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        try:
            body = read_json(self)
            if path == "/api/jobs/train":
                job = JOBS.start_train(body)
                self.send_json(JOBS.describe(job), status=201)
            elif path == "/api/jobs/eval":
                job = JOBS.start_eval(body)
                self.send_json(JOBS.describe(job), status=201)
            elif path == "/api/presets":
                self.send_json(save_preset(body), status=201)
            elif path == "/api/live/start":
                self.send_json(LIVE.start(body))
            elif path == "/api/live/stop":
                self.send_json(LIVE.stop())
            elif path == "/api/play/start":
                self.send_json(PLAY.start(body), status=201)
            elif path == "/api/play/move":
                self.send_json(PLAY.move(body))
            elif path == "/api/play/stop":
                self.send_json(PLAY.stop())
            else:
                self.send_error_json(404, "not found")
        except (ValueError, KeyError) as exc:
            self.send_error_json(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.send_error_json(500, f"{type(exc).__name__}: {exc}")

    def do_DELETE(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        try:
            if path.startswith("/api/jobs/"):
                job_id = path[len("/api/jobs/") :]
                job = JOBS.get(job_id)
                if job is None:
                    self.send_error_json(404, f"unknown job {job_id}")
                elif job.status == "running":
                    self.send_json(JOBS.describe(JOBS.stop(job_id)))
                else:
                    JOBS.remove(job_id)
                    self.send_json({"deleted": job_id})
            elif path.startswith("/api/presets/"):
                name = path[len("/api/presets/") :]
                safe = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
                if not safe:
                    raise ValueError("bad preset name")
                target = resolve_under(PRESETS_DIR, f"{safe}.json")
                if not target.is_file():
                    self.send_error_json(404, f"unknown preset {name}")
                    return
                target.unlink()
                self.send_json({"deleted": name})
            else:
                self.send_error_json(404, "not found")
        except KeyError as exc:
            self.send_error_json(404, f"unknown job {exc}")
        except (ValueError, OSError) as exc:
            self.send_error_json(400, str(exc))

    # -- responses -----------------------------------------------------

    def send_json(self, payload: dict, status: int = 200, cors: bool = False) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if cors:
            # The game client runs on its own dev port and fetches the
            # replay index cross-origin. Simple GETs need no preflight,
            # just the allow header.
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def serve_dashboard(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        target = resolve_under(DASHBOARD_DIR, path.lstrip("/"))
        if not target.is_file():
            self.send_error_json(404, "not found")
            return
        content_types = {
            ".html": "text/html",
            ".js": "text/javascript",
            ".css": "text/css",
            ".json": "application/json",
        }
        raw = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_types.get(target.suffix, "text/plain"))
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(raw)

    def serve_replay_file(self, relpath: str) -> None:
        try:
            target = resolve_under(REPLAYS_DIR, relpath)
        except (ValueError, KeyError):
            self.send_error_json(400, "bad replay path")
            return
        if target.suffix != ".json" or not target.is_file():
            self.send_error_json(404, "not found")
            return
        raw = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def serve_job_logs(self, job_id: str, query: dict) -> None:
        job = JOBS.get(job_id)
        if job is None:
            self.send_error_json(404, "unknown job")
            return
        follow = query.get("follow", ["1"])[0] != "0"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            offset = 0
            try:
                with open(job.log_path, encoding="utf-8", errors="replace") as handle:
                    lines = handle.readlines()
                keep = [line.rstrip("\n") for line in lines[-MAX_LOG_TAIL:]]
                for line in keep:
                    self._sse({"line": line, "status": job.status})
                # Byte offset, not a char count: on Windows the log has
                # CRLF endings, so summing len(line)+1 drifts and replays
                # lines. The follow-up reads below are binary for the same
                # reason.
                offset = os.path.getsize(job.log_path)
            except OSError:
                self._sse({"line": "(no log output yet)", "status": job.status})
            last_beat = time.monotonic()
            while follow:
                time.sleep(0.5)
                try:
                    with open(job.log_path, "rb") as handle:
                        handle.seek(offset)
                        raw = handle.read()
                        offset = handle.tell()
                    chunk = raw.decode("utf-8", errors="replace")
                except OSError:
                    chunk = ""
                for line in chunk.splitlines():
                    self._sse({"line": line, "status": job.status})
                if time.monotonic() - last_beat > 15:
                    self._sse({"beat": True, "status": job.status})
                    last_beat = time.monotonic()
                if job.status != "running" and not chunk:
                    self._sse({"done": True, "status": job.status})
                    return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def serve_live_stream(self) -> None:
        inbox = LIVE.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            state = LIVE.state()
            self._sse({"hello": True, **state})
            while True:
                try:
                    step = inbox.get(timeout=15)
                except queue.Empty:
                    self._sse({"beat": True, "running": LIVE.running})
                    if not LIVE.running:
                        return
                    continue
                self._sse({"step": step, "running": LIVE.running})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            LIVE.unsubscribe(inbox)

    def _sse(self, payload: dict) -> None:
        raw = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(raw)
        self.wfile.flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OpenFront AI control server.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"control server on http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Do not orphan workers on Ctrl-C: stop jobs and the live session.
        for summary in JOBS.summaries():
            if summary["status"] == "running":
                try:
                    JOBS.stop(summary["id"])
                except Exception:
                    pass
        try:
            LIVE.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
