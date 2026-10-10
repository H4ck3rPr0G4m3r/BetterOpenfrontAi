"""Spawning and talking to Node bridge workers (``env/src/bridge/worker_main.ts``).

Two environment details are load-bearing: tsx only forwards compiler
options to files matched by the tsconfig include, so the child gets
``TSX_TSCONFIG_PATH=tsconfig.probes.json`` (otherwise engine classes
miscompile); and ``--preserve-symlinks`` keeps a junctioned OpenFrontIO
matching that include.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import layout as L
from . import protocol as P

ENV_ROOT = Path(__file__).resolve().parents[2] / "env"


@dataclass
class WorkerConfig:
    games: int = 1
    seats: int = 1
    map: str = "World"
    bots: int = 20
    nations: str = "default"
    ticks_per_step: int = 10
    tick_cap: int = 6000
    maps_dir: str | None = None
    disabled_intents: tuple = ()
    disabled_builds: tuple = ()
    reward: dict = field(default_factory=dict)
    socket_timeout: float = 300.0
    boot_timeout: float = 120.0

    def argv(self, port: int) -> list[str]:
        argv = [
            "--port", str(port), "--games", str(self.games), "--seats", str(self.seats),
            "--map", self.map, "--bots", str(self.bots), "--nations", self.nations,
            "--ticks-per-step", str(self.ticks_per_step), "--tick-cap", str(self.tick_cap),
        ]
        if self.maps_dir:
            argv += ["--maps-dir", str(self.maps_dir)]
        for intent in self.disabled_intents:
            argv += ["--disable-intent", str(int(intent))]
        for build in self.disabled_builds:
            argv += ["--disable-build", str(build)]
        if self.reward:
            argv += ["--reward", json.dumps(self.reward)]
        return argv


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def spawn_worker(config: WorkerConfig, port: int, env_root: Path = ENV_ROOT) -> tuple[subprocess.Popen, str]:
    """Boot one worker process. Returns it and its stderr log path."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node not found on PATH, needed to run the bridge worker")
    tsx = Path(env_root) / "node_modules" / "tsx" / "dist" / "cli.mjs"
    if not tsx.is_file():
        raise RuntimeError(f"tsx not installed under {env_root}; run `npm install` in env/")
    fd, log_path = tempfile.mkstemp(prefix=f"openfront-worker-{port}-", suffix=".log")
    env = dict(os.environ)
    env["TSX_TSCONFIG_PATH"] = "tsconfig.probes.json"
    if "--preserve-symlinks" not in env.get("NODE_OPTIONS", ""):
        env["NODE_OPTIONS"] = (env.get("NODE_OPTIONS", "") + " --preserve-symlinks").strip()
    try:
        proc = subprocess.Popen(
            [node, str(tsx), "src/bridge/worker_main.ts", *config.argv(port)],
            cwd=str(env_root), env=env, stdout=subprocess.DEVNULL, stderr=fd,
        )
    finally:
        os.close(fd)
    return proc, log_path


def tail(path: str | None, lines: int = 30) -> str:
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError:
        return ""


class WorkerClient:
    """One connected worker: framing, sequence numbers, process lifetime."""

    def __init__(self, sock: socket.socket, hello: dict, proc=None, log_path: str | None = None):
        self.sock = sock
        self.hello = hello
        self.rows = int(hello["rows"])
        self.games = int(hello["games"])
        self.seats = int(hello["seats"])
        self.proc = proc
        self.log_path = log_path
        self._seq = 0
        self.closed = False

    @classmethod
    def connect(cls, port: int, config: WorkerConfig, proc=None, log_path=None, host="127.0.0.1") -> "WorkerClient":
        deadline = time.monotonic() + config.boot_timeout
        while True:
            try:
                sock = socket.create_connection((host, port), timeout=config.socket_timeout)
                break
            except OSError as exc:
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(f"worker exited during boot (code {proc.returncode}):\n{tail(log_path)}") from exc
                if time.monotonic() > deadline:
                    raise RuntimeError(f"worker on port {port} never accepted:\n{tail(log_path)}") from exc
                time.sleep(0.25)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            seq, msg_type, body = P.recv_frame(sock)
            if msg_type != P.MSG_HELLO:
                raise P.ProtocolError(3, f"expected HELLO, got {msg_type:#x}")
            hello = json.loads(bytes(body))
            L.check_layout(hello["layout"])
            if hello["games"] != config.games or hello["seats"] != config.seats:
                raise P.ProtocolError(3, f"worker hosts {hello['games']}x{hello['seats']}, expected {config.games}x{config.seats}")
        except Exception:
            sock.close()
            raise
        return cls(sock, hello, proc, log_path)

    def _request(self, msg_type: int, body: bytes) -> dict:
        if self.closed:
            raise ConnectionError("worker connection is closed")
        self._seq += 1
        P.send_frame(self.sock, self._seq, msg_type, body)
        seq, reply_type, reply = P.recv_frame(self.sock)
        if reply_type == P.MSG_ERROR:
            code, message = P.decode_error(reply)
            raise P.ProtocolError(code, message)
        if seq != self._seq or reply_type != P.MSG_ROWS:
            raise P.ProtocolError(1, f"expected ROWS seq {self._seq}, got type {reply_type:#x} seq {seq}")
        return P.decode_rows(reply, self.rows)

    def reset(self, seeds: list[str]) -> dict:
        return self._request(P.MSG_RESET, P.encode_json({"seeds": list(seeds)}))

    def step(self, actions: np.ndarray, seeds: list[str] | None = None, abort: list[int] | None = None) -> dict:
        return self._request(P.MSG_STEP, P.encode_step(actions, seeds, abort))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            P.send_frame(self.sock, self._seq + 1, P.MSG_CLOSE, b"")
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def start_worker(config: WorkerConfig, env_root: Path = ENV_ROOT) -> WorkerClient:
    port = find_free_port()
    proc, log_path = spawn_worker(config, port, env_root)
    try:
        return WorkerClient.connect(port, config, proc, log_path)
    except Exception:
        proc.terminate()
        raise
