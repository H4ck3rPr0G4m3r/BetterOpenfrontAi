"""Spawning and speaking to Node bridge workers.

A worker is ``env/src/bridge/worker_main.ts`` run under tsx. Python picks
a free TCP port, spawns the process, and connects. Two environment
details matter here. tsx only forwards its compiler options to files
matched by the tsconfig include, so the child gets
``TSX_TSCONFIG_PATH=tsconfig.probes.json`` or engine classes miscompile.
And when ``OpenFrontIO`` is a symlink or junction, Node must keep the
symlinked path or the tsconfig include stops matching, hence
``--preserve-symlinks``.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import protocol
from .protocol import MessageType


@dataclass
class WorkerConfig:
    """Everything needed to boot one worker process."""

    map: str = "World"
    bots: int = 20
    ticks_per_step: int = 10
    tick_cap: int = 6000
    maps_dir: str | None = None
    num_envs: int = 1
    strict_illegal_actions: bool = True
    socket_timeout: float = 180.0
    spawn_timeout: float = 120.0
    connect_retry_interval: float = 0.5
    # Side-quest AI control UI, enforced engine-side (see
    # env/src/BehaviorConfig.ts). Empty means no limits.
    disabled_intents: tuple = ()
    disabled_builds: tuple = ()
    # (land, troop, gold, win, death) shaping override, or None for the
    # phase-1 default.
    reward_weights: tuple | None = None


def default_env_root() -> Path:
    """The ``env/`` dir next to this checkout's ``train/`` dir."""
    return Path(__file__).resolve().parent.parent.parent / "env"


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def worker_argv(config: WorkerConfig, port: int) -> list[str]:
    argv = [
        "--port",
        str(port),
        "--num-envs",
        str(config.num_envs),
        "--map",
        config.map,
        "--bots",
        str(config.bots),
        "--ticks-per-step",
        str(config.ticks_per_step),
        "--tick-cap",
        str(config.tick_cap),
        "--strict" if config.strict_illegal_actions else "--lenient",
    ]
    if config.maps_dir is not None:
        argv += ["--maps-dir", config.maps_dir]
    for intent in config.disabled_intents:
        argv += ["--disable-intent", str(intent)]
    for build in config.disabled_builds:
        argv += ["--disable-build", str(build)]
    if config.reward_weights is not None:
        argv += ["--reward-weights", ",".join(str(v) for v in config.reward_weights)]
    return argv


def spawn_worker(
    config: WorkerConfig, port: int, env_root: Path | None = None
) -> tuple[subprocess.Popen, str]:
    """Boot a worker on ``port``. Returns the process and its stderr log path."""
    root = Path(env_root) if env_root is not None else default_env_root()
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node not found on PATH, needed to run the bridge worker")
    tsx_cli = root / "node_modules" / "tsx" / "dist" / "cli.mjs"
    if not tsx_cli.is_file():
        raise RuntimeError(f"tsx not installed under {root}, run npm install in env/ first")
    log_fd, log_path = tempfile.mkstemp(prefix=f"openfront-worker-{port}-", suffix=".log")
    child_env = dict(os.environ)
    child_env["TSX_TSCONFIG_PATH"] = "tsconfig.probes.json"
    node_options = child_env.get("NODE_OPTIONS", "")
    if "--preserve-symlinks" not in node_options:
        child_env["NODE_OPTIONS"] = (node_options + " --preserve-symlinks").strip()
    cmd = [node, str(tsx_cli), "src/bridge/worker_main.ts", *worker_argv(config, port)]
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(root), env=child_env, stdout=subprocess.DEVNULL, stderr=log_fd
        )
    except Exception:
        os.close(log_fd)
        raise
    else:
        os.close(log_fd)
    return proc, log_path


class WorkerClient:
    """One connected worker: socket framing, sequence numbers, process handle."""

    def __init__(
        self,
        sock: socket.socket,
        obs_size: int,
        num_envs: int,
        proc: subprocess.Popen | None = None,
        log_path: str | None = None,
    ):
        self._sock = sock
        self.obs_size = obs_size
        self.num_envs = num_envs
        self.proc = proc
        self.log_path = log_path
        self._seq = 0
        self._closed = False

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        config: WorkerConfig,
        proc: subprocess.Popen | None = None,
        log_path: str | None = None,
    ) -> WorkerClient:
        """Connect with retries until the worker accepts, then check HELLO."""
        deadline = time.monotonic() + config.spawn_timeout
        last_error: Exception | None = None
        while True:
            try:
                sock = socket.create_connection((host, port), timeout=config.socket_timeout)
                break
            except OSError as exc:
                last_error = exc
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(
                        f"worker exited during boot, see {log_path}: returncode {proc.returncode}"
                    ) from exc
                if time.monotonic() >= deadline:
                    hint = f", worker log at {log_path}" if log_path else ""
                    raise RuntimeError(
                        f"worker on {host}:{port} never accepted ({last_error}{hint})"
                    ) from exc
                time.sleep(config.connect_retry_interval)
        try:
            seq, msg_type, body = protocol.recv_frame(sock)
            if msg_type != MessageType.HELLO:
                raise protocol.ProtocolError(
                    protocol.ProtocolErrorCode.BAD_MESSAGE,
                    f"expected HELLO, got {msg_type:#x}",
                )
            hello = protocol.decode_hello_body(body)
            if hello.version != protocol.SCHEMA_VERSION:
                raise protocol.ProtocolError(
                    protocol.ProtocolErrorCode.VERSION_MISMATCH,
                    f"worker schema {hello.version}, client {protocol.SCHEMA_VERSION}",
                )
            if hello.num_envs != config.num_envs:
                raise protocol.ProtocolError(
                    protocol.ProtocolErrorCode.BAD_MESSAGE,
                    f"worker hosts {hello.num_envs} envs, expected {config.num_envs}",
                )
        except Exception:
            sock.close()
            raise
        return cls(sock, hello.obs_size, hello.num_envs, proc=proc, log_path=log_path)

    @property
    def closed(self) -> bool:
        return self._closed

    def _request(self, msg_type: int, body: bytes):
        if self._closed:
            raise ConnectionError("worker connection is closed")
        self._seq += 1
        protocol.send_frame(self._sock, self._seq, msg_type, body)
        seq, reply_type, reply = protocol.recv_frame(self._sock)
        if seq != self._seq:
            raise protocol.ProtocolError(
                protocol.ProtocolErrorCode.BAD_SEQUENCE,
                f"expected seq {self._seq}, got {seq}",
            )
        if reply_type == MessageType.ERROR:
            code, message = protocol.decode_error_body(reply)
            raise protocol.ProtocolError(code, f"worker error {code}: {message}")
        return reply_type, reply

    def reset(self, seeds: list[str]):
        reply_type, reply = self._request(
            MessageType.RESET, protocol.encode_reset_body(protocol.SCHEMA_VERSION, seeds)
        )
        if reply_type != MessageType.RESET_REPLY:
            raise protocol.ProtocolError(
                protocol.ProtocolErrorCode.BAD_MESSAGE,
                f"expected RESET_REPLY, got {reply_type:#x}",
            )
        return protocol.decode_reset_reply_body(reply, self.num_envs, self.obs_size)

    def step(self, actions: list[dict], next_seeds: list[str]):
        reply_type, reply = self._request(
            MessageType.STEP,
            protocol.encode_step_body(protocol.SCHEMA_VERSION, actions, next_seeds),
        )
        if reply_type != MessageType.STEP_REPLY:
            raise protocol.ProtocolError(
                protocol.ProtocolErrorCode.BAD_MESSAGE,
                f"expected STEP_REPLY, got {reply_type:#x}",
            )
        return protocol.decode_step_reply_body(reply, self.num_envs, self.obs_size)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            protocol.send_frame(self._sock, self._seq + 1, MessageType.CLOSE, b"")
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)


def connect_worker(
    config: WorkerConfig,
    port: int,
    env_root: Path | None = None,
    proc: subprocess.Popen | None = None,
    log_path: str | None = None,
) -> WorkerClient:
    """Spawn on ``port`` unless ``proc`` is given, then connect."""
    if proc is None:
        proc, log_path = spawn_worker(config, port, env_root)
    try:
        return WorkerClient.connect("127.0.0.1", port, config, proc=proc, log_path=log_path)
    except Exception as exc:
        if proc is not None:
            proc.terminate()
        # The stderr log is the only witness to a boot failure. Staple its
        # path to the exception so callers (e.g. the control server) can
        # surface the tail instead of a bare timeout.
        if log_path is not None and not hasattr(exc, "worker_log_path"):
            exc.worker_log_path = log_path  # noqa: B010
        raise


__all__ = [
    "WorkerClient",
    "WorkerConfig",
    "connect_worker",
    "default_env_root",
    "find_free_port",
    "spawn_worker",
    "worker_argv",
]
