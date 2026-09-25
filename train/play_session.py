"""Local human-vs-AI session: Python side of versus play.

Owns the Node ``play_server`` subprocess (see ``env/src/play/``), runs the
AI predictor for the AI seat, and applies the human's moves to the human
seat, one paired move per ``move()`` call. Turn-paced: no real-time
deadline, each human move advances ``ticks_per_step`` ticks for both seats.

Predictors (AI seat):
* ``scripted`` — the scripted baseline, no torch needed.
* ``checkpoint`` — a MaskablePPO checkpoint (torch, lazy import).
* ``noop`` — the AI idles; useful for testing the plumbing.

Stdlib plus numpy only at module level, so session management and the
JSON protocol stay testable without torch or gymnasium. The flat-mask
head order below mirrors ``openfront_gym/spaces.py`` (intent 14, target
17, ratio 8, spatial 8192, build site 32); a test pins the total.
"""

from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import tempfile
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Must stay identical to openfront_gym/spaces.py HEAD_SIZES. spaces.py
# imports gymnasium, which play-only installs may lack, so the numbers are
# repeated here instead of imported; test_play_session pins the total.
HEAD_NAMES = ("intent_type", "target", "ratio_bucket", "spatial_cell", "build_site")
HEAD_SIZES = (14, 17, 8, 8192, 32)
MASK_TOTAL = sum(HEAD_SIZES)
NOOP_ACTION = {"intentType": 0, "target": 0, "ratioBucket": 0, "spatialCell": 0, "buildSite": 0}


def default_env_root() -> Path:
    return Path(__file__).resolve().parent.parent / "env"


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def decode_obs_b64(raw: str) -> np.ndarray:
    data = base64.b64decode(raw)
    return np.frombuffer(data, dtype=np.float32).copy()


def masks_dict_to_flat(masks: dict) -> np.ndarray:
    flat = np.zeros(MASK_TOTAL, dtype=bool)
    offset = 0
    for name, size in zip(HEAD_NAMES, HEAD_SIZES):
        head = np.asarray(masks[name], dtype=bool).reshape(-1)
        if head.shape[0] != size:
            raise ValueError(f"mask head {name} has {head.shape[0]} entries, expected {size}")
        flat[offset : offset + size] = head
        offset += size
    return flat


def validate_human_action(raw: dict) -> dict:
    """Range-check a dashboard-posted action before it reaches the engine."""
    keys = (("intentType", 14), ("target", 17), ("ratioBucket", 8), ("spatialCell", 8192), ("buildSite", 32))
    if not isinstance(raw, dict):
        raise ValueError("action must be an object")
    out = {}
    for key, size in keys:
        value = raw.get(key, 0)
        if not isinstance(value, int) or not 0 <= value < size:
            raise ValueError(f"action.{key} must be an integer 0..{size - 1}")
        out[key] = value
    return out


class VersusClient:
    """Newline-JSON client for the Node versus play server."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._file = sock.makefile("r", encoding="utf-8")

    @classmethod
    def connect(cls, host: str, port: int, timeout: float = 30.0) -> "VersusClient":
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while True:
            try:
                sock = socket.create_connection((host, port), timeout=timeout)
                return cls(sock)
            except OSError as exc:
                last = exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"play server on {host}:{port} never accepted ({last})")
                time.sleep(0.25)

    def request(self, command: dict) -> dict:
        self._sock.sendall((json.dumps(command) + "\n").encode("utf-8"))
        line = self._file.readline()
        if not line:
            raise ConnectionError("play server closed the connection")
        reply = json.loads(line)
        if not reply.get("ok", False):
            raise RuntimeError(f"play server error: {reply.get('error', 'unknown')}")
        return reply

    def close(self) -> None:
        try:
            self._file.close()
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass


def spawn_play_server(port: int, env_root: Path | None = None) -> tuple[subprocess.Popen, str]:
    root = Path(env_root) if env_root is not None else default_env_root()
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node not found on PATH, needed for the versus play server")
    tsx_cli = root / "node_modules" / "tsx" / "dist" / "cli.mjs"
    if not tsx_cli.is_file():
        raise RuntimeError(f"tsx not installed under {root}, run npm install in env/ first")
    log_fd, log_path = tempfile.mkstemp(prefix=f"openfront-play-{port}-", suffix=".log")
    child_env = dict(os.environ)
    child_env["TSX_TSCONFIG_PATH"] = "tsconfig.probes.json"
    node_options = child_env.get("NODE_OPTIONS", "")
    if "--preserve-symlinks" not in node_options:
        child_env["NODE_OPTIONS"] = (node_options + " --preserve-symlinks").strip()
    cmd = [node, str(tsx_cli), "src/play/play_server.ts", "--port", str(port)]
    try:
        proc = subprocess.Popen(cmd, cwd=str(root), env=child_env, stdout=subprocess.DEVNULL, stderr=log_fd)
    except Exception:
        os.close(log_fd)
        raise
    else:
        os.close(log_fd)
    return proc, log_path


def make_predictor(kind: str, model: str | None = None, min_attack_ratio: int = 0):
    """AI-seat predictor over (obs, flat_masks) -> 5-int action array."""
    from action_filter import apply_aggression_batch

    if kind == "noop":
        return lambda obs, masks: np.zeros((obs.shape[0], 5), dtype=np.int64)
    if kind == "scripted":
        from agents.scripted_agent import ScriptedBaselineAgent

        agent = ScriptedBaselineAgent()
        return lambda obs, masks: apply_aggression_batch(
            agent.act_batch(np.asarray(obs), np.asarray(masks)), int(min_attack_ratio)
        )
    if kind == "checkpoint":
        if not model:
            raise ValueError("checkpoint predictor needs a model path")
        from sb3_contrib import MaskablePPO

        loaded = MaskablePPO.load(model, device="cpu")
        return lambda obs, masks: apply_aggression_batch(
            loaded.predict(np.asarray(obs), deterministic=True, action_masks=np.asarray(masks))[0],
            int(min_attack_ratio),
        )
    raise ValueError(f"unknown play predictor {kind!r}")


@dataclass
class PlaySpec:
    seed: str = "versus"
    bots: int = 5
    nations: str = "default"
    map: str = "World"
    ticks_per_step: int = 10
    tick_cap: int = 6000
    predictor: str = "scripted"
    model: str | None = None
    min_attack_ratio: int = 0


class PlaySession:
    """One local versus match: node sim plus AI predictor for the AI seat."""

    def __init__(self, spec: PlaySpec | None = None):
        self.spec = spec or PlaySpec()
        self._proc: subprocess.Popen | None = None
        self._client: VersusClient | None = None
        self._predict: object = None
        self._obs: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        self.last_ai_action: list[int] = [0, 0, 0, 0, 0]
        self.moves = 0

    def start(self) -> dict:
        if self._client is not None:
            raise RuntimeError("play session already started")
        port = find_free_port()
        self._proc, self._log_path = spawn_play_server(port)
        try:
            self._client = VersusClient.connect("127.0.0.1", port)
            hello = self._client.request({"cmd": "hello"})
            if hello.get("service") != "versus-play":
                raise RuntimeError(f"unexpected play service: {hello}")
            self._predict = make_predictor(self.spec.predictor, self.spec.model, self.spec.min_attack_ratio)
            reply = self._client.request(
                {
                    "cmd": "reset",
                    "seed": self.spec.seed,
                    "bots": self.spec.bots,
                    "nations": self.spec.nations,
                    "map": self.spec.map,
                    "ticksPerStep": self.spec.ticks_per_step,
                    "tickCap": self.spec.tick_cap,
                }
            )
        except Exception:
            self.close()
            raise
        self._obs = decode_obs_b64(reply["aiObsB64"])
        self._masks = masks_dict_to_flat(reply["aiMasks"])
        return reply["snapshot"]

    def move(self, human_action: dict) -> dict:
        """Apply one paired move: AI predicts from cached obs, both seats step."""
        if self._client is None or self._predict is None or self._obs is None or self._masks is None:
            raise RuntimeError("no active game, start one first")
        human = validate_human_action(human_action)
        ai_batch = self._predict(np.asarray([self._obs]), np.asarray([self._masks]))
        ai = [int(v) for v in np.asarray(ai_batch)[0]]
        self.last_ai_action = ai
        reply = self._client.request(
            {
                "cmd": "step",
                "ai": {
                    "intentType": ai[0], "target": ai[1], "ratioBucket": ai[2],
                    "spatialCell": ai[3], "buildSite": ai[4],
                },
                "human": human,
            }
        )
        self._obs = decode_obs_b64(reply["aiObsB64"])
        self._masks = masks_dict_to_flat(reply["aiMasks"])
        self.moves += 1
        return {
            "done": reply["done"],
            "terminal": reply.get("terminal"),
            "ticks": reply["ticks"],
            "aiReward": reply.get("aiReward", 0.0),
            "humanReward": reply.get("humanReward", 0.0),
            "ai_action": ai,
            "snapshot": reply["snapshot"],
        }

    def state(self) -> dict:
        if self._client is None:
            raise RuntimeError("no active game, start one first")
        reply = self._client.request({"cmd": "state"})
        return {
            "done": reply["done"],
            "ticks": reply["ticks"],
            "moves": self.moves,
            "ai_action": self.last_ai_action,
            "snapshot": reply["snapshot"],
        }

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        self._obs = None
        self._masks = None
