"""In-process stand-in for a Node bridge worker, for gym-level tests.

Speaks the real wire protocol with tiny observations and scripted
terminals, mirroring the WorkerServer contract: HELLO on connect, RESET
and STEP replies, same-step auto-reset where a done row carries the fresh
episode, CLOSE drops the client. Kills are abrupt to exercise the
worker-death paths.
"""

from __future__ import annotations

import socket
import struct
import threading
import zlib

from openfront_gym import protocol
from openfront_gym.protocol import FrameReader, MessageType


def seed_fill(seed: str) -> float:
    return (zlib.crc32(seed.encode("utf-8")) % 997) / 997.0


class FakeWorker:
    def __init__(self, num_envs: int = 1, obs_size: int = 8, terminals=None):
        self.num_envs = num_envs
        self.obs_size = obs_size
        self.terminals: dict[int, list[int]] = {
            index: list(codes) for index, codes in (terminals or {}).items()
        }
        self.received: list = []
        self._listener: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._conns: list[socket.socket] = []
        self._lock = threading.Lock()
        self._seeds = [f"fake-{i}" for i in range(num_envs)]
        self._episodes = [0] * num_envs
        self._ticks = [302] * num_envs
        self._port = 0

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> int:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(0.5)
        self._listener = listener
        self._port = int(listener.getsockname()[1])
        thread = threading.Thread(target=self._accept_loop, daemon=True)
        thread.start()
        self._threads.append(thread)
        return self._port

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self._lock:
            conns, self._conns = self._conns, []
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

    def kill(self) -> None:
        """Abrupt death: connections drop, the port goes away."""
        self.stop()

    def _accept_loop(self) -> None:
        while self._listener is not None:
            try:
                conn, _ = self._listener.accept()  # type: ignore[union-attr]
            except (OSError, socket.timeout):
                continue
            with self._lock:
                self._conns.append(conn)
            thread = threading.Thread(target=self._serve, args=(conn,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _serve(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(30.0)
            protocol.send_frame(
                conn,
                0,
                MessageType.HELLO,
                protocol.encode_hello_body(
                    protocol.HelloInfo(
                        version=protocol.SCHEMA_VERSION,
                        num_envs=self.num_envs,
                        obs_size=self.obs_size,
                        intent_types=protocol.MASK_INTENT_TYPES,
                        target_slots=protocol.MASK_TARGET_SLOTS,
                        ratio_buckets=protocol.MASK_RATIO_BUCKETS,
                        spatial_cells=protocol.MASK_SPATIAL_CELLS,
                        build_sites=protocol.MASK_BUILD_SITES,
                    )
                ),
            )
            reader = FrameReader()
            expected = 1
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                for payload in reader.push(chunk):
                    seq, msg_type, body = protocol.decode_frame_payload(payload)
                    if seq != expected:
                        protocol.send_frame(
                            conn,
                            seq,
                            MessageType.ERROR,
                            protocol.encode_error_body(1, "bad seq"),
                        )
                        return
                    expected += 1
                    if msg_type == MessageType.RESET:
                        self._on_reset(conn, seq, body)
                    elif msg_type == MessageType.STEP:
                        self._on_step(conn, seq, body)
                    elif msg_type == MessageType.CLOSE:
                        return
                    else:
                        protocol.send_frame(
                            conn,
                            seq,
                            MessageType.ERROR,
                            protocol.encode_error_body(3, "unknown type"),
                        )
                        return
        except (OSError, protocol.ProtocolError):
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _masks(self) -> bytes:
        return bytes([1]) * protocol.MASK_TOTAL

    def _mask_dict(self) -> dict:
        return {
            "intent_type": bytes([1]) * protocol.MASK_INTENT_TYPES,
            "target": bytes([1]) * protocol.MASK_TARGET_SLOTS,
            "ratio_bucket": bytes([1]) * protocol.MASK_RATIO_BUCKETS,
            "spatial_cell": bytes([1]) * protocol.MASK_SPATIAL_CELLS,
            "build_site": bytes([1]) * protocol.MASK_BUILD_SITES,
        }

    def _on_reset(self, conn: socket.socket, seq: int, body: bytes) -> None:
        version, seeds = protocol.decode_reset_body(body)
        if version != protocol.SCHEMA_VERSION or len(seeds) != self.num_envs:
            protocol.send_frame(
                conn, seq, MessageType.ERROR, protocol.encode_error_body(3, "bad reset")
            )
            return
        self.received.append(("reset", seeds))
        self._seeds = list(seeds)
        self._episodes = [0] * self.num_envs
        self._ticks = [302] * self.num_envs
        obs = [[seed_fill(seed)] * self.obs_size for seed in seeds]
        protocol.send_frame(
            conn,
            seq,
            MessageType.RESET_REPLY,
            protocol.encode_reset_reply_body(
                obs, list(self._ticks), [self._mask_dict() for _ in range(self.num_envs)], self.obs_size
            ),
        )

    def _on_step(self, conn: socket.socket, seq: int, body: bytes) -> None:
        version, actions, next_seeds = protocol.decode_step_body(body, self.num_envs)
        if version != protocol.SCHEMA_VERSION:
            protocol.send_frame(
                conn, seq, MessageType.ERROR, protocol.encode_error_body(4, "bad version")
            )
            return
        self.received.append(("step", actions))
        obs, rewards, done, terminals, ticks, masks = [], [], [], [], [], []
        for index in range(self.num_envs):
            queue = self.terminals.get(index, [])
            code = queue.pop(0) if queue else protocol.TerminalCode.CONTINUING
            is_done = code != protocol.TerminalCode.CONTINUING
            if is_done:
                self._episodes[index] += 1
                fresh = next_seeds[index] or f"{self._seeds[index]}#auto{self._episodes[index]}"
                self._seeds[index] = fresh
                self._ticks[index] = 302
                obs.append([seed_fill(fresh)] * self.obs_size)
                rewards.append(1.0)
                done.append(True)
                terminals.append(code)
                ticks.append(self._ticks[index])
            else:
                self._ticks[index] += 10
                obs.append([seed_fill(self._seeds[index]) + 0.01] * self.obs_size)
                rewards.append(0.1)
                done.append(False)
                terminals.append(protocol.TerminalCode.CONTINUING)
                ticks.append(self._ticks[index])
            masks.append(self._mask_dict())
        protocol.send_frame(
            conn,
            seq,
            MessageType.STEP_REPLY,
            protocol.encode_step_reply_body(
                obs, rewards, done, terminals, ticks, masks, self.obs_size
            ),
        )


def check_struct_available() -> None:
    """Smoke import so test collection fails early without the stdlib."""
    struct.pack("<I", 1)
