"""Binary bridge protocol, Python half of spec section 5.6.

The TypeScript file ``env/src/bridge/Protocol.ts`` is the authoritative
schema description. This module mirrors its layout byte for byte using
only the standard library. If the two disagree, the TypeScript side wins
and this file must be updated to match.

Framing, both directions over a local socket::

    [u32LE payloadLen][payload]
    payload = [u32LE seq][u8 msgType][body...]

Message types: 0x01 RESET, 0x02 STEP, 0x03 CLOSE, 0x10 HELLO,
0x11 RESET_REPLY, 0x12 STEP_REPLY, 0xFF ERROR. Terminal codes:
0 continuing (None), 1 win, 2 death, 3 tick_cap, 4 sim_error.

On a done step the obs, ticks and masks describe the fresh auto-reset
episode while reward, done and terminal describe the finished one.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

SCHEMA_VERSION = 1


class MessageType:
    RESET = 0x01
    STEP = 0x02
    CLOSE = 0x03
    HELLO = 0x10
    RESET_REPLY = 0x11
    STEP_REPLY = 0x12
    ERROR = 0xFF


class TerminalCode:
    CONTINUING = 0
    WIN = 1
    DEATH = 2
    TICK_CAP = 3
    SIM_ERROR = 4


class ProtocolErrorCode:
    BAD_SEQUENCE = 1
    BAD_LENGTH = 2
    BAD_MESSAGE = 3
    VERSION_MISMATCH = 4
    ILLEGAL_ACTION = 5
    INTERNAL = 6


MASK_INTENT_TYPES = 14
MASK_TARGET_SLOTS = 17
MASK_RATIO_BUCKETS = 8
MASK_SPATIAL_CELLS = 8192
MASK_BUILD_SITES = 32
MASK_TOTAL = (
    MASK_INTENT_TYPES
    + MASK_TARGET_SLOTS
    + MASK_RATIO_BUCKETS
    + MASK_SPATIAL_CELLS
    + MASK_BUILD_SITES
)

OBSERVATION_SIZE = 65684

ACTION_U16_PER_ENV = 5
ACTION_BYTES_PER_ENV = ACTION_U16_PER_ENV * 2

MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_SEED_BYTES = 256

ACTION_KEYS = ("intent_type", "target", "ratio_bucket", "spatial_cell", "build_site")
MASK_KEYS = ("intent_type", "target", "ratio_bucket", "spatial_cell", "build_site")
_MASK_WANT = (
    MASK_INTENT_TYPES,
    MASK_TARGET_SLOTS,
    MASK_RATIO_BUCKETS,
    MASK_SPATIAL_CELLS,
    MASK_BUILD_SITES,
)

_TERMINAL_TO_CODE = {
    None: TerminalCode.CONTINUING,
    "win": TerminalCode.WIN,
    "death": TerminalCode.DEATH,
    "tick_cap": TerminalCode.TICK_CAP,
    "sim_error": TerminalCode.SIM_ERROR,
}
_CODE_TO_TERMINAL = {code: reason for reason, code in _TERMINAL_TO_CODE.items()}


class ProtocolError(Exception):
    """A malformed or unexpected bridge frame. Always fatal to the connection."""

    def __init__(self, error_code: int, message: str):
        super().__init__(f"protocol: {message}")
        self.error_code = error_code


def terminal_to_code(reason) -> int:
    try:
        return _TERMINAL_TO_CODE[reason]
    except KeyError:
        raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, f"unknown terminal reason {reason!r}")


def code_to_terminal(code: int):
    try:
        return _CODE_TO_TERMINAL[code]
    except KeyError:
        raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, f"unknown terminal code {code}")


@dataclass
class HelloInfo:
    version: int
    num_envs: int
    obs_size: int
    intent_types: int
    target_slots: int
    ratio_buckets: int
    spatial_cells: int
    build_sites: int


def encode_frame(seq: int, msg_type: int, body: bytes) -> bytes:
    if not 0 <= seq <= 0xFFFFFFFF:
        raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, "seq out of u32 range")
    if not isinstance(msg_type, int) or not 0 <= msg_type <= 0xFF:
        raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, "msg_type out of u8 range")
    return struct.pack("<I", 4 + 1 + len(body)) + struct.pack("<I", seq) + bytes([msg_type]) + body


def decode_frame_payload(payload: bytes):
    if len(payload) < 5:
        raise ProtocolError(ProtocolErrorCode.BAD_LENGTH, f"payload too short: {len(payload)}")
    seq = struct.unpack_from("<I", payload, 0)[0]
    return seq, payload[4], bytes(payload[5:])


class FrameReader:
    """Incremental stream splitter. Push bytes in, get complete payloads out."""

    def __init__(self):
        self._buffered = bytearray()

    def push(self, chunk: bytes):
        self._buffered += chunk
        frames = []
        while True:
            if len(self._buffered) < 4:
                return frames
            (payload_len,) = struct.unpack_from("<I", self._buffered, 0)
            if payload_len < 5:
                raise ProtocolError(
                    ProtocolErrorCode.BAD_LENGTH, f"frame length too short: {payload_len}"
                )
            if payload_len > MAX_FRAME_BYTES:
                raise ProtocolError(
                    ProtocolErrorCode.BAD_LENGTH, f"frame length too large: {payload_len}"
                )
            if len(self._buffered) < 4 + payload_len:
                return frames
            frames.append(bytes(self._buffered[4 : 4 + payload_len]))
            del self._buffered[: 4 + payload_len]


def _encode_seed_list(seeds, allow_empty: bool) -> bytes:
    parts = []
    for seed in seeds:
        raw = seed.encode("utf-8")
        if len(raw) > MAX_SEED_BYTES:
            raise ProtocolError(
                ProtocolErrorCode.BAD_MESSAGE, f"seed longer than {MAX_SEED_BYTES} bytes"
            )
        if not allow_empty and len(raw) == 0:
            raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, "seed must not be empty")
        parts.append(struct.pack("<H", len(raw)) + raw)
    return b"".join(parts)


def _decode_seed_list(body: bytes, offset: int, count: int, allow_empty: bool):
    seeds = []
    for _ in range(count):
        if offset + 2 > len(body):
            raise ProtocolError(ProtocolErrorCode.BAD_LENGTH, "seed list truncated")
        (length,) = struct.unpack_from("<H", body, offset)
        offset += 2
        if length > MAX_SEED_BYTES:
            raise ProtocolError(
                ProtocolErrorCode.BAD_MESSAGE, f"seed longer than {MAX_SEED_BYTES} bytes"
            )
        if offset + length > len(body):
            raise ProtocolError(ProtocolErrorCode.BAD_LENGTH, "seed bytes truncated")
        try:
            seed = body[offset : offset + length].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, "seed is not valid UTF-8") from exc
        if not allow_empty and seed == "":
            raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, "seed must not be empty")
        seeds.append(seed)
        offset += length
    return seeds, offset


def encode_hello_body(info: HelloInfo) -> bytes:
    return struct.pack(
        "<8I",
        info.version,
        info.num_envs,
        info.obs_size,
        info.intent_types,
        info.target_slots,
        info.ratio_buckets,
        info.spatial_cells,
        info.build_sites,
    )


def decode_hello_body(body: bytes) -> HelloInfo:
    if len(body) != 32:
        raise ProtocolError(
            ProtocolErrorCode.BAD_LENGTH, f"hello body must be 32 bytes, got {len(body)}"
        )
    fields = struct.unpack("<8I", body)
    return HelloInfo(*fields)


def encode_reset_body(version: int, seeds) -> bytes:
    seeds = list(seeds)
    if not 1 <= len(seeds) <= 1024:
        raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, f"bad num_envs {len(seeds)}")
    return struct.pack("<II", version, len(seeds)) + _encode_seed_list(seeds, False)


def decode_reset_body(body: bytes):
    if len(body) < 8:
        raise ProtocolError(ProtocolErrorCode.BAD_LENGTH, "reset body too short")
    version, num_envs = struct.unpack_from("<II", body, 0)
    if not 1 <= num_envs <= 1024:
        raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, f"bad num_envs {num_envs}")
    seeds, offset = _decode_seed_list(body, 8, num_envs, False)
    if offset != len(body):
        raise ProtocolError(
            ProtocolErrorCode.BAD_LENGTH, f"reset body has {len(body) - offset} trailing bytes"
        )
    return version, seeds


def _check_action(action) -> None:
    for key in ACTION_KEYS:
        try:
            value = action[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProtocolError(
                ProtocolErrorCode.BAD_MESSAGE, f"action is missing head {key!r}"
            ) from exc
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFF:
            raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, f"{key} out of u16 range")


def encode_step_body(version: int, actions, next_seeds) -> bytes:
    actions = list(actions)
    next_seeds = list(next_seeds)
    if len(actions) != len(next_seeds):
        raise ProtocolError(
            ProtocolErrorCode.BAD_MESSAGE, "actions and next_seeds must cover the same envs"
        )
    out = bytearray(struct.pack("<I", version))
    for action in actions:
        _check_action(action)
        out += struct.pack("<5H", *(action[key] for key in ACTION_KEYS))
    out += _encode_seed_list(next_seeds, True)
    return bytes(out)


def decode_step_body(body: bytes, num_envs: int):
    head = 4 + num_envs * ACTION_BYTES_PER_ENV
    if len(body) < head:
        raise ProtocolError(ProtocolErrorCode.BAD_LENGTH, "step body too short")
    (version,) = struct.unpack_from("<I", body, 0)
    actions = []
    offset = 4
    for _ in range(num_envs):
        values = struct.unpack_from("<5H", body, offset)
        actions.append(dict(zip(ACTION_KEYS, values)))
        offset += ACTION_BYTES_PER_ENV
    next_seeds, offset = _decode_seed_list(body, offset, num_envs, True)
    if offset != len(body):
        raise ProtocolError(
            ProtocolErrorCode.BAD_LENGTH, f"step body has {len(body) - offset} trailing bytes"
        )
    return version, actions, next_seeds


def flatten_masks(mask: dict) -> bytes:
    parts = []
    for key, want in zip(MASK_KEYS, _MASK_WANT):
        raw = bytes(mask[key])
        if len(raw) != want:
            raise ProtocolError(
                ProtocolErrorCode.BAD_MESSAGE,
                f"mask head {key} must be {want} bytes, got {len(raw)}",
            )
        parts.append(raw)
    return b"".join(parts)


def split_masks(block: bytes) -> dict:
    if len(block) != MASK_TOTAL:
        raise ProtocolError(
            ProtocolErrorCode.BAD_LENGTH,
            f"mask block must be {MASK_TOTAL} bytes, got {len(block)}",
        )
    out = {}
    offset = 0
    for key, want in zip(MASK_KEYS, _MASK_WANT):
        out[key] = bytes(block[offset : offset + want])
        offset += want
    return out


def reset_reply_size(num_envs: int, obs_size: int) -> int:
    return num_envs * obs_size * 4 + num_envs * 4 + num_envs * MASK_TOTAL


def step_reply_size(num_envs: int, obs_size: int) -> int:
    return (
        num_envs * obs_size * 4
        + num_envs * 4
        + num_envs
        + num_envs
        + num_envs * 4
        + num_envs * MASK_TOTAL
    )


def encode_reset_reply_body(obs_list, ticks, mask_list, obs_size: int) -> bytes:
    obs_list = [list(obs) for obs in obs_list]
    num_envs = len(obs_list)
    if len(ticks) != num_envs or len(mask_list) != num_envs:
        raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, "reset reply arrays disagree")
    out = bytearray()
    for obs in obs_list:
        if len(obs) != obs_size:
            raise ProtocolError(
                ProtocolErrorCode.BAD_MESSAGE,
                f"obs must hold {obs_size} floats, got {len(obs)}",
            )
        out += struct.pack(f"<{obs_size}f", *obs)
    for tick in ticks:
        out += struct.pack("<I", tick)
    for mask in mask_list:
        out += flatten_masks(mask)
    return bytes(out)


def decode_reset_reply_body(body: bytes, num_envs: int, obs_size: int):
    if len(body) != reset_reply_size(num_envs, obs_size):
        raise ProtocolError(ProtocolErrorCode.BAD_LENGTH, "reset reply wrong size")
    offset = 0
    obs = []
    for _ in range(num_envs):
        values = struct.unpack_from(f"<{obs_size}f", body, offset)
        obs.append(list(values))
        offset += obs_size * 4
    ticks = []
    for _ in range(num_envs):
        (tick,) = struct.unpack_from("<I", body, offset)
        ticks.append(tick)
        offset += 4
    masks = []
    for _ in range(num_envs):
        masks.append(body[offset : offset + MASK_TOTAL])
        offset += MASK_TOTAL
    return obs, ticks, masks


def encode_step_reply_body(
    obs_list, rewards, done, terminal_codes, ticks, mask_list, obs_size: int
) -> bytes:
    obs_list = [list(obs) for obs in obs_list]
    num_envs = len(obs_list)
    if not (
        len(rewards) == len(done) == len(terminal_codes) == len(ticks) == len(mask_list) == num_envs
    ):
        raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, "step reply arrays disagree")
    out = bytearray()
    for obs in obs_list:
        if len(obs) != obs_size:
            raise ProtocolError(
                ProtocolErrorCode.BAD_MESSAGE,
                f"obs must hold {obs_size} floats, got {len(obs)}",
            )
        out += struct.pack(f"<{obs_size}f", *obs)
    out += struct.pack(f"<{num_envs}f", *rewards)
    out += bytes(1 if flag else 0 for flag in done)
    out += bytes(terminal_codes)
    for tick in ticks:
        out += struct.pack("<I", tick)
    for mask in mask_list:
        out += flatten_masks(mask)
    return bytes(out)


def decode_step_reply_body(body: bytes, num_envs: int, obs_size: int):
    if len(body) != step_reply_size(num_envs, obs_size):
        raise ProtocolError(ProtocolErrorCode.BAD_LENGTH, "step reply wrong size")
    offset = 0
    obs = []
    for _ in range(num_envs):
        values = struct.unpack_from(f"<{obs_size}f", body, offset)
        obs.append(list(values))
        offset += obs_size * 4
    rewards = list(struct.unpack_from(f"<{num_envs}f", body, offset))
    offset += num_envs * 4
    done = [body[offset + i] == 1 for i in range(num_envs)]
    offset += num_envs
    terminal_codes = list(body[offset : offset + num_envs])
    offset += num_envs
    ticks = []
    for _ in range(num_envs):
        (tick,) = struct.unpack_from("<I", body, offset)
        ticks.append(tick)
        offset += 4
    masks = []
    for _ in range(num_envs):
        masks.append(body[offset : offset + MASK_TOTAL])
        offset += MASK_TOTAL
    return obs, rewards, done, terminal_codes, ticks, masks


def encode_error_body(code: int, message: str) -> bytes:
    if not isinstance(code, int) or isinstance(code, bool) or not 0 <= code <= 0xFF:
        raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, f"bad error code {code}")
    raw = message.encode("utf-8")
    if len(raw) > 0xFFFF:
        raise ProtocolError(ProtocolErrorCode.BAD_MESSAGE, "error message too long")
    return bytes([code]) + struct.pack("<H", len(raw)) + raw


def decode_error_body(body: bytes):
    if len(body) < 3:
        raise ProtocolError(ProtocolErrorCode.BAD_LENGTH, "error body too short")
    (length,) = struct.unpack_from("<H", body, 1)
    if 3 + length != len(body):
        raise ProtocolError(ProtocolErrorCode.BAD_LENGTH, "error body length mismatch")
    try:
        return body[0], body[3:].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(
            ProtocolErrorCode.BAD_MESSAGE, "error message is not valid UTF-8"
        ) from exc


def send_frame(sock: socket.socket, seq: int, msg_type: int, body: bytes) -> None:
    """Send one framed message, blocking until all bytes are written."""
    sock.sendall(encode_frame(seq, msg_type, body))


def recv_frame(sock: socket.socket):
    """Read one framed message. Raises ProtocolError or ConnectionError."""
    prefix = _recv_exact(sock, 4)
    (payload_len,) = struct.unpack("<I", prefix)
    if payload_len < 5:
        raise ProtocolError(ProtocolErrorCode.BAD_LENGTH, f"frame length too short: {payload_len}")
    if payload_len > MAX_FRAME_BYTES:
        raise ProtocolError(
            ProtocolErrorCode.BAD_LENGTH, f"frame length too large: {payload_len}"
        )
    return decode_frame_payload(_recv_exact(sock, payload_len))


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
