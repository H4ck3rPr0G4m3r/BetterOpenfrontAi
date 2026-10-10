"""Bridge protocol v2, Python side. Mirrors ``env/src/bridge/Protocol.ts``.

Framing: ``[u32 payloadLen][u32 seq][u8 msgType][body]``, little endian.
HELLO/RESET bodies and the STEP trailer are JSON; actions and the ROWS
reply are binary. ROWS decodes into numpy views with no per-row copying.
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np

from . import layout as L

MSG_RESET, MSG_STEP, MSG_CLOSE = 0x01, 0x02, 0x03
MSG_HELLO, MSG_ROWS, MSG_ERROR = 0x10, 0x11, 0xFF
MAX_FRAME_BYTES = 256 * 1024 * 1024
STATS_PER_ROW = len(L.STATS)


class ProtocolError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"protocol error {code}: {message}")
        self.code = code


def send_frame(sock: socket.socket, seq: int, msg_type: int, body: bytes) -> None:
    sock.sendall(struct.pack("<IIB", 5 + len(body), seq, msg_type) + body)


def _recv_exact(sock: socket.socket, n: int) -> memoryview:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("worker closed the connection")
        got += k
    return view


def recv_frame(sock: socket.socket) -> tuple[int, int, memoryview]:
    (length,) = struct.unpack("<I", _recv_exact(sock, 4))
    if length < 5 or length > MAX_FRAME_BYTES:
        raise ProtocolError(2, f"bad frame length {length}")
    payload = _recv_exact(sock, length)
    seq, msg_type = struct.unpack_from("<IB", payload, 0)
    return seq, msg_type, payload[5:]


def encode_json(value) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def encode_step(actions: np.ndarray, seeds: list[str] | None = None, abort: list[int] | None = None) -> bytes:
    actions = np.ascontiguousarray(actions, dtype="<u2").reshape(-1, 5)
    trailer = encode_json({"seeds": seeds or [], "abort": abort or []})
    return struct.pack("<I", actions.shape[0]) + actions.tobytes() + struct.pack("<I", len(trailer)) + trailer


def rows_size(rows: int) -> int:
    return 4 + rows * (L.RASTER_SIZE + L.FEATURE_SIZE * 4 + L.MASK_SIZE + 4 + STATS_PER_ROW * 4 + 2)


def decode_rows(body: memoryview, rows: int) -> dict:
    """ROWS body -> dict of numpy arrays, one leading axis of ``rows``."""
    if len(body) != rows_size(rows):
        raise ProtocolError(2, f"ROWS body is {len(body)} bytes, expected {rows_size(rows)}")
    (n,) = struct.unpack_from("<I", body, 0)
    if n != rows:
        raise ProtocolError(3, f"ROWS covers {n} rows, expected {rows}")
    off = 4

    def take(dtype, count, shape):
        nonlocal off
        arr = np.frombuffer(body, dtype=dtype, count=count, offset=off).reshape(shape)
        off += arr.nbytes
        return arr

    return {
        "raster": take(np.uint8, rows * L.RASTER_SIZE, (rows, L.RASTER_PLANES, L.RASTER_HEIGHT, L.RASTER_WIDTH)),
        "features": take("<f4", rows * L.FEATURE_SIZE, (rows, L.FEATURE_SIZE)),
        "masks": take(np.uint8, rows * L.MASK_SIZE, (rows, L.MASK_SIZE)).view(bool),
        "reward": take("<f4", rows, (rows,)),
        "stats": take("<f4", rows * STATS_PER_ROW, (rows, STATS_PER_ROW)),
        "terminal": take(np.uint8, rows, (rows,)),
        "flags": take(np.uint8, rows, (rows,)),
    }


def decode_error(body: memoryview) -> tuple[int, str]:
    return body[0], bytes(body[1:]).decode("utf-8", errors="replace")
