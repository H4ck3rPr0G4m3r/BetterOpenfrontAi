"""Tests for local versus play plumbing (no torch, no gymnasium, no engine)."""

import base64
import json
import socket
import threading

import numpy as np
import pytest

from play_session import (
    HEAD_SIZES,
    MASK_TOTAL,
    VersusClient,
    decode_obs_b64,
    make_predictor,
    masks_dict_to_flat,
    validate_human_action,
)


def test_mask_layout_total_is_pinned():
    # Mirrors openfront_gym/spaces.py HEAD_SIZES; the play server speaks the
    # same head order, so a drift here breaks every AI move.
    assert HEAD_SIZES == (14, 17, 8, 8192, 32)
    assert MASK_TOTAL == 8263


def test_masks_dict_to_flat_roundtrip():
    masks = {
        "intent_type": [1] * 14,
        "target": [0] * 17,
        "ratio_bucket": [1] * 8,
        "spatial_cell": [0] * 8192,
        "build_site": [1] * 32,
    }
    flat = masks_dict_to_flat(masks)
    assert flat.shape == (MASK_TOTAL,)
    assert flat[:14].all() and not flat[14:31].any()
    with pytest.raises(ValueError):
        masks_dict_to_flat({**masks, "target": [0] * 5})


def test_decode_obs_b64():
    obs = np.arange(8, dtype=np.float32)
    raw = base64.b64encode(obs.tobytes()).decode("ascii")
    np.testing.assert_array_equal(decode_obs_b64(raw), obs)


def test_validate_human_action():
    good = validate_human_action({"intentType": 1, "target": 0, "ratioBucket": 4})
    assert good == {"intentType": 1, "target": 0, "ratioBucket": 4, "spatialCell": 0, "buildSite": 0}
    with pytest.raises(ValueError):
        validate_human_action({"intentType": 99})
    with pytest.raises(ValueError):
        validate_human_action({"spatialCell": 8192})
    with pytest.raises(ValueError):
        validate_human_action("attack")


def test_noop_predictor_shape():
    predict = make_predictor("noop")
    obs = np.zeros((1, 8), dtype=np.float32)
    masks = np.ones((1, MASK_TOTAL), dtype=bool)
    out = predict(obs, masks)
    assert out.shape == (1, 5)
    with pytest.raises(ValueError):
        make_predictor("telepathy")


class FakePlayServer(threading.Thread):
    """Loopback newline-JSON server speaking just enough of the protocol."""

    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.seen = []

    def run(self):
        conn, _ = self.sock.accept()
        with conn, conn.makefile("r", encoding="utf-8") as handle:
            for line in handle:
                command = json.loads(line)
                self.seen.append(command.get("cmd"))
                if command["cmd"] == "hello":
                    conn.sendall(b'{"ok": true, "service": "versus-play", "version": 1}\n')
                else:
                    conn.sendall(b'{"ok": false, "error": "stop"}\n')


def test_versus_client_hello_and_error():
    server = FakePlayServer()
    server.start()
    client = VersusClient.connect("127.0.0.1", server.port, timeout=5.0)
    try:
        hello = client.request({"cmd": "hello"})
        assert hello["service"] == "versus-play"
        with pytest.raises(RuntimeError):
            client.request({"cmd": "bogus"})
    finally:
        client.close()
    assert server.seen == ["hello", "bogus"]
