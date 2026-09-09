"""Mirror of env/tests/Protocol.test.ts core vectors.

Run with ``python -m pytest`` or ``python -m unittest`` from train/.
"""

import unittest

from openfront_gym.protocol import (
    MASK_TOTAL,
    SCHEMA_VERSION,
    FrameReader,
    HelloInfo,
    MessageType,
    ProtocolError,
    TerminalCode,
    code_to_terminal,
    decode_error_body,
    decode_frame_payload,
    decode_hello_body,
    decode_reset_body,
    decode_reset_reply_body,
    decode_step_body,
    decode_step_reply_body,
    encode_error_body,
    encode_frame,
    encode_hello_body,
    encode_reset_body,
    encode_reset_reply_body,
    encode_step_body,
    encode_step_reply_body,
    flatten_masks,
    split_masks,
    terminal_to_code,
)

OBS = 5


def _masks(fill=1):
    return bytes([fill]) * MASK_TOTAL


def _mask_dict(fill=1):
    return {
        "intent_type": bytes([fill]) * 14,
        "target": bytes([fill]) * 17,
        "ratio_bucket": bytes([fill]) * 8,
        "spatial_cell": bytes([fill]) * 8192,
        "build_site": bytes([fill]) * 32,
    }


def _action(**overrides):
    action = {
        "intent_type": 1,
        "target": 0,
        "ratio_bucket": 4,
        "spatial_cell": 100,
        "build_site": 3,
    }
    action.update(overrides)
    return action


class ProtocolTest(unittest.TestCase):
    def test_frame_round_trip(self):
        framed = encode_frame(41, MessageType.STEP, bytes([9, 8, 7, 6]))
        self.assertEqual(len(framed), 4 + 4 + 1 + 4)
        seq, msg_type, body = decode_frame_payload(framed[4:])
        self.assertEqual((seq, msg_type, body), (41, MessageType.STEP, bytes([9, 8, 7, 6])))

    def test_anchor_vector(self):
        # Byte for byte identical to the TypeScript anchor vector. If this
        # changes, env/tests/Protocol.test.ts must change with it.
        self.assertEqual(
            list(encode_frame(7, MessageType.CLOSE, b"")), [5, 0, 0, 0, 7, 0, 0, 0, 3]
        )

    def test_frame_rejects_bad_message_type(self):
        with self.assertRaises(ProtocolError):
            encode_frame(1, 256, b"")
        with self.assertRaises(ProtocolError):
            encode_frame(1, -1, b"")

    def test_decode_rejects_non_utf8_text(self):
        import struct as _struct

        bad_seed = _struct.pack("<II", 1, 1) + _struct.pack("<H", 1) + b"\xff"
        with self.assertRaises(ProtocolError):
            decode_reset_body(bad_seed)
        with self.assertRaises(ProtocolError):
            decode_error_body(bytes([4]) + _struct.pack("<H", 1) + b"\xff")

    def test_frame_reader_splits_and_joins(self):
        first = encode_frame(1, MessageType.RESET, bytes([1]))
        second = encode_frame(2, MessageType.STEP, bytes([2, 3]))
        self.assertEqual(len(FrameReader().push(first + second)), 2)
        reader = FrameReader()
        self.assertEqual(reader.push((first + second)[:3]), [])
        self.assertEqual(len(reader.push((first + second)[3:])), 2)

    def test_frame_reader_rejects_absurd_length(self):
        with self.assertRaises(ProtocolError):
            FrameReader().push(b"\xff\xff\xff\xff")

    def test_hello_round_trip(self):
        info = HelloInfo(1, 4, 65684, 14, 17, 8, 8192, 32)
        self.assertEqual(decode_hello_body(encode_hello_body(info)), info)

    def test_reset_round_trip(self):
        version, seeds = decode_reset_body(encode_reset_body(SCHEMA_VERSION, ["ep-0", "s-1"]))
        self.assertEqual((version, seeds), (SCHEMA_VERSION, ["ep-0", "s-1"]))

    def test_reset_rejects_empty_seed(self):
        with self.assertRaises(ProtocolError):
            encode_reset_body(SCHEMA_VERSION, [""])
        with self.assertRaises(ProtocolError):
            encode_reset_body(SCHEMA_VERSION, [])

    def test_step_round_trip(self):
        actions = [_action(), _action(intent_type=0, spatial_cell=8191)]
        version, decoded, next_seeds = decode_step_body(
            encode_step_body(SCHEMA_VERSION, actions, ["", "custom-next"]), 2
        )
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertEqual(decoded, actions)
        self.assertEqual(next_seeds, ["", "custom-next"])

    def test_step_rejects_malformed_actions(self):
        incomplete = _action()
        del incomplete["target"]
        with self.assertRaises(ProtocolError):
            encode_step_body(SCHEMA_VERSION, [incomplete], [""])
        out_of_range = _action(intent_type=70000)
        with self.assertRaises(ProtocolError):
            encode_step_body(SCHEMA_VERSION, [out_of_range], [""])

    def test_reset_reply_round_trip(self):
        body = encode_reset_reply_body(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [302, 303],
            [_mask_dict(1), _mask_dict(0)],
            OBS,
        )
        obs, ticks, masks = decode_reset_reply_body(body, 2, OBS)
        self.assertEqual(obs[0], [1, 2, 3, 4, 5])
        self.assertEqual(ticks, [302, 303])
        self.assertEqual(masks[0], _masks(1))

    def test_step_reply_round_trip(self):
        body = encode_step_reply_body(
            [[0.5, -1.25, 3, 4, 5]],
            [0.25],
            [True],
            [TerminalCode.WIN],
            [312],
            [_mask_dict(1)],
            OBS,
        )
        obs, rewards, done, terminals, ticks, _ = decode_step_reply_body(body, 1, OBS)
        self.assertAlmostEqual(rewards[0], 0.25, places=6)
        self.assertEqual((done, terminals, ticks), ([True], [TerminalCode.WIN], [312]))
        self.assertAlmostEqual(obs[0][0], 0.5, places=6)

    def test_terminal_mapping(self):
        self.assertEqual(terminal_to_code(None), TerminalCode.CONTINUING)
        self.assertEqual(terminal_to_code("sim_error"), TerminalCode.SIM_ERROR)
        self.assertEqual(code_to_terminal(TerminalCode.WIN), "win")
        self.assertIsNone(code_to_terminal(TerminalCode.CONTINUING))
        with self.assertRaises(ProtocolError):
            code_to_terminal(99)

    def test_masks_split_in_schema_order(self):
        flat = flatten_masks(_mask_dict(1))
        self.assertEqual(len(flat), MASK_TOTAL)
        split = split_masks(flat)
        self.assertEqual(split["intent_type"], bytes([1]) * 14)
        with self.assertRaises(ProtocolError):
            split_masks(b"\x00\x01\x02")

    def test_error_round_trip(self):
        self.assertEqual(
            decode_error_body(encode_error_body(4, "version drift")), (4, "version drift")
        )

    def test_error_body_rejects_bad_code(self):
        with self.assertRaises(ProtocolError):
            encode_error_body(256, "too big")
        with self.assertRaises(ProtocolError):
            encode_error_body(-1, "negative")


if __name__ == "__main__":
    unittest.main()
