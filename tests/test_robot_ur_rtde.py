import struct
import unittest

from vision_robot_arm.robot.ur_rtde import (
    CONTROL_PACKAGE_SETUP_OUTPUTS,
    CONTROL_PACKAGE_START,
    DATA_PACKAGE,
    REQUEST_PROTOCOL_VERSION,
    TEXT_MESSAGE,
    RtdeClient,
    decode_packets,
    decode_values,
    encode_packet,
)

VARIABLES = ("actual_q", "robot_mode", "safety_status")
TYPES = b"VECTOR6D,INT32,UINT32"


def data_package(joints: tuple[float, ...], mode: int = 7, safety: int = 1) -> bytes:
    payload = b"\x01" + struct.pack(">6d", *joints) + struct.pack(">i", mode) + struct.pack(">I", safety)
    return encode_packet(DATA_PACKAGE, payload)


class ScriptedSocket:
    """Answers each request with the next queued reply, like the controller would."""

    def __init__(self, replies: list[bytes]) -> None:
        self._replies = list(replies)
        self.sent: list[bytes] = []
        self.timeouts: list[float] = []
        self.closed = False

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)

    def recv(self, size: int) -> bytes:
        if not self._replies:
            raise BlockingIOError
        return self._replies.pop(0)

    def close(self) -> None:
        self.closed = True


def client_with(replies: list[bytes], variables: tuple[str, ...] = VARIABLES) -> tuple[RtdeClient, ScriptedSocket]:
    socket = ScriptedSocket(replies)
    client = RtdeClient("10.0.0.2", variables=variables, connector=lambda host, port: socket)
    return client, socket


def handshake_replies(types: bytes = TYPES) -> list[bytes]:
    return [
        encode_packet(REQUEST_PROTOCOL_VERSION, b"\x01"),
        encode_packet(CONTROL_PACKAGE_SETUP_OUTPUTS, b"\x01" + types),
        encode_packet(CONTROL_PACKAGE_START, b"\x01"),
    ]


class FramingTests(unittest.TestCase):
    def test_packet_carries_its_own_size(self) -> None:
        packet = encode_packet(CONTROL_PACKAGE_START, b"\x01")

        self.assertEqual(struct.unpack(">HB", packet[:3]), (4, CONTROL_PACKAGE_START))

    def test_decode_splits_complete_packets_and_keeps_the_remainder(self) -> None:
        stream = encode_packet(TEXT_MESSAGE, b"ok") + encode_packet(DATA_PACKAGE, b"\x01")[:2]

        packets, remainder = decode_packets(stream)

        self.assertEqual(packets, [(TEXT_MESSAGE, b"ok")])
        self.assertEqual(len(remainder), 2)

    def test_incomplete_header_is_left_in_the_buffer(self) -> None:
        self.assertEqual(decode_packets(b"\x00"), ([], b"\x00"))


class DecodeValuesTests(unittest.TestCase):
    def test_reads_each_variable_by_its_declared_type(self) -> None:
        payload = struct.pack(">6d", 0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0)
        payload += struct.pack(">i", 7) + struct.pack(">I", 1)

        values = decode_values(payload, (("actual_q", "VECTOR6D"), ("robot_mode", "INT32"), ("safety_status", "UINT32")))

        self.assertAlmostEqual(values["actual_q"][1], -1.5708)
        self.assertEqual(values["robot_mode"], 7)
        self.assertEqual(values["safety_status"], 1)

    def test_unsigned_vectors_decode_as_unsigned(self) -> None:
        payload = struct.pack(">6I", 4_000_000_000, 0, 0, 0, 0, 0)

        values = decode_values(payload, (("digital_inputs", "VECTOR6UINT32"),))

        self.assertEqual(values["digital_inputs"][0], 4_000_000_000)

    def test_signed_vectors_decode_as_signed(self) -> None:
        payload = struct.pack(">6i", -5, 0, 0, 0, 0, 0)

        values = decode_values(payload, (("joint_modes", "VECTOR6INT32"),))

        self.assertEqual(values["joint_modes"][0], -5)

    def test_truncated_payload_stops_without_raising(self) -> None:
        values = decode_values(b"\x00\x00", (("robot_mode", "INT32"),))

        self.assertEqual(values, {})


class HandshakeTests(unittest.TestCase):
    def test_connects_and_requests_the_variables(self) -> None:
        client, socket = client_with(handshake_replies())

        self.assertTrue(client.connect())
        self.assertTrue(client.connected)
        self.assertIn(b"actual_q,robot_mode,safety_status", socket.sent[1])
        self.assertEqual(socket.timeouts[-1], 0.0)

    def test_missing_variable_fails_with_a_reason(self) -> None:
        client, _ = client_with(handshake_replies(b"VECTOR6D,INT32,NOT_FOUND"))

        self.assertFalse(client.connect())
        self.assertIn("safety_status", client.last_error)
        self.assertFalse(client.connected)

    def test_variable_held_by_another_client_is_refused(self) -> None:
        client, _ = client_with(handshake_replies(b"VECTOR6D,INT32,IN_USE"))

        self.assertFalse(client.connect())
        self.assertIn("IN_USE", client.last_error)
        self.assertFalse(client.connected)

    def test_rejected_protocol_version_is_reported(self) -> None:
        client, _ = client_with([encode_packet(REQUEST_PROTOCOL_VERSION, b"\x00")])

        self.assertFalse(client.connect())
        self.assertIn("protocol version", client.last_error)

    def test_unreachable_controller_is_not_fatal(self) -> None:
        def refuse(host: str, port: int) -> None:
            raise OSError("refused")

        client = RtdeClient("10.0.0.2", connector=refuse)

        self.assertFalse(client.connect())
        self.assertIn("refused", client.last_error)


class ReadTests(unittest.TestCase):
    def test_returns_the_newest_sample(self) -> None:
        joints = (0.0, -1.0, 0.5, -1.5, 0.0, 0.25)
        replies = handshake_replies() + [data_package((0.0,) * 6) + data_package(joints)]
        client, _ = client_with(replies)
        client.connect()

        sample = client.read()

        self.assertAlmostEqual(sample["actual_q"][5], 0.25)
        self.assertEqual(sample["robot_mode"], 7)

    def test_no_data_yet_returns_none(self) -> None:
        client, _ = client_with(handshake_replies())
        client.connect()

        self.assertIsNone(client.read())

    def test_closed_connection_drops_the_client(self) -> None:
        client, _ = client_with(handshake_replies() + [b""])
        client.connect()

        self.assertIsNone(client.read())
        self.assertFalse(client.connected)
        self.assertIn("closed", client.last_error)

    def test_reading_without_connecting_returns_none(self) -> None:
        self.assertIsNone(RtdeClient("10.0.0.2").read())


if __name__ == "__main__":
    unittest.main()
