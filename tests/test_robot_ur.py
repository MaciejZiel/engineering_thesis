import math
import unittest

from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.targets import GRIPPER_CLOSE, GRIPPER_OPEN, ArmTargets, JointTargets
from vision_robot_arm.robot.ur_backend import URBackend, encode_gripper, encode_servoj


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeSocket:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.sent: list[bytes] = []
        self.closed = False

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True


class FakeConnector:
    def __init__(self) -> None:
        self.sockets: dict[str, FakeSocket] = {}

    def __call__(self, host: str, port: int) -> FakeSocket:
        sock = FakeSocket(host, port)
        self.sockets[host] = sock
        return sock


def failing_connector(host: str, port: int) -> None:
    raise OSError("timed out")


def targets(right: dict[str, float] | None = None, left: dict[str, float] | None = None,
            right_gripper: str | None = None, ts: int = 1) -> JointTargets:
    arms = {}
    if right is not None or right_gripper is not None:
        arms["right"] = ArmTargets(joints=right or {}, gripper=right_gripper)
    if left is not None:
        arms["left"] = ArmTargets(joints=left)
    return JointTargets(timestamp_ms=ts, arms=arms)


class EncodeTests(unittest.TestCase):
    def test_servoj_uses_ur_joint_order_in_radians_with_home_for_held_joints(self) -> None:
        frame = encode_servoj({"shoulder": -45.0, "elbow": 90.0}, 0.05, 0.1, 300)

        self.assertEqual(
            frame,
            b"servoj([0.0000, -0.7854, 1.5708, -1.5708, 0.0000, 0.0000], 0, 0, 0.050, 0.100, 300)\n",
        )

    def test_servoj_rounds_to_four_decimals(self) -> None:
        frame = encode_servoj({"wrist_1": math.degrees(1.23456)}, 0.008, 0.05, 1000)

        self.assertIn(b"1.2346", frame)
        self.assertTrue(frame.endswith(b", 0, 0, 0.008, 0.050, 1000)\n"))

    def test_gripper_uses_tool_digital_output(self) -> None:
        self.assertEqual(encode_gripper(True), b"set_tool_digital_out(0, True)\n")
        self.assertEqual(encode_gripper(False, tool_output=1), b"set_tool_digital_out(1, False)\n")


class URBackendTests(unittest.TestCase):
    def make_backend(self, connector: FakeConnector, **overrides: object) -> URBackend:
        config = RobotConfig(
            backend="ur",
            right_host="192.168.1.10",
            left_host="192.168.1.11",
            send_interval=1.0,
            **overrides,
        )
        return URBackend(config, connector=connector, clock=FakeClock())

    def test_connects_to_every_configured_arm(self) -> None:
        connector = FakeConnector()

        self.make_backend(connector)

        self.assertEqual(sorted(connector.sockets), ["192.168.1.10", "192.168.1.11"])
        self.assertEqual(connector.sockets["192.168.1.10"].port, 30002)

    def test_connection_failure_exits_with_hint(self) -> None:
        config = RobotConfig(backend="ur", right_host="10.0.0.9")

        with self.assertRaises(SystemExit) as context:
            URBackend(config, connector=failing_connector)

        self.assertIn("10.0.0.9", str(context.exception))
        self.assertIn("Remote Control", str(context.exception))

    def test_sends_servoj_per_arm_and_gripper_once(self) -> None:
        connector = FakeConnector()
        backend = self.make_backend(connector)

        backend.send(targets(right={"shoulder": -45.0}, left={"elbow": 30.0}, right_gripper=GRIPPER_CLOSE))

        right_sent = connector.sockets["192.168.1.10"].sent
        left_sent = connector.sockets["192.168.1.11"].sent
        self.assertTrue(right_sent[0].startswith(b"servoj(["))
        self.assertIn(b"-0.7854", right_sent[0])
        self.assertEqual(right_sent[1], b"set_tool_digital_out(0, True)\n")
        self.assertEqual(len(left_sent), 2)
        self.assertIn(b"0.5236", left_sent[0])
        self.assertEqual(left_sent[1], b"set_tool_digital_out(0, False)\n")

    def test_gripper_command_is_only_sent_on_change(self) -> None:
        connector = FakeConnector()
        backend = self.make_backend(connector)
        clock = backend._clock

        backend.send(targets(right={"shoulder": -45.0}, right_gripper=GRIPPER_CLOSE))
        clock.now = 1.0
        backend.send(targets(right={"shoulder": -40.0}, ts=2))
        clock.now = 2.0
        backend.send(targets(right={"shoulder": -35.0}, right_gripper=GRIPPER_OPEN, ts=3))

        sent = connector.sockets["192.168.1.10"].sent
        gripper_frames = [frame for frame in sent if frame.startswith(b"set_tool")]
        self.assertEqual(gripper_frames, [b"set_tool_digital_out(0, True)\n", b"set_tool_digital_out(0, False)\n"])
        self.assertEqual(len(sent), 5)

    def test_frames_are_rate_limited_and_arms_without_pose_are_skipped(self) -> None:
        connector = FakeConnector()
        backend = self.make_backend(connector)
        clock = backend._clock

        backend.send(targets(right={"shoulder": -45.0}))
        clock.now = 0.5
        backend.send(targets(right={"shoulder": -44.0}, ts=2))

        self.assertEqual(len(connector.sockets["192.168.1.10"].sent), 2)
        self.assertEqual(connector.sockets["192.168.1.11"].sent, [])

    def test_status_lines_and_state_and_close(self) -> None:
        connector = FakeConnector()
        backend = self.make_backend(connector)

        self.assertEqual(
            backend.status_lines(),
            [
                "ur R 192.168.1.10:30002 waiting for pose grip n/a",
                "ur L 192.168.1.11:30002 waiting for pose grip n/a",
            ],
        )
        self.assertIsNone(backend.robot_state())

        backend.send(targets(right={"shoulder": -45.0}, right_gripper=GRIPPER_CLOSE))
        backend.close()

        self.assertEqual(backend.status_lines()[0], "ur R 192.168.1.10:30002 shoulder  -45.0 grip close")
        self.assertEqual(backend.robot_state().arm("right").joints, {"shoulder": -45.0})
        self.assertTrue(connector.sockets["192.168.1.10"].closed)
        self.assertTrue(connector.sockets["192.168.1.11"].closed)


if __name__ == "__main__":
    unittest.main()
