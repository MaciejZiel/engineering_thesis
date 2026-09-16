import math
import unittest

from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.targets import GRIPPER_CLOSE, GRIPPER_OPEN, ArmTargets, JointTargets
from vision_robot_arm.robot.ur_backend import (
    URBackend,
    encode_gripper,
    encode_movej,
    encode_servoj,
    encode_stopj,
)
from vision_robot_arm.robot.ur_dashboard import DashboardStatus


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
        self.fail_on_send = False

    def sendall(self, data: bytes) -> None:
        if self.fail_on_send:
            raise OSError("connection reset")
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True

    def commands(self, prefix: bytes) -> list[bytes]:
        return [frame for frame in self.sent if frame.startswith(prefix)]


class FakeConnector:
    def __init__(self) -> None:
        self.sockets: dict[str, FakeSocket] = {}

    def __call__(self, host: str, port: int) -> FakeSocket:
        sock = FakeSocket(host, port)
        self.sockets[host] = sock
        return sock


class FakeRtde:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.sample: dict | None = None
        self.connected = True
        self.closed = False

    def drop(self) -> None:
        self.sample = None
        self.connected = False

    def read(self) -> dict | None:
        return self.sample

    def close(self) -> None:
        self.closed = True


class FakeRtdeFactory:
    def __init__(self) -> None:
        self.clients: dict[str, FakeRtde] = {}

    def __call__(self, host: str, port: int) -> FakeRtde:
        client = FakeRtde(host, port)
        self.clients[host] = client
        return client


def failing_connector(host: str, port: int) -> None:
    raise OSError("timed out")


def ready_status(host: str, port: int) -> DashboardStatus:
    return DashboardStatus("RUNNING", "NORMAL", True)


def no_status(host: str, port: int) -> None:
    return None


def targets(
    right: dict[str, float] | None = None,
    left: dict[str, float] | None = None,
    right_gripper: str | None = None,
    lift_mode: bool = False,
    ts: int = 1,
) -> JointTargets:
    arms = {}
    if right is not None or right_gripper is not None:
        arms["right"] = ArmTargets(joints=right or {}, gripper=right_gripper)
    if left is not None:
        arms["left"] = ArmTargets(joints=left)
    return JointTargets(timestamp_ms=ts, arms=arms, lift_mode=lift_mode)


class EncodeTests(unittest.TestCase):
    def test_servoj_uses_ur_joint_order_in_radians_with_home_for_held_joints(self) -> None:
        frame = encode_servoj({"shoulder": -45.0, "elbow": 90.0}, 0.05, 0.1, 300)

        self.assertEqual(
            frame,
            b"servoj([0.0000, -0.7854, 1.5708, -1.5708, 0.0000, 0.0000], 0, 0, 0.050, 0.100, 300)\n",
        )

    def test_movej_converts_speed_and_acceleration_to_radians(self) -> None:
        frame = encode_movej({"shoulder": -90.0}, speed_deg_s=30.0, accel_deg_s2=60.0)

        self.assertTrue(frame.startswith(b"movej([0.0000, -1.5708, 0.0000, -1.5708, 0.0000, 0.0000]"))
        self.assertIn(f"a={math.radians(60.0):.3f}".encode(), frame)
        self.assertIn(f"v={math.radians(30.0):.3f}".encode(), frame)

    def test_stopj_decelerates_in_radians(self) -> None:
        self.assertEqual(encode_stopj(120.0), f"stopj({math.radians(120.0):.3f})\n".encode())

    def test_gripper_uses_the_configured_tool_output(self) -> None:
        self.assertEqual(encode_gripper(True), b"set_tool_digital_out(0, True)\n")
        self.assertEqual(encode_gripper(False, tool_output=1), b"set_tool_digital_out(1, False)\n")


class URBackendTests(unittest.TestCase):
    def make_backend(
        self,
        connector: FakeConnector,
        rtde_factory: FakeRtdeFactory | None = None,
        clock: FakeClock | None = None,
        **overrides: object,
    ) -> tuple[URBackend, FakeClock]:
        clock = clock or FakeClock()
        settings = {
            "backend": "ur",
            "right_host": "192.168.1.10",
            "left_host": "192.168.1.11",
            "send_interval": 1.0,
            "max_speed_deg_s": 90.0,
            "start_seconds": 2.0,
            "preflight": False,
        }
        settings.update(overrides)
        backend = URBackend(
            RobotConfig(**settings),
            connector=connector,
            clock=clock,
            rtde_factory=rtde_factory,
            status_query=no_status,
        )
        return backend, clock

    def test_connects_to_every_arm_and_homes_first(self) -> None:
        connector = FakeConnector()

        self.make_backend(connector)

        self.assertEqual(sorted(connector.sockets), ["192.168.1.10", "192.168.1.11"])
        self.assertEqual(connector.sockets["192.168.1.10"].port, 30002)
        for sock in connector.sockets.values():
            self.assertEqual(len(sock.commands(b"movej(")), 1)

    def test_connection_failure_exits_with_hint(self) -> None:
        config = RobotConfig(backend="ur", right_host="10.0.0.9", preflight=False)

        with self.assertRaises(SystemExit) as context:
            URBackend(config, connector=failing_connector, rtde_factory=None, status_query=no_status)

        self.assertIn("10.0.0.9", str(context.exception))
        self.assertIn("Remote Control", str(context.exception))

    def test_no_servoj_while_the_arm_is_still_homing(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make_backend(connector)

        clock.now = 1.0
        backend.send(targets(right={"shoulder": -45.0}))

        self.assertEqual(connector.sockets["192.168.1.10"].commands(b"servoj("), [])

    def test_setpoint_waits_at_home_while_the_arm_is_homing(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make_backend(connector, send_interval=0.1, max_speed_deg_s=60.0)

        for step in range(1, 21):
            clock.now = 0.1 * step
            backend.send(targets(right={"shoulder": 0.0}))
        clock.now = 2.1
        backend.send(targets(right={"shoulder": 0.0}, ts=2))

        first = connector.sockets["192.168.1.10"].commands(b"servoj(")[0]
        shoulder = math.degrees(float(first.split(b"[")[1].split(b",")[1]))
        self.assertAlmostEqual(shoulder, -90.0 + 60.0 * 0.1, delta=0.5)

    def test_second_arm_failure_closes_the_first_connection(self) -> None:
        opened: list[FakeSocket] = []

        def connector(host: str, port: int) -> FakeSocket:
            if host.endswith(".11"):
                raise OSError("timed out")
            sock = FakeSocket(host, port)
            opened.append(sock)
            return sock

        with self.assertRaises(SystemExit):
            URBackend(
                RobotConfig(
                    backend="ur",
                    right_host="192.168.1.10",
                    left_host="192.168.1.11",
                    preflight=False,
                ),
                connector=connector,
                rtde_factory=None,
                status_query=no_status,
            )

        self.assertTrue(opened[0].closed)

    def test_setpoints_ramp_at_the_configured_speed_instead_of_jumping(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make_backend(connector, max_speed_deg_s=30.0)

        clock.now = 3.0
        backend.send(targets(right={"elbow": 90.0}))
        clock.now = 4.0
        backend.send(targets(right={"elbow": 90.0}, ts=2))

        state = backend.robot_state()
        self.assertAlmostEqual(state.arm("right").joints["elbow"], 60.0)
        self.assertEqual(state.arm("right").targets["elbow"], 90.0)

    def test_servoj_carries_the_ramped_setpoint(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make_backend(connector, max_speed_deg_s=10.0)

        clock.now = 3.0
        backend.send(targets(right={"shoulder": 0.0}))

        frame = connector.sockets["192.168.1.10"].commands(b"servoj(")[0]
        radians = float(frame.split(b"[")[1].split(b",")[1])
        self.assertAlmostEqual(math.degrees(radians), -80.0, delta=0.01)

    def test_frames_are_rate_limited_and_untracked_arms_hold_position(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make_backend(connector)

        clock.now = 3.0
        backend.send(targets(right={"shoulder": -45.0}))
        clock.now = 3.5
        backend.send(targets(right={"shoulder": -44.0}, ts=2))

        self.assertEqual(len(connector.sockets["192.168.1.10"].commands(b"servoj(")), 1)
        left = connector.sockets["192.168.1.11"]
        self.assertEqual(len(left.commands(b"servoj(")), 1)
        self.assertIn(b"-1.5708", left.commands(b"servoj(")[0])

    def test_gripper_command_is_only_sent_on_change(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make_backend(connector, tool_output=1)

        clock.now = 3.0
        backend.send(targets(right={"shoulder": -45.0}, right_gripper=GRIPPER_CLOSE))
        clock.now = 4.0
        backend.send(targets(right={"shoulder": -45.0}, right_gripper=GRIPPER_CLOSE, ts=2))
        clock.now = 5.0
        backend.send(targets(right={"shoulder": -45.0}, right_gripper=GRIPPER_OPEN, ts=3))

        self.assertEqual(
            connector.sockets["192.168.1.10"].commands(b"set_tool_digital_out"),
            [b"set_tool_digital_out(1, True)\n", b"set_tool_digital_out(1, False)\n"],
        )

    def test_state_and_status_report_actual_joints_from_rtde(self) -> None:
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        backend, clock = self.make_backend(connector, rtde_factory=factory)
        factory.clients["192.168.1.10"].sample = {
            "actual_q": (0.0, math.radians(-89.5), 0.0, math.radians(-90.0), 0.0, 0.0),
            "robot_mode": 7,
            "safety_status": 1,
        }

        clock.now = 3.0
        backend.send(targets(right={"shoulder": -45.0}))

        state = backend.robot_state()
        self.assertAlmostEqual(state.arm("right").joints["shoulder"], -89.5, places=3)
        self.assertEqual(state.arm("right").targets["shoulder"], -45.0)
        self.assertIn("RUNNING/NORMAL", backend.status_lines()[0])

    def test_status_says_open_loop_without_feedback(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make_backend(connector)

        line = backend.status_lines()[0]

        self.assertIn("192.168.1.10", line)
        self.assertIn("open-loop", line)
        self.assertIn("homing", line)

    def test_lift_mode_is_reported_in_the_robot_state(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make_backend(connector)

        clock.now = 3.0
        backend.send(targets(right={"shoulder": -45.0}, lift_mode=True))

        self.assertTrue(backend.robot_state().lift_mode)

    def test_close_stops_the_arms_and_releases_sockets(self) -> None:
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        backend, _ = self.make_backend(connector, rtde_factory=factory)

        backend.close()

        for sock in connector.sockets.values():
            self.assertEqual(len(sock.commands(b"stopj(")), 1)
            self.assertTrue(sock.closed)
        for client in factory.clients.values():
            self.assertTrue(client.closed)

    def test_a_dropped_connection_exits_with_a_readable_message(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make_backend(connector)
        connector.sockets["192.168.1.10"].fail_on_send = True

        clock.now = 3.0
        with self.assertRaises(SystemExit) as context:
            backend.send(targets(right={"shoulder": -45.0}))

        self.assertIn("Lost the connection", str(context.exception))


class SafetyTests(unittest.TestCase):
    """Failure modes that would put a real arm at risk."""

    def make(self, connector: FakeConnector, factory: FakeRtdeFactory | None = None,
             **overrides: object) -> tuple[URBackend, FakeClock]:
        clock = FakeClock()
        settings = {
            "backend": "ur",
            "right_host": "192.168.1.10",
            "left_host": "192.168.1.11",
            "send_interval": 0.05,
            "max_speed_deg_s": 60.0,
            "start_seconds": 1.0,
            "preflight": False,
        }
        settings.update(overrides)
        backend = URBackend(
            RobotConfig(**settings),
            connector=connector,
            clock=clock,
            rtde_factory=factory,
            status_query=no_status,
        )
        return backend, clock

    def test_a_pose_gap_does_not_buy_one_giant_catch_up_step(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make(connector)

        clock.now = 2.0
        backend.send(targets(right={"elbow": 0.0}))
        clock.now = 12.0  # the person was out of frame for ten seconds
        backend.send(targets(right={"elbow": 150.0}, ts=2))

        elbow = backend.robot_state().arm("right").joints["elbow"]
        self.assertLessEqual(elbow, 60.0 * 0.05 * 3 + 0.001)

    def test_a_gripper_command_between_two_frames_still_reaches_the_arm(self) -> None:
        connector = FakeConnector()
        backend, clock = self.make(connector, start_seconds=0.0)

        for frame in range(6):  # 30 fps into a 20 Hz link: half the frames are skipped
            clock.now = 0.001 + frame * 0.0333
            gripper = GRIPPER_CLOSE if frame in (1, 3) else None
            backend.send(targets(right={"shoulder": -90.0 + frame}, right_gripper=gripper))

        self.assertEqual(
            connector.sockets["192.168.1.10"].commands(b"set_tool_digital_out"),
            [b"set_tool_digital_out(0, True)\n"],
        )

    def test_close_stops_every_arm_even_when_one_socket_is_dead(self) -> None:
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        backend, _ = self.make(connector, factory)
        connector.sockets["192.168.1.10"].fail_on_send = True

        backend.close()

        self.assertEqual(len(connector.sockets["192.168.1.11"].commands(b"stopj(")), 1)
        for sock in connector.sockets.values():
            self.assertTrue(sock.closed)
        for client in factory.clients.values():
            self.assertTrue(client.closed)

    def test_homing_waits_until_feedback_says_the_arm_arrived(self) -> None:
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        backend, clock = self.make(connector, factory)
        far = (0.0, math.radians(-20.0), 0.0, math.radians(-90.0), 0.0, 0.0)
        factory.clients["192.168.1.10"].sample = {"actual_q": far}

        clock.now = 1.5  # past start_seconds, but the movej is still running
        backend.send(targets(right={"shoulder": 0.0}))

        self.assertEqual(connector.sockets["192.168.1.10"].commands(b"servoj("), [])

        home = (0.0, math.radians(-90.0), 0.0, math.radians(-90.0), 0.0, 0.0)
        factory.clients["192.168.1.10"].sample = {"actual_q": home}
        clock.now = 2.0
        backend.send(targets(right={"shoulder": 0.0}, ts=2))

        self.assertEqual(len(connector.sockets["192.168.1.10"].commands(b"servoj(")), 1)

    def test_servo_lag_does_not_send_the_arm_back_into_homing(self) -> None:
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        backend, clock = self.make(connector, factory)
        client = factory.clients["192.168.1.10"]

        def actual(shoulder_deg: float) -> dict:
            return {"actual_q": (0.0, math.radians(shoulder_deg), 0.0, math.radians(-90.0), 0.0, 0.0)}

        client.sample = actual(-90.0)
        clock.now = 1.1
        backend.send(targets(right={"shoulder": 0.0}))

        for step in range(1, 30):
            clock.now = 1.1 + 0.06 * step
            client.sample = actual(-90.0 + 0.6 * step)  # the arm trails the setpoint, as servos do
            backend.send(targets(right={"shoulder": 0.0}, ts=step + 1))

        self.assertEqual(len(connector.sockets["192.168.1.10"].commands(b"servoj(")), 30)

    def test_homing_gives_up_after_its_deadline(self) -> None:
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        backend, clock = self.make(connector, factory)
        factory.clients["192.168.1.10"].sample = {
            "actual_q": (0.0, math.radians(-20.0), 0.0, math.radians(-90.0), 0.0, 0.0)
        }

        clock.now = 99.0
        backend.send(targets(right={"shoulder": -90.0}))

        self.assertEqual(len(connector.sockets["192.168.1.10"].commands(b"servoj(")), 1)

    def test_lost_feedback_stops_presenting_a_stale_pose(self) -> None:
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        backend, clock = self.make(connector, factory)
        client = factory.clients["192.168.1.10"]
        client.sample = {
            "actual_q": (0.0, math.radians(-30.0), 0.0, math.radians(-90.0), 0.0, 0.0),
            "robot_mode": 7,
            "safety_status": 1,
        }
        clock.now = 2.0
        backend.send(targets(right={"shoulder": -80.0}))
        self.assertAlmostEqual(backend.robot_state().arm("right").joints["shoulder"], -30.0, places=3)

        client.drop()
        clock.now = 2.5
        backend.send(targets(right={"shoulder": -80.0}, ts=2))

        state = backend.robot_state()
        self.assertNotAlmostEqual(state.arm("right").joints["shoulder"], -30.0, places=3)
        self.assertIn("feedback-lost", backend.status_lines()[0])

    def test_a_failed_first_command_releases_the_socket(self) -> None:
        opened: list[FakeSocket] = []
        clients: list[FakeRtde] = []

        def connector(host: str, port: int) -> FakeSocket:
            sock = FakeSocket(host, port)
            sock.fail_on_send = True
            opened.append(sock)
            return sock

        def factory(host: str, port: int) -> FakeRtde:
            client = FakeRtde(host, port)
            clients.append(client)
            return client

        with self.assertRaises(SystemExit):
            URBackend(
                RobotConfig(backend="ur", right_host="10.0.0.2", preflight=False),
                connector=connector,
                rtde_factory=factory,
                status_query=no_status,
            )

        self.assertTrue(opened[0].closed)
        self.assertTrue(clients[0].closed)


class PreflightTests(unittest.TestCase):
    def config(self) -> RobotConfig:
        return RobotConfig(backend="ur", right_host="10.0.0.2", preflight=True)

    def test_local_control_is_refused_with_an_explanation(self) -> None:
        def local(host: str, port: int) -> DashboardStatus:
            return DashboardStatus("RUNNING", "NORMAL", False)

        with self.assertRaises(SystemExit) as context:
            URBackend(self.config(), connector=FakeConnector(), rtde_factory=None, status_query=local)

        message = str(context.exception)
        self.assertIn("Local control", message)
        self.assertIn("--no-robot-preflight", message)

    def test_unreachable_dashboard_does_not_block_startup(self) -> None:
        connector = FakeConnector()

        URBackend(self.config(), connector=connector, rtde_factory=None, status_query=no_status)

        self.assertIn("10.0.0.2", connector.sockets)

    def test_ready_robot_starts(self) -> None:
        connector = FakeConnector()

        URBackend(self.config(), connector=connector, rtde_factory=None, status_query=ready_status)

        self.assertIn("10.0.0.2", connector.sockets)


if __name__ == "__main__":
    unittest.main()
