import math
import json
import tempfile
import unittest
from pathlib import Path

from vision_robot_arm.robot.config import (
    OPERATION_COMMISSIONING,
    OPERATION_TRACKING,
    RobotConfig,
)
from vision_robot_arm.robot.targets import GRIPPER_CLOSE, GRIPPER_OPEN, ArmTargets, JointTargets
from vision_robot_arm.robot.ur_backend import (
    URBackend,
    ControlFault,
    encode_gripper,
    encode_robotiq_gripper,
    encode_servoj,
    encode_speedj,
    encode_speedj_vector,
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


def reduced_status(host: str, port: int) -> DashboardStatus:
    return DashboardStatus("RUNNING", "REDUCED", True)


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
    def test_encodes_single_joint_speed_command(self) -> None:
        self.assertEqual(
            encode_speedj("shoulder", 0.5, 0.15),
            b"speedj([0.00000, 0.00873, 0.00000, 0.00000, 0.00000, 0.00000], 0.17453, 0.150)\n",
        )

    def test_encodes_multiple_joint_velocities_in_one_command(self) -> None:
        command = encode_speedj_vector(
            {"base": -2.0, "elbow": 3.0, "wrist_3": 1.0}, 0.15
        )

        values = command.split(b"[")[1].split(b"]")[0].split(b", ")
        self.assertAlmostEqual(math.degrees(float(values[0])), -2.0, delta=0.01)
        self.assertAlmostEqual(math.degrees(float(values[2])), 3.0, delta=0.01)
        self.assertAlmostEqual(math.degrees(float(values[5])), 1.0, delta=0.01)

    def test_servoj_uses_ur_joint_order_in_radians_with_home_for_held_joints(self) -> None:
        frame = encode_servoj({"shoulder": -45.0, "elbow": 90.0}, 0.05, 0.1, 300)

        self.assertEqual(
            frame,
            b"servoj([0.0000, -0.7854, 1.5708, -1.5708, 0.0000, 0.0000], 0, 0, 0.050, 0.100, 300)\n",
        )

    def test_stopj_decelerates_in_radians(self) -> None:
        self.assertEqual(encode_stopj(120.0), f"stopj({math.radians(120.0):.3f})\n".encode())

    def test_gripper_uses_the_configured_tool_output(self) -> None:
        self.assertEqual(encode_gripper(True), b"set_tool_digital_out(0, True)\n")
        self.assertEqual(encode_gripper(False, tool_output=1), b"set_tool_digital_out(1, False)\n")

    def test_robotiq_gripper_uses_urcap_socket_with_scaled_settings(self) -> None:
        opened = encode_robotiq_gripper(False, speed_percent=80, force_percent=50)
        closed = encode_robotiq_gripper(True, speed_percent=80, force_percent=50)

        self.assertIn(b'socket_open("127.0.0.1", 63352', opened)
        self.assertIn(b'socket_set_var("SPE", 204', opened)
        self.assertIn(b'socket_set_var("FOR", 127', opened)
        self.assertIn(b'socket_set_var("POS", 0', opened)
        self.assertIn(b'socket_set_var("POS", 255', closed)


class URBackendTests(unittest.TestCase):
    def test_default_construction_never_homes(self):
        connector = FakeConnector()
        backend = URBackend(
            RobotConfig(backend="ur", right_host="test", preflight=False),
            connector=connector, rtde_factory=None,
        )
        backend.send(targets(right={"shoulder": -80.0}))
        self.assertEqual(connector.sockets["test"].sent, [])
        backend.close()

    def test_required_feedback_failure_closes_connection_without_homing(self):
        connector = FakeConnector()
        with self.assertRaises(ControlFault):
            URBackend(RobotConfig(backend="ur", right_host="test", preflight=False),
                      connector=connector, rtde_factory=None,
                      require_feedback=True)
        self.assertEqual(connector.sockets["test"].commands(b"movej("), [])
        self.assertTrue(connector.sockets["test"].closed)

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
        for arm in backend._arms.values():
            arm._armed = True
        return backend, clock

    def test_connects_to_every_arm_without_sending_motion(self) -> None:
        connector = FakeConnector()

        self.make_backend(connector)

        self.assertEqual(sorted(connector.sockets), ["192.168.1.10", "192.168.1.11"])
        self.assertEqual(connector.sockets["192.168.1.10"].port, 30002)
        for sock in connector.sockets.values():
            self.assertEqual(sock.sent, [])

    def test_connection_failure_exits_with_hint(self) -> None:
        config = RobotConfig(backend="ur", right_host="10.0.0.9", preflight=False)

        with self.assertRaises(SystemExit) as context:
            URBackend(config, connector=failing_connector, rtde_factory=None, status_query=no_status)

        self.assertIn("10.0.0.9", str(context.exception))
        self.assertIn("Remote Control", str(context.exception))

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
        for host, shoulder in (("192.168.1.10", -89.5), ("192.168.1.11", -70.0)):
            factory.clients[host].sample = {
                "actual_q": (0.0, math.radians(shoulder), 0.0, math.radians(-90.0), 0.0, 0.0),
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
        self.assertIn("S", line)

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
        for arm in backend._arms.values():
            arm._armed = True
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
        backend, clock = self.make(connector)

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

    def test_silent_connected_feedback_expires_and_blocks_motion(self) -> None:
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        backend, clock = self.make(connector, factory)
        for client in factory.clients.values():
            client.sample = {"actual_q": (0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0)}
        clock.now = 1.1
        backend.send(targets(right={"shoulder": -80.0}))
        for client in factory.clients.values():
            client.sample = None  # Socket remains connected, but packets stop arriving.
        clock.now = 1.7
        with self.assertRaises(ControlFault):
            backend.send(targets(right={"shoulder": -70.0}))
        for sock in connector.sockets.values():
            self.assertTrue(sock.closed)
            self.assertEqual(len(sock.commands(b"servoj(")), 1)
        with self.assertRaises(ControlFault):
            backend.send(targets(right={"shoulder": -70.0}))

    def test_controller_safety_fault_blocks_both_arms(self) -> None:
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        backend, clock = self.make(connector, factory)
        factory.clients["192.168.1.11"].sample = {"safety_status": 3}
        clock.now = 1.1
        with self.assertRaises(ControlFault):
            backend.send(targets(right={"shoulder": -80.0}))
        for sock in connector.sockets.values():
            self.assertEqual(sock.commands(b"servoj("), [])
            self.assertTrue(sock.closed)

    def test_lost_feedback_stops_presenting_a_stale_pose(self) -> None:
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        backend, clock = self.make(connector, factory)
        client = factory.clients["192.168.1.10"]
        for host in ("192.168.1.10", "192.168.1.11"):
            factory.clients[host].sample = {
                "actual_q": (0.0, math.radians(-30.0), 0.0, math.radians(-90.0), 0.0, 0.0),
                "robot_mode": 7,
                "safety_status": 1,
            }
        clock.now = 2.0
        backend.send(targets(right={"shoulder": -80.0}))
        self.assertAlmostEqual(backend.robot_state().arm("right").joints["shoulder"], -30.0, places=3)

        client.drop()
        clock.now = 2.5
        with self.assertRaises(ControlFault):
            backend.send(targets(right={"shoulder": -80.0}, ts=2))

        state = backend.robot_state()
        self.assertNotAlmostEqual(state.arm("right").joints["shoulder"], -30.0, places=3)
        self.assertIn("feedback-lost", backend.status_lines()[0])

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

    def test_unreachable_dashboard_blocks_before_opening_command_connection(self) -> None:
        connector = FakeConnector()

        with self.assertRaisesRegex(SystemExit, "dashboard unavailable"):
            URBackend(self.config(), connector=connector, rtde_factory=None, status_query=no_status)

        self.assertEqual(connector.sockets, {})

    def test_ready_robot_starts(self) -> None:
        connector = FakeConnector()

        URBackend(self.config(), connector=connector, rtde_factory=None, status_query=ready_status)

        self.assertIn("10.0.0.2", connector.sockets)


class CommissioningTests(unittest.TestCase):
    def test_position_move_is_relative_bounded_and_requires_refresh(self):
        backend, socket, clock = self.make()
        backend.arm_commissioning()
        backend.start_joint_positions({"shoulder": 1.0, "elbow": -1.0}, {"shoulder": 2.0, "elbow": 1.0}, relative=True)
        self.assertEqual(socket.sent, [])
        self.assertAlmostEqual(backend._position_targets["shoulder"], -39)
        clock.now = .05
        backend.robot_state()
        self.assertEqual(len(socket.commands(b"servoj(")), 1)
        command = socket.commands(b"servoj(")[0]
        values = [float(v) for v in command.split(b"[")[1].split(b"]")[0].split(b",")]
        self.assertEqual(values[0], 0)
        self.assertEqual(values[4:], [0, 0])
        self.assertGreater(math.degrees(values[1]), -40)
        self.assertLess(math.degrees(values[2]), 20)
        clock.now = .16
        backend.robot_state()
        self.assertFalse(backend.position_move_active)
        self.assertEqual(len(socket.commands(b"stopj(")), 1)

    def test_position_move_rejects_entire_request_if_one_joint_is_invalid(self):
        backend, socket, clock = self.make()
        backend.arm_commissioning()
        for joints in ({"shoulder": -39, "elbow": 30}, {"shoulder": float("nan")}):
            with self.assertRaises(ValueError):
                backend.start_joint_positions(joints, {j: 2 for j in joints})
        self.assertEqual(socket.sent, [])
        self.assertFalse(backend.position_move_active)

    def test_position_move_stops_once_feedback_reaches_destination(self):
        backend, socket, clock = self.make()
        backend.arm_commissioning()
        backend.start_joint_positions({"shoulder": -39}, {"shoulder": 2})
        backend._arms["right"]._rtde.sample["actual_q"] = (0, math.radians(-39), math.radians(20), math.radians(-80), 0, 0)
        clock.now = .05
        backend.robot_state()
        self.assertFalse(backend.position_move_active)
        self.assertEqual(len(socket.commands(b"stopj(")), 1)

    def make(self):
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        clock = FakeClock()
        config = RobotConfig(
            backend="ur",
            operation=OPERATION_COMMISSIONING,
            right_host="10.0.0.2",
            send_interval=0.05,
            commissioning_speed_deg_s=2.0,
            commissioning_excursion_deg=2.0,
            commissioning_watchdog_s=0.15,
        )
        backend = URBackend(
            config,
            connector=connector,
            rtde_factory=factory,
            status_query=reduced_status,
            clock=clock,
            require_feedback=True,
        )
        factory.clients["10.0.0.2"].sample = {
            "actual_q": (
                0.0,
                math.radians(-40.0),
                math.radians(20.0),
                math.radians(-80.0),
                0.0,
                0.0,
            ),
            "actual_qd": (0.0,) * 6,
            "robot_mode": 7,
            "safety_status": 2,
        }
        return backend, connector.sockets["10.0.0.2"], clock

    def test_arming_captures_feedback_without_sending_motion(self) -> None:
        backend, socket, _ = self.make()

        backend.arm_commissioning()

        self.assertEqual(socket.sent, [])
        self.assertAlmostEqual(
            backend.robot_state().arm("right").targets["shoulder"], -40.0
        )

    def test_jog_is_slow_bounded_and_stops_when_refresh_expires(self) -> None:
        backend, socket, clock = self.make()
        backend.arm_commissioning()
        backend.refresh_jog(1)

        clock.now = 0.05
        backend.robot_state()

        commands = socket.commands(b"speedj(")
        self.assertEqual(len(commands), 1)
        shoulder_speed = math.degrees(
            float(commands[0].split(b"[")[1].split(b",")[1])
        )
        self.assertAlmostEqual(shoulder_speed, 2.0, delta=0.01)
        self.assertEqual(socket.commands(b"movej("), [])

        clock.now = 0.16
        backend.robot_state()
        self.assertEqual(len(socket.commands(b"stopj(")), 1)

    def test_multiple_joints_share_one_speed_command(self) -> None:
        backend, socket, clock = self.make()
        backend.arm_commissioning()
        backend.refresh_joint_jogs({"base": -1.0, "elbow": 2.5, "wrist_2": 0.5})

        clock.now = 0.05
        backend.robot_state()

        commands = socket.commands(b"speedj(")
        self.assertEqual(len(commands), 1)
        values = commands[0].split(b"[")[1].split(b"]")[0].split(b", ")
        self.assertAlmostEqual(math.degrees(float(values[0])), -1.0, delta=0.01)
        self.assertAlmostEqual(math.degrees(float(values[2])), 2.5, delta=0.01)
        self.assertAlmostEqual(math.degrees(float(values[4])), 0.5, delta=0.01)

    def test_normal_safety_mode_is_refused_before_command_socket_opens(self) -> None:
        connector = FakeConnector()
        config = RobotConfig(
            backend="ur",
            operation=OPERATION_COMMISSIONING,
            right_host="10.0.0.2",
        )

        with self.assertRaisesRegex(SystemExit, "REDUCED"):
            URBackend(
                config,
                connector=connector,
                rtde_factory=FakeRtdeFactory(),
                status_query=ready_status,
                require_feedback=True,
            )

        self.assertEqual(connector.sockets, {})

    def test_moving_robot_cannot_be_armed(self) -> None:
        backend, _, _ = self.make()
        backend._arms["right"]._rtde.sample["actual_qd"] = (
            0.0,
            math.radians(1.0),
            0.0,
            0.0,
            0.0,
            0.0,
        )

        with self.assertRaisesRegex(ControlFault, "stationary"):
            backend.arm_commissioning()


class TrackingArmingTests(unittest.TestCase):
    def test_velocity_accumulates_and_reversal_never_teleports(self):
        backend, _ = self.make()
        backend.arm_tracking()
        arm = backend._arms["right"]
        arm._setpoints.targets["shoulder"] = -10
        for _ in range(10):
            arm._step_tracking_setpoints(.05)
        self.assertAlmostEqual(arm._tracking_velocities["shoulder"], 3.5)
        before = arm._setpoints.joints["shoulder"]
        arm._setpoints.targets["shoulder"] = -60
        arm._step_tracking_setpoints(.05)
        self.assertLess(abs(arm._setpoints.joints["shoulder"] - before), .2)
        self.assertNotEqual(arm._setpoints.joints["shoulder"], -60)
    def make(self, velocity: float = 0.0, **overrides):
        connector = FakeConnector()
        factory = FakeRtdeFactory()
        settings = dict(
            backend="ur",
            operation=OPERATION_TRACKING,
            right_host="10.0.0.2",
        )
        settings.update(overrides)
        config = RobotConfig(**settings)
        backend = URBackend(
            config,
            connector=connector,
            rtde_factory=factory,
            status_query=ready_status,
            require_feedback=True,
        )
        factory.clients["10.0.0.2"].sample = {
            "actual_q": (0.0, math.radians(-37.0), 0.0, 0.0, 0.0, 0.0),
            "actual_qd": (0.0, math.radians(velocity), 0.0, 0.0, 0.0, 0.0),
            "robot_mode": 7,
            "safety_status": 1,
        }
        return backend, connector.sockets["10.0.0.2"]

    def test_arming_tracking_captures_feedback_without_motion(self) -> None:
        backend, socket = self.make()

        backend.arm_tracking()

        self.assertEqual(socket.sent, [])
        self.assertAlmostEqual(
            backend.robot_state().arm("right").targets["shoulder"], -37.0
        )
        self.assertTrue(backend.ready())

    def test_moving_robot_cannot_be_armed_for_tracking(self) -> None:
        backend, socket = self.make(velocity=1.0)

        with self.assertRaisesRegex(ControlFault, "stationary"):
            backend.arm_tracking()

        self.assertEqual(socket.commands(b"movej("), [])

    def test_tracking_targets_are_bounded_from_the_captured_pose(self) -> None:
        backend, _ = self.make()
        backend.arm_tracking()

        bounded = backend._bounded_tracking_targets(
            "right", ArmTargets(joints={"shoulder": 120.0, "elbow": -100.0})
        )

        self.assertEqual(bounded.joints["shoulder"], 3.0)
        self.assertEqual(bounded.joints["elbow"], -40.0)

    def test_tracking_accelerates_instead_of_starting_at_full_speed(self) -> None:
        backend, socket = self.make()
        backend.arm_tracking()

        backend.send(targets(right={"shoulder": 3.0}))

        command = socket.commands(b"servoj(")[0]
        shoulder = math.degrees(float(command.split(b"[")[1].split(b",")[1]))
        self.assertGreater(shoulder, -37.0)
        self.assertLess(shoulder, -36.9)

    def test_tracking_telemetry_records_targets_setpoint_and_rtde(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracking.jsonl"
            backend, _ = self.make(telemetry_log_path=str(path))
            backend.arm_tracking()
            backend.send(targets(right={"shoulder": 3.0}))
            backend.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]

        sample = next(record for record in records if record["event"] == "tracking_sample")
        self.assertEqual(sample["robot_mode"], "RUNNING")
        self.assertEqual(sample["safety_status"], "NORMAL")
        self.assertIn("shoulder", sample["raw_target_deg"])
        self.assertIn("shoulder", sample["sent_setpoint_deg"])
        self.assertEqual(sample["settings"]["excursion_deg"], 40.0)

    def test_robot_feedback_is_written_to_a_separate_rtde_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot-feedback.jsonl"
            backend, _ = self.make(feedback_log_path=str(path))
            backend.arm_tracking()
            backend.send(targets(right={"shoulder": 3.0}))
            backend.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]

        sample = next(record for record in records if record["event"] == "rtde_sample")
        self.assertEqual(sample["arm"], "right")
        self.assertEqual(sample["robot_mode_name"], "RUNNING")
        self.assertEqual(sample["safety_status_name"], "NORMAL")
        self.assertIn("shoulder", sample["actual_joint_deg"])
        self.assertIn("actual_tcp_pose", sample)
        self.assertEqual(records[0]["event"], "session_started")
        self.assertEqual(records[-1]["event"], "session_closed")


if __name__ == "__main__":
    unittest.main()
