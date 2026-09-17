import unittest

from vision_robot_arm.robot.serial_backend import (
    SerialBackend,
    SerialWriteError,
    WRITE_TIMEOUT_S,
    encode_targets,
    load_serial_module,
)
from vision_robot_arm.robot.targets import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    ArmTargets,
    JointTargets,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeSerialPort:
    def __init__(self, port: str, baud_rate: int, timeout: float, write_timeout: float) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.written: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class FakeSerialModule:
    def __init__(self) -> None:
        self.ports: list[FakeSerialPort] = []

    def Serial(self, port: str, baud_rate: int, timeout: float, write_timeout: float) -> FakeSerialPort:
        fake = FakeSerialPort(port, baud_rate, timeout, write_timeout)
        self.ports.append(fake)
        return fake


class FailingSerialModule:
    def Serial(self, port: str, baud_rate: int, timeout: float, write_timeout: float) -> None:
        raise OSError("device not found")


def right(joints: dict[str, float] | None = None, gripper: str | None = None) -> JointTargets:
    return JointTargets(timestamp_ms=1, arms={"right": ArmTargets(joints=joints or {}, gripper=gripper)})


class EncodeTargetsTests(unittest.TestCase):
    def test_encodes_both_arms_grippers_and_lift_mode(self) -> None:
        targets = JointTargets(
            timestamp_ms=1,
            arms={
                "right": ArmTargets(
                    joints={"shoulder": 90.0, "elbow": 45.0, "wrist_1": 120.0},
                    gripper=GRIPPER_CLOSE,
                ),
                "left": ArmTargets(joints={"shoulder": 30.0}, gripper=GRIPPER_OPEN),
            },
            lift_mode=False,
        )

        self.assertEqual(
            encode_targets(targets),
            b"RS:90.0;RE:45.0;RW:120.0;RG:1;LS:30.0;LG:0;L:0\n",
        )

    def test_omits_gripper_when_not_commanded(self) -> None:
        targets = JointTargets(
            timestamp_ms=1,
            arms={"left": ArmTargets(joints={"elbow": 10.0})},
            lift_mode=True,
        )

        self.assertEqual(encode_targets(targets), b"LE:10.0;L:1\n")

    def test_unknown_joints_are_skipped(self) -> None:
        self.assertEqual(encode_targets(right({"finger": 10.0, "shoulder": 1.0})), b"RS:1.0;L:0\n")

    def test_empty_targets_still_carry_lift_mode(self) -> None:
        self.assertEqual(encode_targets(JointTargets(timestamp_ms=1)), b"L:0\n")


class LoadSerialModuleTests(unittest.TestCase):
    def test_missing_pyserial_gives_install_hint(self) -> None:
        def failing_importer(name: str) -> None:
            raise ImportError(name)

        with self.assertRaises(SystemExit) as context:
            load_serial_module(importer=failing_importer)

        self.assertIn("pip install pyserial", str(context.exception))

    def test_returns_module_from_importer(self) -> None:
        sentinel = object()

        self.assertIs(load_serial_module(importer=lambda name: sentinel), sentinel)


class SerialBackendTests(unittest.TestCase):
    def test_opens_port_with_configured_settings(self) -> None:
        module = FakeSerialModule()

        SerialBackend("COM3", 9600, 0.05, serial_module=module, clock=FakeClock())

        self.assertEqual(module.ports[0].port, "COM3")
        self.assertEqual(module.ports[0].baud_rate, 9600)
        self.assertEqual(module.ports[0].timeout, 0)
        self.assertEqual(module.ports[0].write_timeout, WRITE_TIMEOUT_S)

    def test_writes_frames_rate_limited(self) -> None:
        module = FakeSerialModule()
        clock = FakeClock()
        backend = SerialBackend("COM3", 9600, 1.0, serial_module=module, clock=clock)
        targets = right({"shoulder": 90.0})

        backend.send(targets)
        clock.now = 0.5
        backend.send(targets)
        clock.now = 1.0
        backend.send(targets)

        self.assertEqual(module.ports[0].written, [b"RS:90.0;L:0\n", b"RS:90.0;L:0\n"])

    def test_a_gripper_command_between_two_frames_still_reaches_the_device(self) -> None:
        module = FakeSerialModule()
        clock = FakeClock()
        backend = SerialBackend("COM3", 9600, 0.05, serial_module=module, clock=clock)

        for frame in range(6):  # 30 fps into a 20 Hz link: half the frames are skipped
            clock.now = frame * 0.0333
            gripper = GRIPPER_CLOSE if frame in (1, 3) else None
            backend.send(right({"shoulder": -90.0 + frame}, gripper=gripper))

        self.assertTrue(any(b"RG:1" in frame for frame in module.ports[0].written))

    def test_a_cleared_lift_mode_is_transmitted_after_tracking_drops(self) -> None:
        module = FakeSerialModule()
        clock = FakeClock()
        backend = SerialBackend("COM3", 9600, 0.05, serial_module=module, clock=clock)
        backend.send(JointTargets(1, {"right": ArmTargets(joints={"shoulder": -90.0})}, lift_mode=True))

        clock.now = 0.1
        backend.send(JointTargets(2, {"right": ArmTargets()}, lift_mode=False))

        self.assertEqual(module.ports[0].written[-1], b"RS:-90.0;L:0\n")

    def test_frames_carry_the_accumulated_state_of_both_arms(self) -> None:
        module = FakeSerialModule()
        clock = FakeClock()
        backend = SerialBackend("COM3", 9600, 0.05, serial_module=module, clock=clock)

        backend.send(JointTargets(1, {"right": ArmTargets(joints={"shoulder": -90.0})}))
        clock.now = 0.1
        backend.send(JointTargets(2, {"left": ArmTargets(joints={"elbow": 20.0})}))

        self.assertEqual(module.ports[0].written[-1], b"RS:-90.0;LE:20.0;L:0\n")

    def test_empty_targets_are_not_sent(self) -> None:
        module = FakeSerialModule()
        backend = SerialBackend("COM3", 9600, 0.05, serial_module=module, clock=FakeClock())

        backend.send(JointTargets(timestamp_ms=1))

        self.assertEqual(module.ports[0].written, [])

    def test_status_and_robot_state_follow_last_frame(self) -> None:
        module = FakeSerialModule()
        backend = SerialBackend("COM3", 9600, 0.05, serial_module=module, clock=FakeClock())

        self.assertEqual(backend.status_lines(), ["serial COM3 @ 9600: idle"])
        self.assertIsNone(backend.robot_state())

        backend.send(right({"elbow": 45.0}, gripper=GRIPPER_CLOSE))

        self.assertEqual(backend.status_lines(), ["serial COM3 @ 9600: RE:45.0;RG:1;L:0"])
        self.assertEqual(backend.robot_state().arm("right").joints, {"elbow": 45.0})
        self.assertEqual(backend.robot_state().arm("right").gripper, GRIPPER_CLOSE)

    def test_close_closes_port(self) -> None:
        module = FakeSerialModule()
        backend = SerialBackend("COM3", 9600, 0.05, serial_module=module, clock=FakeClock())

        backend.close()

        self.assertTrue(module.ports[0].closed)

    def test_failed_or_partial_write_closes_and_latches_link(self) -> None:
        from unittest.mock import Mock

        for result in (0, 3, None, OSError("disconnected"), TimeoutError("write timed out")):
            with self.subTest(result=result):
                module = FakeSerialModule()
                backend = SerialBackend("COM3", 9600, 0.05, serial_module=module)
                port = module.ports[0]
                port.write = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
                with self.assertRaisesRegex(SerialWriteError, "COM3"):
                    backend.send(right({"elbow": 45.0}))
                self.assertTrue(port.closed)
                with self.assertRaises(SerialWriteError):
                    backend.send(right({"elbow": 45.0}))
                port.write.assert_called_once()

    def test_closed_backend_cannot_send_even_empty_targets(self) -> None:
        module = FakeSerialModule()
        backend = SerialBackend("COM3", 9600, 0.05, serial_module=module)
        backend.close()
        backend.close()
        with self.assertRaises(SerialWriteError):
            backend.send(JointTargets(timestamp_ms=1))
        self.assertEqual(module.ports[0].written, [])

    def test_unopenable_port_exits_with_message(self) -> None:
        with self.assertRaises(SystemExit) as context:
            SerialBackend("COM9", 9600, 0.05, serial_module=FailingSerialModule())

        self.assertIn("COM9", str(context.exception))


if __name__ == "__main__":
    unittest.main()
