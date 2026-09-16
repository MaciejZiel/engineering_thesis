import unittest

from vision_robot_arm.robot.serial_backend import (
    SerialBackend,
    encode_targets,
    load_serial_module,
)
from vision_robot_arm.robot.targets import GRIPPER_CLOSE, GRIPPER_OPEN, JointTargets


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeSerialPort:
    def __init__(self, port: str, baud_rate: int, timeout: float) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
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

    def Serial(self, port: str, baud_rate: int, timeout: float) -> FakeSerialPort:
        fake = FakeSerialPort(port, baud_rate, timeout)
        self.ports.append(fake)
        return fake


class FailingSerialModule:
    def Serial(self, port: str, baud_rate: int, timeout: float) -> None:
        raise OSError("device not found")


class EncodeTargetsTests(unittest.TestCase):
    def test_encodes_joints_gripper_and_lift_mode(self) -> None:
        targets = JointTargets(
            timestamp_ms=1,
            joints={"shoulder": 90.0, "elbow": 45.0},
            gripper=GRIPPER_CLOSE,
            lift_mode=False,
        )

        self.assertEqual(encode_targets(targets), b"S:90.0;E:45.0;G:1;L:0\n")

    def test_omits_gripper_when_not_commanded(self) -> None:
        targets = JointTargets(timestamp_ms=1, joints={"elbow": 10.0}, lift_mode=True)

        self.assertEqual(encode_targets(targets), b"E:10.0;L:1\n")

    def test_open_gripper_is_zero(self) -> None:
        targets = JointTargets(timestamp_ms=1, gripper=GRIPPER_OPEN)

        self.assertEqual(encode_targets(targets), b"G:0;L:0\n")

    def test_unknown_joints_are_skipped(self) -> None:
        targets = JointTargets(timestamp_ms=1, joints={"wrist": 10.0, "shoulder": 1.0})

        self.assertEqual(encode_targets(targets), b"S:1.0;L:0\n")


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

    def test_writes_frames_rate_limited(self) -> None:
        module = FakeSerialModule()
        clock = FakeClock()
        backend = SerialBackend("COM3", 9600, 1.0, serial_module=module, clock=clock)
        targets = JointTargets(timestamp_ms=1, joints={"shoulder": 90.0})

        backend.send(targets)
        clock.now = 0.5
        backend.send(targets)
        clock.now = 1.0
        backend.send(targets)

        self.assertEqual(module.ports[0].written, [b"S:90.0;L:0\n", b"S:90.0;L:0\n"])

    def test_empty_targets_are_not_sent(self) -> None:
        module = FakeSerialModule()
        backend = SerialBackend("COM3", 9600, 0.05, serial_module=module, clock=FakeClock())

        backend.send(JointTargets(timestamp_ms=1))

        self.assertEqual(module.ports[0].written, [])

    def test_status_lines_show_last_frame(self) -> None:
        module = FakeSerialModule()
        backend = SerialBackend("COM3", 9600, 0.05, serial_module=module, clock=FakeClock())

        self.assertEqual(backend.status_lines(), ["serial COM3 @ 9600: idle"])

        backend.send(JointTargets(timestamp_ms=1, joints={"elbow": 45.0}))

        self.assertEqual(backend.status_lines(), ["serial COM3 @ 9600: E:45.0;L:0"])

    def test_close_closes_port(self) -> None:
        module = FakeSerialModule()
        backend = SerialBackend("COM3", 9600, 0.05, serial_module=module, clock=FakeClock())

        backend.close()

        self.assertTrue(module.ports[0].closed)

    def test_unopenable_port_exits_with_message(self) -> None:
        with self.assertRaises(SystemExit) as context:
            SerialBackend("COM9", 9600, 0.05, serial_module=FailingSerialModule())

        self.assertIn("COM9", str(context.exception))


if __name__ == "__main__":
    unittest.main()
