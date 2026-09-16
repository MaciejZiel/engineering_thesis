import contextlib
import io
import unittest

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.backend import DebugBackend
from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.controller import MappedRobotController, NullRobotController
from vision_robot_arm.robot.factory import create_robot_controller
from vision_robot_arm.robot.targets import JointTargets


def make_state(angles: dict[str, float | None], gestures: tuple[str, ...] = ()) -> PoseState:
    return PoseState(
        timestamp_ms=1,
        landmarks=[],
        world_landmarks=None,
        raw_angles={},
        angles=angles,
        relative_angles={},
        gestures=gestures,
        calibrated=False,
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FactoryTests(unittest.TestCase):
    def test_none_backend_creates_null_controller(self) -> None:
        controller = create_robot_controller(RobotConfig(backend="none"))

        self.assertIsInstance(controller, NullRobotController)
        self.assertEqual(controller.status_lines(), [])

    def test_debug_backend_creates_mapped_controller(self) -> None:
        controller = create_robot_controller(RobotConfig(backend="debug"))

        self.assertIsInstance(controller, MappedRobotController)

    def test_mapped_controller_prints_commands_through_debug_backend(self) -> None:
        controller = create_robot_controller(RobotConfig(backend="debug"))
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            controller.update(make_state({"right_shoulder": 45.0}, ("right_elbow_bent",)))

        self.assertEqual(
            output.getvalue().strip(),
            "robot shoulder= 45.0 | gripper=close | lift_mode=off",
        )
        self.assertEqual(
            controller.status_lines(),
            ["robot debug: shoulder= 45.0 | gripper=close | lift_mode=off"],
        )


class DebugBackendTests(unittest.TestCase):
    def test_prints_are_rate_limited(self) -> None:
        clock = FakeClock()
        backend = DebugBackend(print_interval=1.0, clock=clock)
        targets = JointTargets(timestamp_ms=1, joints={"elbow": 90.0})
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            backend.send(targets)
            clock.now = 0.5
            backend.send(targets)
            clock.now = 1.0
            backend.send(targets)

        self.assertEqual(output.getvalue().count("robot "), 2)

    def test_empty_targets_are_not_printed(self) -> None:
        backend = DebugBackend(print_interval=1.0, clock=FakeClock())
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            backend.send(JointTargets(timestamp_ms=1))

        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
