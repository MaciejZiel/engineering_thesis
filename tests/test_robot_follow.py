"""Following the operator every N seconds instead of streaming."""

import unittest
from unittest.mock import Mock

from vision_robot_arm.cli import parse_args
from vision_robot_arm.robot.config import OPERATION_TRACKING, RobotConfig
from vision_robot_arm.robot.controller import MappedRobotController, NullRobotController
from vision_robot_arm.robot.mapping import RobotMapper
from vision_robot_arm.robot.session import HardwareSession
from vision_robot_arm.robot.simulation import SimulationBackend
from vision_robot_arm.robot.targets import ArmTargets, JointTargets


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def right(joints: dict[str, float]) -> JointTargets:
    return JointTargets(timestamp_ms=1, arms={"right": ArmTargets(joints=joints)})


class FollowIntervalConfigTests(unittest.TestCase):
    def test_zero_means_streaming_and_is_the_default(self) -> None:
        self.assertEqual(RobotConfig().follow_interval_s, 0.0)
        RobotConfig(follow_interval_s=0.0).validate()
        RobotConfig(follow_interval_s=10.0).validate()

    def test_out_of_range_or_non_finite_values_are_rejected(self) -> None:
        for bad in (-0.5, 10.5, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(SystemExit, "follow-interval"):
                RobotConfig(follow_interval_s=bad).validate()

    def test_command_line_flag_reaches_the_config(self) -> None:
        config = parse_args(["--robot-follow-interval", "1.5"])
        self.assertEqual(config.robot.follow_interval_s, 1.5)


class SimulationFollowTests(unittest.TestCase):
    def test_targets_are_only_accepted_once_per_interval(self) -> None:
        clock = FakeClock()
        backend = SimulationBackend(
            RobotConfig(follow_interval_s=1.0, max_speed_deg_s=1000.0), clock=clock
        )

        backend.send(right({"shoulder": -60.0}))
        clock.now = 0.5
        backend.send(right({"shoulder": -30.0}))
        self.assertEqual(backend.state.arm("right").targets["shoulder"], -60.0, "ignored inside the interval")

        clock.now = 1.0
        backend.send(right({"shoulder": -30.0}))
        self.assertEqual(backend.state.arm("right").targets["shoulder"], -30.0)
        self.assertIn("sim follow every 1.00 s", " ".join(backend.status_lines()))

    def test_setting_zero_returns_to_continuous_following(self) -> None:
        clock = FakeClock()
        backend = SimulationBackend(RobotConfig(follow_interval_s=1.0), clock=clock)
        backend.send(right({"shoulder": -60.0}))

        backend.set_follow_interval(0.0)
        clock.now = 0.1
        backend.send(right({"shoulder": -30.0}))

        self.assertEqual(backend.state.arm("right").targets["shoulder"], -30.0)
        self.assertNotIn("sim follow", " ".join(backend.status_lines()))

    def test_controllers_pass_the_setting_through(self) -> None:
        clock = FakeClock()
        backend = SimulationBackend(RobotConfig(), clock=clock)
        controller = MappedRobotController(RobotMapper(RobotConfig()), backend)

        controller.set_follow_interval(2.0)

        self.assertEqual(backend.follow_interval_s, 2.0)
        NullRobotController().set_follow_interval(2.0)  # must simply be accepted


class SessionFollowTests(unittest.TestCase):
    def test_choice_made_before_connecting_is_applied_on_connect(self) -> None:
        backend = Mock()
        session = HardwareSession(
            RobotConfig(backend="ur", operation=OPERATION_TRACKING, right_host="test"),
            Mock(return_value=backend),
        )

        session.set_follow_interval(1.5)
        backend.set_follow_interval.assert_not_called()
        session.advance()

        backend.set_follow_interval.assert_called_once_with(1.5)
        self.assertEqual(session.follow_interval_s, 1.5)

    def test_choice_made_while_connected_is_forwarded_immediately(self) -> None:
        backend = Mock()
        session = HardwareSession(
            RobotConfig(backend="ur", operation=OPERATION_TRACKING, right_host="test"),
            Mock(return_value=backend),
        )
        session.advance()

        session.set_follow_interval(0.75)

        backend.set_follow_interval.assert_called_once_with(0.75)

    def test_a_backend_that_rejects_the_value_faults_the_session(self) -> None:
        backend = Mock()
        backend.set_follow_interval.side_effect = RuntimeError("out of range")
        session = HardwareSession(
            RobotConfig(backend="ur", operation=OPERATION_TRACKING, right_host="test"),
            Mock(return_value=backend),
        )
        session.advance()

        session.set_follow_interval(99.0)

        self.assertEqual(session.phase, "fault")


if __name__ == "__main__":
    unittest.main()
