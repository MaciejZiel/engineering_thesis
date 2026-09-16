import unittest

from vision_robot_arm.robot.config import JointLimit, RobotConfig
from vision_robot_arm.robot.simulation import SimulationBackend
from vision_robot_arm.robot.targets import GRIPPER_CLOSE, GRIPPER_OPEN, JointTargets


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SimulationBackendTests(unittest.TestCase):
    def test_starts_at_home_position_with_open_gripper(self) -> None:
        backend = SimulationBackend(RobotConfig(home_deg=45.0), clock=FakeClock())

        state = backend.state

        self.assertEqual(state.joints, {"shoulder": 45.0, "elbow": 45.0})
        self.assertEqual(state.gripper, GRIPPER_OPEN)
        self.assertFalse(state.lift_mode)
        self.assertTrue(state.settled)

    def test_joint_moves_at_most_max_speed_per_step(self) -> None:
        backend = SimulationBackend(RobotConfig(max_speed_deg_s=90.0), clock=FakeClock())
        backend.send(JointTargets(timestamp_ms=1, joints={"shoulder": 180.0}))

        state = backend.step(0.5)

        self.assertAlmostEqual(state.joints["shoulder"], 135.0)
        self.assertEqual(state.targets["shoulder"], 180.0)
        self.assertFalse(state.settled)

    def test_joint_reaches_target_without_overshoot(self) -> None:
        backend = SimulationBackend(RobotConfig(max_speed_deg_s=90.0), clock=FakeClock())
        backend.send(JointTargets(timestamp_ms=1, joints={"elbow": 100.0}))

        state = backend.step(1.0)

        self.assertEqual(state.joints["elbow"], 100.0)
        self.assertTrue(state.settled)

    def test_send_uses_clock_to_advance_motion(self) -> None:
        clock = FakeClock()
        backend = SimulationBackend(RobotConfig(max_speed_deg_s=10.0), clock=clock)

        backend.send(JointTargets(timestamp_ms=1, joints={"shoulder": 0.0}))
        clock.now = 2.0
        backend.send(JointTargets(timestamp_ms=2, joints={"shoulder": 0.0}))

        self.assertAlmostEqual(backend.state.joints["shoulder"], 70.0)

    def test_targets_are_clamped_to_joint_limits(self) -> None:
        config = RobotConfig(elbow_limit=JointLimit(30.0, 150.0), home_deg=90.0)
        backend = SimulationBackend(config, clock=FakeClock())

        backend.send(JointTargets(timestamp_ms=1, joints={"elbow": 500.0}))

        self.assertEqual(backend.state.targets["elbow"], 150.0)

    def test_gripper_latches_until_new_command(self) -> None:
        backend = SimulationBackend(RobotConfig(), clock=FakeClock())

        backend.send(JointTargets(timestamp_ms=1, gripper=GRIPPER_CLOSE))
        backend.send(JointTargets(timestamp_ms=2, gripper=None, lift_mode=True))

        self.assertEqual(backend.state.gripper, GRIPPER_CLOSE)
        self.assertTrue(backend.state.lift_mode)

    def test_unknown_joints_are_ignored(self) -> None:
        backend = SimulationBackend(RobotConfig(), clock=FakeClock())

        backend.send(JointTargets(timestamp_ms=1, joints={"wrist": 10.0}))

        self.assertNotIn("wrist", backend.state.joints)

    def test_status_lines_describe_joints_and_gripper(self) -> None:
        backend = SimulationBackend(RobotConfig(), clock=FakeClock())
        backend.send(JointTargets(timestamp_ms=1, joints={"shoulder": 120.0}))

        lines = backend.status_lines()

        self.assertEqual(lines[0], "sim shoulder= 90.0 -> 120.0")
        self.assertEqual(lines[1], "sim elbow= 90.0 ->  90.0")
        self.assertEqual(lines[2], "sim gripper=open | lift_mode=off")


if __name__ == "__main__":
    unittest.main()
