import unittest

from vision_robot_arm.robot.config import JointLimit, JointMapping, RobotConfig
from vision_robot_arm.robot.simulation import SimulationBackend
from vision_robot_arm.robot.targets import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    UR_HOME_DEG,
    ArmTargets,
    JointTargets,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def right(joints: dict[str, float] | None = None, gripper: str | None = None, ts: int = 1) -> JointTargets:
    return JointTargets(timestamp_ms=ts, arms={"right": ArmTargets(joints=joints or {}, gripper=gripper)})


class SimulationBackendTests(unittest.TestCase):
    def test_both_arms_start_at_ur_home_with_open_grippers(self) -> None:
        backend = SimulationBackend(RobotConfig(), clock=FakeClock())

        state = backend.state

        self.assertEqual(sorted(state.arms), ["left", "right"])
        for arm in state.arms.values():
            self.assertEqual(arm.joints, UR_HOME_DEG)
            self.assertEqual(arm.gripper, GRIPPER_OPEN)
            self.assertTrue(arm.settled)
        self.assertFalse(state.lift_mode)

    def test_joint_moves_at_most_max_speed_per_step(self) -> None:
        backend = SimulationBackend(RobotConfig(max_speed_deg_s=90.0), clock=FakeClock())
        backend.send(right({"shoulder": 0.0}))

        state = backend.step(0.5)

        self.assertAlmostEqual(state.arm("right").joints["shoulder"], -45.0)
        self.assertEqual(state.arm("right").targets["shoulder"], 0.0)
        self.assertFalse(state.arm("right").settled)
        self.assertTrue(state.arm("left").settled)

    def test_joint_reaches_target_without_overshoot(self) -> None:
        backend = SimulationBackend(RobotConfig(max_speed_deg_s=90.0), clock=FakeClock())
        backend.send(right({"wrist_1": -20.0}))

        state = backend.step(1.0)

        self.assertEqual(state.arm("right").joints["wrist_1"], -20.0)
        self.assertTrue(state.arm("right").settled)

    def test_send_uses_clock_to_advance_motion(self) -> None:
        clock = FakeClock()
        backend = SimulationBackend(RobotConfig(max_speed_deg_s=10.0), clock=clock)

        backend.send(right({"elbow": 90.0}))
        clock.now = 0.2
        backend.send(right({"elbow": 90.0}, ts=2))

        self.assertAlmostEqual(backend.state.arm("right").joints["elbow"], 2.0)

    def test_a_tracking_gap_cannot_buy_a_giant_step(self) -> None:
        clock = FakeClock()
        backend = SimulationBackend(RobotConfig(max_speed_deg_s=60.0), clock=clock)
        backend.send(right({"shoulder": -170.0}))

        clock.now = 5.0  # the operator was out of frame for five seconds
        backend.send(right({"shoulder": -170.0}, ts=2))

        moved = abs(backend.state.arm("right").joints["shoulder"] - (-90.0))
        self.assertLessEqual(moved, 60.0 * 0.2 + 0.001)

    def test_targets_are_clamped_to_joint_limits(self) -> None:
        config = RobotConfig(elbow=JointMapping("elbow", 180.0, -1.0, JointLimit(-30.0, 30.0)))
        backend = SimulationBackend(config, clock=FakeClock())

        backend.send(right({"elbow": 500.0}))

        self.assertEqual(backend.state.arm("right").targets["elbow"], 30.0)

    def test_gripper_latches_until_new_command(self) -> None:
        backend = SimulationBackend(RobotConfig(), clock=FakeClock())

        backend.send(right(gripper=GRIPPER_CLOSE))
        backend.send(JointTargets(timestamp_ms=2, arms={"right": ArmTargets()}, lift_mode=True))

        self.assertEqual(backend.state.arm("right").gripper, GRIPPER_CLOSE)
        self.assertEqual(backend.state.arm("left").gripper, GRIPPER_OPEN)
        self.assertTrue(backend.state.lift_mode)

    def test_unknown_joints_and_arms_are_ignored(self) -> None:
        backend = SimulationBackend(RobotConfig(), clock=FakeClock())

        backend.send(
            JointTargets(
                timestamp_ms=1,
                arms={"right": ArmTargets(joints={"finger": 10.0}), "tail": ArmTargets(joints={"shoulder": 1.0})},
            )
        )

        self.assertNotIn("finger", backend.state.arm("right").joints)
        self.assertIsNone(backend.state.arm("tail"))

    def test_status_lines_describe_both_arms(self) -> None:
        backend = SimulationBackend(RobotConfig(), clock=FakeClock())
        backend.send(right({"shoulder": -30.0}))

        lines = backend.status_lines()

        self.assertEqual(lines[0], "sim R: S  -90.0-> -30.0 E    0.0->   0.0 W1  -90.0-> -90.0 grip open")
        self.assertEqual(lines[1], "sim L: S  -90.0-> -90.0 E    0.0->   0.0 W1  -90.0-> -90.0 grip open")
        self.assertEqual(lines[2], "sim lift_mode=off")

    def test_robot_state_matches_state_property(self) -> None:
        backend = SimulationBackend(RobotConfig(), clock=FakeClock())
        backend.send(right({"shoulder": -30.0}))

        self.assertEqual(backend.robot_state(), backend.state)


if __name__ == "__main__":
    unittest.main()
