import unittest

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.config import JointLimit, RobotConfig
from vision_robot_arm.robot.mapping import RobotMapper
from vision_robot_arm.robot.targets import GRIPPER_CLOSE, GRIPPER_OPEN, JointTargets


def make_state(
    angles: dict[str, float | None],
    gestures: tuple[str, ...] = (),
    timestamp_ms: int = 1,
) -> PoseState:
    return PoseState(
        timestamp_ms=timestamp_ms,
        landmarks=[],
        world_landmarks=None,
        raw_angles={},
        angles=angles,
        relative_angles={},
        gestures=gestures,
        calibrated=False,
    )


class RobotMapperTests(unittest.TestCase):
    def test_maps_right_arm_angles_to_joints(self) -> None:
        mapper = RobotMapper(RobotConfig())

        targets = mapper.map(make_state({"right_shoulder": 45.0, "right_elbow": 120.0}))

        self.assertEqual(targets.timestamp_ms, 1)
        self.assertEqual(targets.joints, {"shoulder": 45.0, "elbow": 120.0})
        self.assertIsNone(targets.gripper)
        self.assertFalse(targets.lift_mode)

    def test_clamps_angles_to_joint_limits(self) -> None:
        config = RobotConfig(elbow_limit=JointLimit(20.0, 160.0))
        mapper = RobotMapper(config)

        targets = mapper.map(make_state({"right_shoulder": 210.0, "right_elbow": 5.0}))

        self.assertEqual(targets.joints, {"shoulder": 180.0, "elbow": 20.0})

    def test_omits_joints_without_reliable_angle(self) -> None:
        mapper = RobotMapper(RobotConfig())

        targets = mapper.map(make_state({"right_shoulder": None, "right_elbow": 90.0}))

        self.assertEqual(targets.joints, {"elbow": 90.0})

    def test_deadband_keeps_previous_value_for_small_changes(self) -> None:
        mapper = RobotMapper(RobotConfig(joint_deadband_deg=2.0))
        mapper.map(make_state({"right_shoulder": 90.0}))

        small_change = mapper.map(make_state({"right_shoulder": 91.0}))
        large_change = mapper.map(make_state({"right_shoulder": 93.0}))

        self.assertEqual(small_change.joints["shoulder"], 90.0)
        self.assertEqual(large_change.joints["shoulder"], 93.0)

    def test_reset_forgets_deadband_reference(self) -> None:
        mapper = RobotMapper(RobotConfig(joint_deadband_deg=2.0))
        mapper.map(make_state({"right_shoulder": 90.0}))
        mapper.reset()

        targets = mapper.map(make_state({"right_shoulder": 91.0}))

        self.assertEqual(targets.joints["shoulder"], 91.0)

    def test_gestures_control_gripper_and_lift_mode(self) -> None:
        mapper = RobotMapper(RobotConfig())

        closed = mapper.map(make_state({}, gestures=("right_elbow_bent",)))
        opened = mapper.map(make_state({}, gestures=("right_arm_side",)))
        both = mapper.map(make_state({}, gestures=("right_arm_side", "right_elbow_bent")))
        lifted = mapper.map(make_state({}, gestures=("right_hand_up",)))

        self.assertEqual(closed.gripper, GRIPPER_CLOSE)
        self.assertEqual(opened.gripper, GRIPPER_OPEN)
        self.assertEqual(both.gripper, GRIPPER_CLOSE)
        self.assertIsNone(lifted.gripper)
        self.assertTrue(lifted.lift_mode)


class JointTargetsTests(unittest.TestCase):
    def test_format_lists_joints_gripper_and_lift_mode(self) -> None:
        targets = JointTargets(
            timestamp_ms=1,
            joints={"shoulder": 45.0, "elbow": 180.0},
            gripper=GRIPPER_CLOSE,
            lift_mode=True,
        )

        self.assertEqual(
            targets.format(),
            "shoulder= 45.0 | elbow=180.0 | gripper=close | lift_mode=on",
        )

    def test_has_data_is_false_for_empty_targets(self) -> None:
        self.assertFalse(JointTargets(timestamp_ms=1).has_data)
        self.assertTrue(JointTargets(timestamp_ms=1, joints={"elbow": 1.0}).has_data)


class RobotConfigTests(unittest.TestCase):
    def test_serial_backend_requires_port(self) -> None:
        with self.assertRaises(SystemExit):
            RobotConfig(backend="serial").validate()

    def test_rejects_unknown_backend(self) -> None:
        with self.assertRaises(SystemExit):
            RobotConfig(backend="teleport").validate()

    def test_rejects_inverted_joint_limit(self) -> None:
        with self.assertRaises(SystemExit):
            RobotConfig(shoulder_limit=JointLimit(90.0, 10.0)).validate()

    def test_default_config_is_valid_and_disabled(self) -> None:
        config = RobotConfig()

        config.validate()

        self.assertFalse(config.enabled)
        self.assertTrue(RobotConfig(backend="sim").enabled)


if __name__ == "__main__":
    unittest.main()
