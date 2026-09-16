import unittest

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.config import (
    DEFAULT_ELBOW_MAPPING,
    DEFAULT_SHOULDER_MAPPING,
    DEFAULT_WRIST_MAPPING,
    JointLimit,
    JointMapping,
    RobotConfig,
)
from vision_robot_arm.robot.mapping import RobotMapper
from vision_robot_arm.robot.targets import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    UR_HOME_DEG,
    ArmTargets,
    JointTargets,
    full_joint_pose,
)


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


class JointMappingTests(unittest.TestCase):
    def test_default_shoulder_mapping_matches_ur_conventions(self) -> None:
        self.assertEqual(DEFAULT_SHOULDER_MAPPING.to_robot(90.0), -90.0)
        self.assertEqual(DEFAULT_SHOULDER_MAPPING.to_robot(180.0), 0.0)
        self.assertEqual(DEFAULT_SHOULDER_MAPPING.to_robot(0.0), -180.0)

    def test_default_elbow_and_wrist_are_zero_when_straight(self) -> None:
        self.assertEqual(DEFAULT_ELBOW_MAPPING.to_robot(180.0), 0.0)
        self.assertEqual(DEFAULT_ELBOW_MAPPING.to_robot(90.0), 90.0)
        self.assertEqual(DEFAULT_WRIST_MAPPING.to_robot(180.0), 0.0)

    def test_mapping_clamps_to_its_limit_and_inverts(self) -> None:
        mapping = JointMapping(source="elbow", offset_deg=180.0, sign=-1.0, limit=JointLimit(-160.0, 160.0))

        self.assertEqual(mapping.to_robot(0.0), 160.0)
        self.assertEqual(mapping.to_body(90.0), 90.0)


class RobotMapperTests(unittest.TestCase):
    def test_maps_both_arms_to_ur_joint_angles(self) -> None:
        mapper = RobotMapper(RobotConfig())

        targets = mapper.map(
            make_state(
                {
                    "right_shoulder": 90.0,
                    "right_elbow": 120.0,
                    "right_wrist": 160.0,
                    "left_shoulder": 45.0,
                    "left_elbow": 180.0,
                }
            )
        )

        self.assertEqual(targets.timestamp_ms, 1)
        self.assertEqual(
            targets.arm("right").joints,
            {"shoulder": -90.0, "elbow": 60.0, "wrist_1": 20.0},
        )
        self.assertEqual(targets.arm("left").joints, {"shoulder": -135.0, "elbow": 0.0})
        self.assertIsNone(targets.arm("right").gripper)
        self.assertFalse(targets.lift_mode)

    def test_clamps_angles_to_joint_limits(self) -> None:
        config = RobotConfig(
            elbow=JointMapping("elbow", 180.0, -1.0, JointLimit(-160.0, 100.0)),
            wrist=JointMapping("wrist", 180.0, -1.0, JointLimit(-45.0, 45.0)),
        )
        mapper = RobotMapper(config)

        targets = mapper.map(
            make_state({"right_shoulder": 210.0, "right_elbow": 5.0, "right_wrist": 20.0})
        )

        self.assertEqual(
            targets.arm("right").joints,
            {"shoulder": 0.0, "elbow": 100.0, "wrist_1": 45.0},
        )

    def test_omits_joints_without_reliable_angle(self) -> None:
        mapper = RobotMapper(RobotConfig())

        targets = mapper.map(make_state({"right_shoulder": None, "right_elbow": 90.0}))

        self.assertEqual(targets.arm("right").joints, {"elbow": 90.0})
        self.assertEqual(targets.arm("left").joints, {})

    def test_deadband_is_tracked_per_arm_and_joint(self) -> None:
        mapper = RobotMapper(RobotConfig(joint_deadband_deg=2.0))
        mapper.map(make_state({"right_shoulder": 90.0, "left_shoulder": 90.0}))

        targets = mapper.map(make_state({"right_shoulder": 91.0, "left_shoulder": 93.0}))

        self.assertEqual(targets.arm("right").joints["shoulder"], -90.0)
        self.assertEqual(targets.arm("left").joints["shoulder"], -87.0)

    def test_reset_forgets_deadband_reference(self) -> None:
        mapper = RobotMapper(RobotConfig(joint_deadband_deg=2.0))
        mapper.map(make_state({"right_shoulder": 90.0}))
        mapper.reset()

        targets = mapper.map(make_state({"right_shoulder": 91.0}))

        self.assertEqual(targets.arm("right").joints["shoulder"], -89.0)

    def test_hand_gestures_control_each_gripper(self) -> None:
        mapper = RobotMapper(RobotConfig())

        targets = mapper.map(make_state({}, gestures=("right_fist", "left_hand_open")))

        self.assertEqual(targets.arm("right").gripper, GRIPPER_CLOSE)
        self.assertEqual(targets.arm("left").gripper, GRIPPER_OPEN)

    def test_no_hand_gesture_keeps_gripper_undefined(self) -> None:
        mapper = RobotMapper(RobotConfig())

        targets = mapper.map(make_state({}, gestures=("right_elbow_bent", "right_hand_up")))

        self.assertIsNone(targets.arm("right").gripper)
        self.assertTrue(targets.lift_mode)


class JointTargetsTests(unittest.TestCase):
    def test_format_lists_arms_and_lift_mode(self) -> None:
        targets = JointTargets(
            timestamp_ms=1,
            arms={
                "right": ArmTargets(joints={"shoulder": -45.0, "elbow": 120.0}, gripper=GRIPPER_CLOSE),
                "left": ArmTargets(),
            },
            lift_mode=True,
        )

        self.assertEqual(
            targets.format(),
            "R: shoulder= -45.0 elbow= 120.0 gripper=close | lift_mode=on",
        )

    def test_has_data_is_false_for_empty_targets(self) -> None:
        self.assertFalse(JointTargets(timestamp_ms=1).has_data)
        self.assertTrue(
            JointTargets(timestamp_ms=1, arms={"left": ArmTargets(joints={"elbow": 1.0})}).has_data
        )

    def test_full_joint_pose_fills_held_joints_from_home(self) -> None:
        pose = full_joint_pose({"shoulder": -45.0, "finger": 3.0})

        self.assertEqual(pose["shoulder"], -45.0)
        self.assertEqual(pose["base"], UR_HOME_DEG["base"])
        self.assertEqual(pose["wrist_1"], UR_HOME_DEG["wrist_1"])
        self.assertNotIn("finger", pose)
        self.assertEqual(len(pose), 6)


class RobotConfigTests(unittest.TestCase):
    def test_ur_backend_requires_a_host(self) -> None:
        with self.assertRaises(SystemExit):
            RobotConfig(backend="ur").validate()

        RobotConfig(backend="ur", left_host="192.168.1.11").validate()

    def test_serial_backend_requires_port(self) -> None:
        with self.assertRaises(SystemExit):
            RobotConfig(backend="serial").validate()

    def test_rejects_unknown_backend(self) -> None:
        with self.assertRaises(SystemExit):
            RobotConfig(backend="teleport").validate()

    def test_rejects_speed_above_ur7e_limit(self) -> None:
        with self.assertRaises(SystemExit):
            RobotConfig(max_speed_deg_s=181.0).validate()

    def test_rejects_inverted_joint_limit(self) -> None:
        bad = JointMapping("wrist", 180.0, -1.0, JointLimit(90.0, 10.0))

        with self.assertRaises(SystemExit):
            RobotConfig(wrist=bad).validate()

    def test_default_config_is_valid_and_disabled(self) -> None:
        config = RobotConfig()

        config.validate()

        self.assertFalse(config.enabled)
        self.assertTrue(RobotConfig(backend="sim").enabled)
        self.assertEqual(config.limit_for("elbow"), JointLimit(-160.0, 160.0))
        self.assertEqual(config.limit_for("base"), JointLimit(-360.0, 360.0))
        self.assertEqual(config.home_for("wrist_1"), -90.0)
        self.assertEqual(config.hosts, {})
        self.assertEqual(RobotConfig(right_host="10.0.0.2").hosts, {"right": "10.0.0.2"})


if __name__ == "__main__":
    unittest.main()
