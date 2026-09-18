"""Base, wrist 2 and wrist 3 driven from the operator's rotation angles."""

import unittest
from unittest.mock import Mock

from vision_robot_arm.cli import parse_args
from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.config import OPERATION_TRACKING, RobotConfig
from vision_robot_arm.robot.mapping import RobotMapper
from vision_robot_arm.robot.session import HardwareSession
from vision_robot_arm.robot.targets import ArmState, RobotState

PITCH = {"right_shoulder_elevation": 90.0, "right_elbow": 120.0, "right_wrist": 180.0}


def state(**angles) -> PoseState:
    merged = dict(PITCH)
    merged.update(angles)
    return PoseState(1, [], None, merged, merged, {}, (), False)


def robot_state(**joints) -> RobotState:
    pose = {"base": -33.0, "shoulder": -79.0, "elbow": 15.0, "wrist_1": -59.0, "wrist_2": 55.0, "wrist_3": 175.0}
    pose.update(joints)
    return RobotState({"right": ArmState(pose, dict(pose), "open")}, False)


class RotationMappingTests(unittest.TestCase):
    def test_3d_space_drives_all_six_joints(self) -> None:
        mapper = RobotMapper(RobotConfig(tracking_space="3d", joint_deadband_deg=0.0))

        joints = mapper.map(
            state(right_shoulder_azimuth=30.0, right_wrist_deviation=-10.0, right_forearm_roll=45.0)
        ).arm("right").joints

        self.assertEqual(set(joints), {"base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"})
        self.assertAlmostEqual(joints["base"], 30.0)
        self.assertAlmostEqual(joints["wrist_2"], -10.0)
        self.assertAlmostEqual(joints["wrist_3"], 45.0)

    def test_2d_space_leaves_the_rotation_joints_held(self) -> None:
        mapper = RobotMapper(RobotConfig(tracking_space="2d", joint_deadband_deg=0.0))

        joints = mapper.map(
            state(right_shoulder_azimuth=30.0, right_wrist_deviation=-10.0, right_forearm_roll=45.0)
        ).arm("right").joints

        self.assertEqual(set(joints), {"shoulder", "elbow", "wrist_1"})

    def test_unmeasured_rotation_leaves_that_joint_out(self) -> None:
        mapper = RobotMapper(RobotConfig(tracking_space="3d", joint_deadband_deg=0.0))

        joints = mapper.map(state(right_shoulder_azimuth=30.0)).arm("right").joints

        self.assertIn("base", joints)
        self.assertNotIn("wrist_2", joints)
        self.assertNotIn("wrist_3", joints)

    def test_rotation_signs_come_from_the_command_line(self) -> None:
        config = parse_args(
            ["--tracking-space", "3d", "--robot-rotation-signs", "-1", "1", "-1",
             "--robot-base-excursion", "15", "--robot-base-range", "-90", "90"]
        ).robot
        config.validate()
        mapper = RobotMapper(RobotConfig(
            tracking_space="3d", joint_deadband_deg=0.0,
            base=config.base, wrist_2=config.wrist_2, wrist_3=config.wrist_3,
        ))

        joints = mapper.map(
            state(right_shoulder_azimuth=30.0, right_wrist_deviation=10.0, right_forearm_roll=45.0)
        ).arm("right").joints

        self.assertAlmostEqual(joints["base"], -30.0)
        self.assertAlmostEqual(joints["wrist_2"], 10.0)
        self.assertAlmostEqual(joints["wrist_3"], -45.0)
        self.assertEqual(config.base_excursion_deg, 15.0)
        self.assertEqual((config.base.limit.minimum, config.base.limit.maximum), (-90.0, 90.0))

    def test_base_excursion_is_validated(self) -> None:
        for bad in (0.0, -5.0, 91.0, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                RobotConfig(base_excursion_deg=bad).validate()
        RobotConfig(base_excursion_deg=90.0).validate()


class RotationAnchoringTests(unittest.TestCase):
    """Enabling control must never turn an arm: the first angles define zero."""

    def setUp(self) -> None:
        self.backend = Mock()
        self.backend.ready.return_value = True
        self.backend.robot_state.return_value = robot_state()
        self.session = HardwareSession(
            RobotConfig(
                backend="ur", operation=OPERATION_TRACKING, tracking_space="3d",
                right_host="test", joint_deadband_deg=0.0,
            ),
            Mock(return_value=self.backend),
        )
        self.session.advance()  # connect
        self.session.advance()  # capture pose

    def activate(self, **angles) -> None:
        self.session.update(state(**angles))
        self.session.advance()  # enable control -> active

    def sent(self, index: int = -1) -> dict[str, float]:
        return self.backend.send.call_args_list[index].args[0].arm("right").joints

    def test_first_rotation_after_enabling_commands_the_robots_own_pose(self) -> None:
        self.activate(right_shoulder_azimuth=40.0, right_forearm_roll=-20.0, right_wrist_deviation=5.0)

        self.session.update(state(right_shoulder_azimuth=40.0, right_forearm_roll=-20.0, right_wrist_deviation=5.0))

        joints = self.sent()
        self.assertAlmostEqual(joints["base"], -33.0)
        self.assertAlmostEqual(joints["wrist_3"], 175.0)
        self.assertAlmostEqual(joints["wrist_2"], 55.0)

    def test_only_the_change_since_enabling_is_added(self) -> None:
        self.activate(right_shoulder_azimuth=40.0, right_forearm_roll=-20.0)
        self.session.update(state(right_shoulder_azimuth=40.0, right_forearm_roll=-20.0))

        self.session.update(state(right_shoulder_azimuth=55.0, right_forearm_roll=-50.0))

        joints = self.sent()
        self.assertAlmostEqual(joints["base"], -33.0 + 15.0)
        self.assertAlmostEqual(joints["wrist_3"], 175.0 - 30.0)

    def test_pitch_joints_are_untouched_by_anchoring(self) -> None:
        self.activate(right_shoulder_azimuth=40.0)
        self.session.update(state(right_shoulder_azimuth=40.0))

        joints = self.sent()
        # shoulder_elevation 90 -> UR shoulder -90 through the ordinary mapping.
        self.assertAlmostEqual(joints["shoulder"], -90.0)

    def test_rotation_measured_later_anchors_when_it_first_appears(self) -> None:
        self.activate()
        self.session.update(state())
        self.assertNotIn("base", self.sent())

        self.session.update(state(right_shoulder_azimuth=70.0))
        self.assertAlmostEqual(self.sent()["base"], -33.0)
        self.session.update(state(right_shoulder_azimuth=75.0))
        self.assertAlmostEqual(self.sent()["base"], -28.0)

    def test_resume_after_pause_re_anchors_instead_of_jumping(self) -> None:
        self.activate(right_shoulder_azimuth=40.0)
        self.session.update(state(right_shoulder_azimuth=40.0))
        self.session.pause()
        self.backend.robot_state.return_value = robot_state(base=-25.0)
        self.backend.send.reset_mock()

        self.session.update(state(right_shoulder_azimuth=90.0))
        self.session.advance()  # resume
        self.session.update(state(right_shoulder_azimuth=90.0))

        self.assertAlmostEqual(self.sent()["base"], -25.0)

    def test_without_a_robot_pose_the_rotation_joints_are_held(self) -> None:
        self.backend.robot_state.return_value = None
        self.activate(right_shoulder_azimuth=40.0)

        self.session.update(state(right_shoulder_azimuth=40.0))

        joints = self.sent()
        self.assertNotIn("base", joints)
        self.assertIn("shoulder", joints)

    def test_2d_space_never_anchors_rotation(self) -> None:
        backend = Mock()
        backend.ready.return_value = True
        backend.robot_state.return_value = robot_state()
        session = HardwareSession(
            RobotConfig(backend="ur", operation=OPERATION_TRACKING, tracking_space="2d", right_host="test"),
            Mock(return_value=backend),
        )
        session.advance()
        session.advance()
        session.update(state(right_shoulder_azimuth=40.0))
        session.advance()

        self.assertEqual(session._rotation_robot_origin, {})


if __name__ == "__main__":
    unittest.main()
