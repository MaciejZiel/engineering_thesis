import unittest
from unittest.mock import Mock

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.config import (
    OPERATION_COMMISSIONING,
    OPERATION_KEYFRAME,
    OPERATION_TRACKING,
    RobotConfig,
)
from vision_robot_arm.robot.session import HardwareSession
from vision_robot_arm.robot.targets import ArmState, RobotState


def pose(complete=True):
    angles = {"right_shoulder_elevation": 90.0, "right_elbow": 120.0}
    if complete:
        angles["right_wrist"] = 180.0
    return PoseState(1, [], None, angles, angles, {}, (), False)


class HardwareSessionTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.backend.ready.return_value = False
        self.factory = Mock(return_value=self.backend)
        self.session = HardwareSession(
            RobotConfig(
                backend="ur", operation=OPERATION_TRACKING, right_host="test"
            ),
            self.factory,
        )

    def prepare(self):
        self.session.advance()
        self.session.advance()
        self.backend.ready.return_value = True
        self.session.update(pose())

    def test_startup_and_tracking_do_not_connect_or_move_hardware(self):
        self.session.update(pose())
        self.factory.assert_not_called()
        self.assertEqual(self.session.phase, "disconnected")

    def test_connect_capture_and_arm_require_separate_actions_without_homing(self):
        self.session.advance()
        self.backend.arm_tracking.assert_not_called()
        self.session.advance()
        self.backend.arm_tracking.assert_called_once()
        self.backend.home.assert_not_called()
        self.session.update(pose())
        self.assertEqual(self.session.phase, "ready")
        self.backend.send.assert_not_called()
        self.backend.ready.return_value = True
        self.session.advance()
        self.session.update(pose())
        self.backend.send.assert_called_once()

    def test_partial_tracking_pauses_and_recovery_does_not_auto_resume(self):
        self.prepare()
        self.session.advance()
        self.session.update(pose(False))
        self.backend.pause.assert_called_once()
        self.assertEqual(self.session.phase, "paused")
        self.session.update(pose())
        self.backend.send.assert_not_called()
        self.session.advance()
        self.session.update(pose())
        self.backend.send.assert_called_once()

    def test_brief_tracking_loss_repeats_last_target_before_stopping(self):
        now = [10.0]
        backend = Mock()
        backend.ready.return_value = True
        session = HardwareSession(
            RobotConfig(
                backend="ur",
                operation=OPERATION_TRACKING,
                right_host="test",
                tracking_loss_grace_s=0.4,
            ),
            Mock(return_value=backend),
            clock=lambda: now[0],
        )
        session.advance()
        session.advance()
        session.update(pose())
        session.advance()
        session.update(pose())
        valid_target = backend.send.call_args.args[0]

        session.tracking_lost()
        now[0] += 0.3
        session.tracking_lost()

        self.assertEqual(session.phase, "active")
        self.assertEqual(backend.send.call_args.args[0], valid_target)
        backend.pause.assert_not_called()
        backend.note_tracking_event.assert_any_call("tracking_gap_held", 0.0)

        now[0] += 0.11
        session.tracking_lost()

        self.assertEqual(session.phase, "paused")
        backend.pause.assert_called_once()
        event, elapsed = backend.note_tracking_event.call_args.args
        self.assertEqual(event, "tracking_gap_stopped")
        self.assertAlmostEqual(elapsed, 0.41)

    def test_fault_is_latched_and_closes_backend(self):
        self.prepare()
        self.session.advance()
        self.backend.send.side_effect = RuntimeError("feedback expired")
        self.session.update(pose())
        self.assertEqual(self.session.phase, "fault")
        self.backend.close.assert_called_once()
        self.session.advance()
        self.assertEqual(self.session.phase, "fault")

    def test_tracking_loss_before_activation_does_not_authorize_motion(self):
        self.session.advance()
        self.session.advance()
        self.session.reset()
        self.assertEqual(self.session.phase, "ready")
        self.backend.pause.assert_not_called()
        self.backend.send.assert_not_called()

    def test_monitor_mode_connects_without_calling_any_motion_method(self):
        backend = Mock()
        session = HardwareSession(
            RobotConfig(backend="ur", right_host="test"),
            Mock(return_value=backend),
        )

        session.advance()
        session.advance()
        session.update(pose())

        self.assertEqual(session.phase, "monitoring")
        backend.home.assert_not_called()
        backend.send.assert_not_called()

    def test_commissioning_captures_current_pose_instead_of_homing(self):
        backend = Mock()
        session = HardwareSession(
            RobotConfig(
                backend="ur",
                operation=OPERATION_COMMISSIONING,
                right_host="test",
            ),
            Mock(return_value=backend),
        )

        session.advance()
        session.advance()

        self.assertEqual(session.phase, "commissioning")
        backend.arm_commissioning.assert_called_once()
        backend.home.assert_not_called()

        session.jog(-1)
        backend.refresh_jog.assert_called_once_with(-1)
        session.update(pose())
        backend.send.assert_not_called()

    def test_commissioning_reset_from_missing_pose_does_not_disarm(self):
        backend = Mock()
        session = HardwareSession(
            RobotConfig(
                backend="ur",
                operation=OPERATION_COMMISSIONING,
                right_host="test",
            ),
            Mock(return_value=backend),
        )
        session.advance()
        session.advance()

        session.reset()

        self.assertEqual(session.phase, "commissioning")
        backend.pause.assert_not_called()

    def test_commissioning_control_action_disarms_motion(self):
        backend = Mock()
        session = HardwareSession(
            RobotConfig(
                backend="ur",
                operation=OPERATION_COMMISSIONING,
                right_host="test",
            ),
            Mock(return_value=backend),
        )
        session.advance()
        session.advance()

        session.advance()

        backend.pause.assert_called_once()
        self.assertEqual(session.phase, "connected")

    def test_keyframe_mode_moves_by_the_captured_body_pose_delta(self):
        backend = Mock()
        backend.ready.return_value = True
        backend.robot_state.return_value = RobotState(
            arms={
                "right": ArmState(
                    joints={
                        "base": 0.0,
                        "shoulder": -40.0,
                        "elbow": 20.0,
                        "wrist_1": -30.0,
                        "wrist_2": 0.0,
                        "wrist_3": 0.0,
                    },
                    targets={},
                    gripper="open",
                )
            },
            lift_mode=False,
        )
        session = HardwareSession(
            RobotConfig(
                backend="ur",
                operation=OPERATION_KEYFRAME,
                right_host="test",
            ),
            Mock(return_value=backend),
        )
        start = PoseState(
            1,
            [],
            None,
            start_angles := {
                "right_shoulder_elevation": 90.0,
                "right_elbow": 120.0,
                "right_wrist": 180.0,
            },
            start_angles,
            {},
            (),
            False,
        )
        end = PoseState(
            2,
            [],
            None,
            end_angles := {
                "right_shoulder_elevation": 100.0,
                "right_elbow": 100.0,
                "right_wrist": 170.0,
            },
            end_angles,
            {},
            (),
            False,
        )

        session.advance()
        session.advance()
        self.assertEqual(session.phase, "keyframe_start")
        session.update(start)
        session.advance()
        self.assertEqual(session.phase, "keyframe_end")
        session.update(end)
        session.advance()
        self.assertEqual(session.phase, "active")
        session.update(end)

        target = backend.send.call_args.args[0].arm("right")
        self.assertEqual(
            target.joints,
            {"shoulder": -50.0, "elbow": 40.0, "wrist_1": -20.0},
        )
