import unittest
from unittest.mock import Mock

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.config import OPERATION_TRACKING, RobotConfig
from vision_robot_arm.robot.session import HardwareSession


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

    def test_connect_home_and_arm_require_separate_actions(self):
        self.session.advance()
        self.backend.home.assert_not_called()
        self.session.advance()
        self.backend.home.assert_called_once()
        self.session.update(pose())
        self.assertEqual(self.session.phase, "homing")
        self.backend.ready.return_value = True
        self.session.update(pose())
        self.assertEqual(self.session.phase, "ready")
        self.backend.send.assert_not_called()
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

    def test_fault_is_latched_and_closes_backend(self):
        self.prepare()
        self.session.advance()
        self.backend.send.side_effect = RuntimeError("feedback expired")
        self.session.update(pose())
        self.assertEqual(self.session.phase, "fault")
        self.backend.close.assert_called_once()
        self.session.advance()
        self.assertEqual(self.session.phase, "fault")

    def test_tracking_loss_during_homing_requires_new_home_action(self):
        self.session.advance()
        self.session.advance()
        self.session.reset()
        self.assertEqual(self.session.phase, "connected")
        self.backend.pause.assert_called_once()

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
