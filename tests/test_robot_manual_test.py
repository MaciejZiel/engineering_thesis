import unittest
from unittest.mock import Mock

from vision_robot_arm.robot.config import (
    OPERATION_COMMISSIONING,
    OPERATION_MONITOR,
)
from vision_robot_arm.robot.manual_test import (
    ManualArmTestSession,
    ManualTestSettings,
)


class ManualTestSettingsTests(unittest.TestCase):
    def test_builds_a_single_arm_monitor_configuration(self):
        config = ManualTestSettings(
            host="192.168.1.10", side="left", joint="wrist_2"
        ).config(OPERATION_MONITOR)

        self.assertEqual(config.operation, OPERATION_MONITOR)
        self.assertEqual(config.hosts, {"left": "192.168.1.10"})
        self.assertEqual(config.commissioning_joint, "wrist_2")

    def test_builds_a_bounded_commissioning_configuration(self):
        config = ManualTestSettings(
            host="192.168.1.10",
            joint="elbow",
            speed_deg_s=0.5,
            excursion_deg=1.0,
        ).config(OPERATION_COMMISSIONING)

        self.assertEqual(config.operation, OPERATION_COMMISSIONING)
        self.assertEqual(config.commissioning_speed_deg_s, 0.5)
        self.assertEqual(config.commissioning_excursion_deg, 1.0)
        self.assertFalse(config.commissioning_require_reduced)
        self.assertTrue(config.feedback)
        self.assertTrue(config.preflight)

    def test_rejects_empty_host_and_out_of_range_motion(self):
        with self.assertRaisesRegex(ValueError, "IP"):
            ManualTestSettings(host=" ").config(OPERATION_MONITOR)
        with self.assertRaisesRegex(ValueError, "between 0 and 30"):
            ManualTestSettings(host="robot", speed_deg_s=31).config(
                OPERATION_COMMISSIONING
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 80"):
            ManualTestSettings(host="robot", excursion_deg=81).config(
                OPERATION_COMMISSIONING
            )

    def test_large_manual_motion_is_available_in_normal_mode(self):
        config = ManualTestSettings(
            host="robot",
            speed_deg_s=30,
            excursion_deg=80,
        ).config(OPERATION_COMMISSIONING)

        self.assertFalse(config.commissioning_require_reduced)


class ManualArmTestSessionTests(unittest.TestCase):
    def setUp(self):
        self.monitor = Mock()
        self.control = Mock()
        self.state = Mock()
        self.monitor.robot_state.return_value = self.state
        self.control.robot_state.return_value = self.state
        self.monitor_factory = Mock(return_value=self.monitor)
        self.control_factory = Mock(return_value=self.control)
        self.session = ManualArmTestSession(
            self.monitor_factory, self.control_factory
        )
        self.settings = ManualTestSettings(host="192.168.1.10")

    def tearDown(self):
        self.session.close()

    def prepare_and_arm(self):
        self.session.connect_monitor(self.settings)
        self.session.prepare_control()
        self.session.arm()

    def test_connect_starts_read_only_and_does_not_open_control_backend(self):
        self.session.connect_monitor(self.settings)

        config = self.monitor_factory.call_args.args[0]
        self.assertEqual(config.operation, OPERATION_MONITOR)
        self.control_factory.assert_not_called()
        self.assertEqual(self.session.phase, "monitoring")
        self.assertIs(self.session.tick(), self.state)

    def test_prepare_closes_monitor_but_does_not_arm_or_move(self):
        self.session.connect_monitor(self.settings)
        self.session.prepare_control()

        config = self.control_factory.call_args.args[0]
        self.assertEqual(config.operation, OPERATION_COMMISSIONING)
        self.monitor.close.assert_called_once()
        self.control.arm_commissioning.assert_not_called()
        self.control.refresh_jog.assert_not_called()
        self.assertEqual(self.session.phase, "prepared")

    def test_arm_captures_current_pose_without_jogging(self):
        self.session.connect_monitor(self.settings)
        self.session.prepare_control()
        self.session.arm()

        self.control.arm_commissioning.assert_called_once()
        self.control.refresh_jog.assert_not_called()
        self.assertTrue(self.session.can_jog)

    def test_hold_refreshes_one_direction_and_release_stops_immediately(self):
        self.prepare_and_arm()

        self.session.begin_jog(-1)
        self.session.tick()
        self.session.tick()
        self.session.end_jog()
        self.session.tick()

        self.assertEqual(
            self.control.refresh_jog.call_args_list,
            [unittest.mock.call(-1), unittest.mock.call(-1)],
        )
        self.control.pause.assert_called_once()
        self.assertEqual(self.session.phase, "armed")

    def test_speed_can_be_changed_after_control_is_prepared(self):
        self.session.connect_monitor(self.settings)
        self.session.prepare_control()

        self.session.set_speed(0.3)

        self.control.set_commissioning_speed.assert_called_once_with(0.3)

    def test_jog_is_impossible_before_explicit_arming(self):
        for prepare in (False, True):
            with self.subTest(prepared=prepare):
                session = ManualArmTestSession(
                    self.monitor_factory, self.control_factory
                )
                session.connect_monitor(self.settings)
                if prepare:
                    session.prepare_control()
                with self.assertRaisesRegex(RuntimeError, "requires armed"):
                    session.begin_jog(1)
                session.close()

    def test_stop_disarms_closes_and_forgets_settings(self):
        self.prepare_and_arm()

        self.session.stop_and_disconnect()

        self.control.pause.assert_called_once()
        self.control.close.assert_called_once()
        self.assertEqual(self.session.phase, "disconnected")
        self.assertIsNone(self.session.settings)

    def test_stop_failure_still_closes_and_disables_control(self):
        self.prepare_and_arm()
        self.control.pause.side_effect = OSError("connection lost")

        with self.assertRaisesRegex(RuntimeError, "shutdown"):
            self.session.stop_and_disconnect()

        self.control.close.assert_called_once()
        self.assertEqual(self.session.phase, "disconnected")
        self.assertFalse(self.session.can_jog)

    def test_fault_is_latched_and_closes_the_backend(self):
        self.prepare_and_arm()
        self.control.robot_state.side_effect = RuntimeError("feedback expired")

        self.assertIsNone(self.session.tick())

        self.assertEqual(self.session.phase, "fault")
        self.assertIn("feedback expired", self.session.error)
        self.control.close.assert_called_once()
        self.assertFalse(self.session.can_jog)


if __name__ == "__main__":
    unittest.main()
