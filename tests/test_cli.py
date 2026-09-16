import unittest

from vision_robot_arm.cli import parse_args
from vision_robot_arm.robot.config import JointLimit


class CliRobotOptionsTests(unittest.TestCase):
    def test_robot_is_disabled_by_default(self) -> None:
        config = parse_args([])

        self.assertEqual(config.robot.backend, "none")
        self.assertFalse(config.robot.enabled)

    def test_robot_debug_flag_is_alias_for_debug_backend(self) -> None:
        config = parse_args(["--robot-debug"])

        self.assertEqual(config.robot.backend, "debug")

    def test_explicit_backend_wins_over_deprecated_alias(self) -> None:
        config = parse_args(["--robot-debug", "--robot-backend", "sim"])

        self.assertEqual(config.robot.backend, "sim")

    def test_robot_options_are_mapped_into_robot_config(self) -> None:
        config = parse_args(
            [
                "--robot-backend",
                "serial",
                "--robot-port",
                "COM3",
                "--robot-baud",
                "9600",
                "--robot-max-speed",
                "45",
                "--robot-elbow-range",
                "10",
                "170",
                "--robot-wrist-range",
                "20",
                "160",
                "--robot-right-host",
                "192.168.1.10",
                "--robot-ur-port",
                "30001",
                "--robot-servo-gain",
                "500",
            ]
        )

        self.assertEqual(config.robot.port, "COM3")
        self.assertEqual(config.robot.baud_rate, 9600)
        self.assertEqual(config.robot.max_speed_deg_s, 45.0)
        self.assertEqual(config.robot.elbow.limit, JointLimit(10.0, 170.0))
        self.assertEqual(config.robot.shoulder.limit, JointLimit(-180.0, 0.0))
        self.assertEqual(config.robot.wrist.limit, JointLimit(20.0, 160.0))
        self.assertEqual(config.robot.elbow.offset_deg, 180.0)
        self.assertEqual(config.robot.hosts, {"right": "192.168.1.10"})
        self.assertEqual(config.robot.ur_port, 30001)
        self.assertEqual(config.robot.servo_gain, 500)

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--robot-backend", "teleport"])

    def test_ur_backend_without_host_fails_validation(self) -> None:
        config = parse_args(["--robot-backend", "ur"])

        with self.assertRaises(SystemExit):
            config.validate()

        parse_args(["--robot-backend", "ur", "--robot-left-host", "10.0.0.5"]).robot.validate()

    def test_serial_backend_without_port_fails_validation(self) -> None:
        config = parse_args(["--robot-backend", "serial"])

        with self.assertRaises(SystemExit):
            config.validate()

    def test_test_mode_defaults_to_simulated_robot(self) -> None:
        config = parse_args(["--test-mode"])

        self.assertTrue(config.test_mode)
        self.assertEqual(config.robot.backend, "sim")

    def test_hand_tracking_is_on_by_default_and_can_be_disabled(self) -> None:
        self.assertTrue(parse_args([]).hands)
        self.assertFalse(parse_args(["--no-hands"]).hands)
        self.assertEqual(parse_args(["--hand-model", "x.task"]).hand_model_path.name, "x.task")

    def test_test_mode_keeps_explicit_backend(self) -> None:
        config = parse_args(["--test-mode", "--robot-backend", "debug"])

        self.assertEqual(config.robot.backend, "debug")

    def test_mirror_is_on_by_default_and_can_be_disabled(self) -> None:
        self.assertTrue(parse_args([]).mirror)
        self.assertFalse(parse_args(["--no-mirror"]).mirror)

    def test_existing_flags_still_parse(self) -> None:
        config = parse_args(["--camera", "1", "--smoothing-alpha", "0.2", "--loop-video"])

        self.assertEqual(config.camera, 1)
        self.assertEqual(config.smoothing_alpha, 0.2)
        self.assertTrue(config.loop_video)


if __name__ == "__main__":
    unittest.main()
