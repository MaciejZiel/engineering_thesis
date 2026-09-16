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
            ]
        )

        self.assertEqual(config.robot.port, "COM3")
        self.assertEqual(config.robot.baud_rate, 9600)
        self.assertEqual(config.robot.max_speed_deg_s, 45.0)
        self.assertEqual(config.robot.elbow_limit, JointLimit(10.0, 170.0))
        self.assertEqual(config.robot.shoulder_limit, JointLimit(0.0, 180.0))

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--robot-backend", "teleport"])

    def test_serial_backend_without_port_fails_validation(self) -> None:
        config = parse_args(["--robot-backend", "serial"])

        with self.assertRaises(SystemExit):
            config.validate()

    def test_existing_flags_still_parse(self) -> None:
        config = parse_args(["--camera", "1", "--smoothing-alpha", "0.2", "--loop-video"])

        self.assertEqual(config.camera, 1)
        self.assertEqual(config.smoothing_alpha, 0.2)
        self.assertTrue(config.loop_video)


if __name__ == "__main__":
    unittest.main()
