from dataclasses import replace
import unittest

from vision_robot_arm.core.config import AppConfig
from vision_robot_arm.robot.config import RobotConfig


INVALID_NUMBERS = (float("nan"), float("inf"), -float("inf"), True, "1", None, 10**400)


class NumericValidationTests(unittest.TestCase):
    def test_robot_rejects_nonfinite_or_non_numeric_values(self):
        for field in (
            "print_interval", "send_interval", "max_speed_deg_s", "joint_deadband_deg",
            "start_seconds", "start_speed_deg_s", "start_accel_deg_s2", "servo_lookahead_s",
        ):
            for value in INVALID_NUMBERS:
                with self.subTest(field=field, value=str(value)[:30]):
                    with self.assertRaisesRegex(SystemExit, "finite number"):
                        replace(RobotConfig(), **{field: value}).validate()

    def test_robot_integer_fields_reject_floats_and_bools(self):
        for field in ("baud_rate", "ur_port", "rtde_port", "dashboard_port", "tool_output", "servo_gain"):
            for value in (True, 1.0, 1.5, float("nan"), float("inf"), "1", None):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(SystemExit, "integer"):
                        replace(RobotConfig(), **{field: value}).validate()

    def test_mapping_parameters_are_finite(self):
        config = RobotConfig()
        for joint in ("shoulder", "elbow", "wrist"):
            mapping = getattr(config, joint)
            for field in ("offset_deg", "sign", "minimum", "maximum"):
                for value in INVALID_NUMBERS:
                    with self.subTest(joint=joint, field=field, value=str(value)[:30]):
                        if field in ("minimum", "maximum"):
                            changed = replace(mapping, limit=replace(mapping.limit, **{field: value}))
                        else:
                            changed = replace(mapping, **{field: value})
                        with self.assertRaisesRegex(SystemExit, "finite number"):
                            replace(config, **{joint: changed}).validate()

    def test_mapping_empty_source_and_zero_sign_rejected(self):
        config = RobotConfig()
        for source in ("", " ", None):
            with self.subTest(source=source), self.assertRaises(SystemExit):
                replace(config, shoulder=replace(config.shoulder, source=source)).validate()
        with self.assertRaises(SystemExit):
            replace(config, shoulder=replace(config.shoulder, sign=0)).validate()

    def test_valid_robot_config_and_boundaries(self):
        RobotConfig().validate()
        RobotConfig(joint_deadband_deg=0, start_seconds=0, servo_lookahead_s=0.03).validate()
        RobotConfig(servo_gain=2000, servo_lookahead_s=0.2, max_speed_deg_s=180).validate()

    def test_app_rejects_invalid_numbers_before_model_lookup(self):
        for field in (
            "print_interval", "camera_fps", "visibility_threshold", "smoothing_alpha",
            "min_detection_confidence", "min_pose_presence_confidence",
            "min_tracking_confidence", "hand_detection_confidence", "hand_presence_confidence",
        ):
            for value in INVALID_NUMBERS:
                with self.subTest(field=field, value=str(value)[:30]):
                    with self.assertRaisesRegex(SystemExit, "finite number"):
                        replace(AppConfig(), **{field: value}).validate()

    def test_app_rejects_non_integral_dimensions(self):
        for field in ("width", "height", "inference_width", "inference_height", "num_poses"):
            for value in (True, 1.5, float("inf"), None):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(SystemExit, "integer"):
                        replace(AppConfig(), **{field: value}).validate()

    def test_app_rejects_invalid_camera_target(self):
        for camera in (-1, True, 1.5, "invalid", "-1", None):
            with self.subTest(camera=camera), self.assertRaisesRegex(SystemExit, "--camera"):
                AppConfig(camera=camera).validate()

    def test_negative_dimensions_rejected(self):
        for field in ("width", "height"):
            with self.subTest(field=field), self.assertRaisesRegex(SystemExit, "0 or greater"):
                replace(AppConfig(), **{field: -1}).validate()
