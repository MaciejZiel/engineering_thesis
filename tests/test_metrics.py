from dataclasses import dataclass
import unittest

from vision_robot_arm.metrics import calculate_angle, calculate_angles
from vision_robot_arm.smoothing import LowPassValueFilter
from vision_robot_arm.calibration import PoseCalibration


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class MetricsTests(unittest.TestCase):
    def test_calculate_angle_for_right_angle(self) -> None:
        a = FakeLandmark(0.0, 1.0)
        b = FakeLandmark(0.0, 0.0)
        c = FakeLandmark(1.0, 0.0)

        self.assertAlmostEqual(calculate_angle(a, b, c), 90.0)

    def test_calculate_angles_marks_low_visibility_as_missing(self) -> None:
        indices = {
            "LEFT_SHOULDER": 0,
            "LEFT_ELBOW": 1,
            "LEFT_WRIST": 2,
        }
        landmarks = [
            FakeLandmark(0.0, 1.0),
            FakeLandmark(0.0, 0.0, visibility=0.1),
            FakeLandmark(1.0, 0.0),
        ]

        angles = calculate_angles(
            landmarks,
            indices,
            min_visibility=0.55,
            angle_definitions={
                "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST")
            },
        )

        self.assertIsNone(angles["left_elbow"])


class SmoothingTests(unittest.TestCase):
    def test_low_pass_value_filter_smooths_toward_new_value(self) -> None:
        value_filter = LowPassValueFilter(alpha=0.5)

        self.assertEqual(value_filter.update(10.0), 10.0)
        self.assertEqual(value_filter.update(20.0), 15.0)


class CalibrationTests(unittest.TestCase):
    def test_calibration_returns_relative_angle_offsets(self) -> None:
        calibration = PoseCalibration()
        calibration.capture({"left_elbow": 90.0, "right_elbow": None})

        relative = calibration.relative_angles(
            {"left_elbow": 110.0, "right_elbow": 80.0}
        )

        self.assertEqual(relative["left_elbow"], 20.0)
        self.assertIsNone(relative["right_elbow"])


if __name__ == "__main__":
    unittest.main()
