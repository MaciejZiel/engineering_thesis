from dataclasses import dataclass
import unittest

from vision_robot_arm.metrics import calculate_angle, calculate_angles


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


if __name__ == "__main__":
    unittest.main()
