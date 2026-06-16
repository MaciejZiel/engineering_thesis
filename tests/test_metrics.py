from dataclasses import dataclass
import csv
from pathlib import Path
import tempfile
import unittest

from vision_robot_arm.metrics import calculate_angle, calculate_angles
from vision_robot_arm.smoothing import LowPassValueFilter
from vision_robot_arm.calibration import PoseCalibration
from vision_robot_arm.pose_state import LandmarkPoint, PoseState
from vision_robot_arm.recording import CsvPoseRecorder
from vision_robot_arm.gestures import detect_gestures


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


class RecordingTests(unittest.TestCase):
    def test_csv_recorder_writes_pose_state(self) -> None:
        state = PoseState(
            timestamp_ms=123,
            landmarks=[LandmarkPoint(0.1, 0.2, 0.3, 0.9)],
            world_landmarks=[LandmarkPoint(1.0, 2.0, 3.0, 1.0)],
            raw_angles={"left_elbow": 91.0},
            angles={"left_elbow": 90.0},
            relative_angles={"left_elbow": 5.0},
            gestures=("left_elbow_bent",),
            calibrated=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = CsvPoseRecorder(Path(temp_dir))
            path = recorder.start({0: "nose"})
            recorder.write_state(state, {0: "nose"})
            recorder.stop()

            with path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["timestamp_ms"], "123")
        self.assertEqual(rows[0]["calibrated"], "True")
        self.assertEqual(rows[0]["angle_left_elbow"], "90.0")
        self.assertEqual(rows[0]["relative_left_elbow"], "5.0")
        self.assertEqual(rows[0]["gestures"], "left_elbow_bent")
        self.assertEqual(rows[0]["nose_x"], "0.1")
        self.assertEqual(rows[0]["nose_world_z"], "3.0")


class GestureTests(unittest.TestCase):
    def test_detects_hand_up_and_bent_elbow(self) -> None:
        indices = {
            "LEFT_SHOULDER": 0,
            "LEFT_ELBOW": 1,
            "LEFT_WRIST": 2,
            "RIGHT_SHOULDER": 3,
            "RIGHT_ELBOW": 4,
            "RIGHT_WRIST": 5,
        }
        landmarks = [
            FakeLandmark(0.4, 0.5),
            FakeLandmark(0.35, 0.35),
            FakeLandmark(0.35, 0.25),
            FakeLandmark(0.6, 0.5),
            FakeLandmark(0.65, 0.5),
            FakeLandmark(0.68, 0.5),
        ]

        gestures = detect_gestures(
            landmarks,
            {"left_elbow": 75.0, "right_elbow": 170.0},
            indices,
            min_visibility=0.55,
        )

        self.assertIn("left_hand_up", gestures)
        self.assertIn("left_elbow_bent", gestures)
        self.assertNotIn("right_elbow_bent", gestures)


if __name__ == "__main__":
    unittest.main()
