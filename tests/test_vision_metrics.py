import csv
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from vision_robot_arm.core.pose_state import LandmarkPoint, PoseState
from vision_robot_arm.vision.calibration import PoseCalibration
from vision_robot_arm.vision.gestures import detect_gestures
from vision_robot_arm.vision.metrics import (
    ANGLE_DEFINITIONS,
    calculate_angle,
    calculate_angles,
)
from vision_robot_arm.vision.recording import CsvPoseRecorder
from vision_robot_arm.vision.smoothing import LowPassValueFilter


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


class AngleDefinitionTests(unittest.TestCase):
    def test_wrist_angles_use_elbow_wrist_and_index(self) -> None:
        self.assertEqual(
            ANGLE_DEFINITIONS["right_wrist"],
            ("RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_INDEX"),
        )
        self.assertEqual(
            ANGLE_DEFINITIONS["left_wrist"], ("LEFT_ELBOW", "LEFT_WRIST", "LEFT_INDEX")
        )


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

    def test_csv_records_the_angle_that_drives_the_robot_shoulder(self) -> None:
        state = PoseState(
            timestamp_ms=1,
            landmarks=[LandmarkPoint(0.1, 0.2, 0.3, 0.9)],
            world_landmarks=None,
            raw_angles={},
            angles={"left_shoulder_elevation": 120.5},
            relative_angles={},
            gestures=(),
            calibrated=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = CsvPoseRecorder(Path(temp_dir))
            path = recorder.start({0: "nose"})
            recorder.write_state(state, {0: "nose"})
            recorder.stop()

            with path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["angle_left_shoulder_elevation"], "120.5")
        self.assertEqual(rows[0]["angle_right_shoulder_elevation"], "")

    def test_csv_records_3d_hands_coordinate_frames_and_angle_sources(self) -> None:
        point = LandmarkPoint(0.1, 0.2, -0.3)
        world = LandmarkPoint(0.4, 0.5, -0.6)
        state = PoseState(
            1,
            [],
            [world],
            {},
            {"left_wrist": 170.0},
            {},
            (),
            False,
            hand_landmarks={"left": (point,)},
            hand_world_landmarks={"left": (world,)},
            angle_sources={"left_wrist": "hand_world_3d"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = CsvPoseRecorder(Path(temp_dir))
            path = recorder.start({0: "nose"})
            recorder.write_state(state, {0: "nose"})
            recorder.stop()
            with path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))
        self.assertEqual(row["source_left_wrist"], "hand_world_3d")
        self.assertEqual(row["pose_world_frame"], "body_relative_m")
        self.assertEqual(row["hand_world_frame"], "pose_wrist_anchored_m")
        self.assertEqual(row["left_hand_wrist_z"], "-0.3")
        self.assertEqual(row["left_hand_wrist_world_z"], "-0.6")


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

    def test_arm_side_fires_when_the_arm_is_extended_not_when_it_is_crossed(
        self,
    ) -> None:
        """Detection runs on the un-mirrored frame: the person's left is at larger image x."""
        indices = {
            "LEFT_SHOULDER": 0,
            "RIGHT_SHOULDER": 1,
            "LEFT_ELBOW": 2,
            "RIGHT_ELBOW": 3,
            "LEFT_WRIST": 4,
            "RIGHT_WRIST": 5,
        }
        straight = {"left_elbow": 175.0, "right_elbow": 175.0}
        extended = [
            FakeLandmark(0.60, 0.50),
            FakeLandmark(0.40, 0.50),
            FakeLandmark(0.75, 0.50),
            FakeLandmark(0.25, 0.50),
            FakeLandmark(0.90, 0.50),
            FakeLandmark(0.10, 0.50),
        ]
        crossed = [
            FakeLandmark(0.60, 0.50),
            FakeLandmark(0.40, 0.50),
            FakeLandmark(0.50, 0.50),
            FakeLandmark(0.50, 0.50),
            FakeLandmark(0.30, 0.50),
            FakeLandmark(0.70, 0.50),
        ]

        spread = detect_gestures(extended, straight, indices, min_visibility=0.55)
        folded = detect_gestures(crossed, straight, indices, min_visibility=0.55)

        self.assertIn("left_arm_side", spread)
        self.assertIn("right_arm_side", spread)
        self.assertNotIn("left_arm_side", folded)
        self.assertNotIn("right_arm_side", folded)


if __name__ == "__main__":
    unittest.main()
