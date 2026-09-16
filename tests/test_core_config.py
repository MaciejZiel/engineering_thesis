from pathlib import Path
import unittest

from vision_robot_arm.core.config import DEFAULT_MODEL_PATH, AppConfig
from vision_robot_arm.core.pose_state import LandmarkPoint, mirror_landmarks


class LandmarkPointTests(unittest.TestCase):
    def test_missing_or_none_visibility_defaults_to_one(self) -> None:
        class HandLandmark:
            x, y, z, visibility = 0.1, 0.2, 0.3, None

        class BareLandmark:
            x, y, z = 0.4, 0.5, 0.6

        self.assertEqual(LandmarkPoint.from_landmark(HandLandmark()).visibility, 1.0)
        self.assertEqual(LandmarkPoint.from_landmark(BareLandmark()).visibility, 1.0)


class MirrorLandmarksTests(unittest.TestCase):
    def test_mirrors_x_and_keeps_other_fields(self) -> None:
        mirrored = mirror_landmarks([LandmarkPoint(0.25, 0.4, -0.1, visibility=0.8)])

        self.assertEqual(mirrored, [LandmarkPoint(0.75, 0.4, -0.1, visibility=0.8)])


class ConfigTests(unittest.TestCase):
    def test_config_rejects_missing_video_file(self) -> None:
        config = AppConfig(video_path=Path("does_not_exist.mp4"))

        with self.assertRaises(SystemExit):
            config.validate()

    def test_missing_hand_model_is_only_an_error_when_hands_enabled(self) -> None:
        missing = Path("no_such_hand_model.task")

        AppConfig(hand_model_path=missing, hands=False).validate()
        with self.assertRaises(SystemExit):
            AppConfig(hand_model_path=missing, hands=True).validate()

    def test_default_model_path_points_into_models_directory(self) -> None:
        self.assertEqual(DEFAULT_MODEL_PATH.parent.name, "models")
        self.assertTrue(DEFAULT_MODEL_PATH.exists())


if __name__ == "__main__":
    unittest.main()
