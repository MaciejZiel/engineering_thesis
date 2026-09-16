from pathlib import Path
import unittest

from vision_robot_arm.core.config import DEFAULT_MODEL_PATH, AppConfig


class ConfigTests(unittest.TestCase):
    def test_config_rejects_missing_video_file(self) -> None:
        config = AppConfig(video_path=Path("does_not_exist.mp4"))

        with self.assertRaises(SystemExit):
            config.validate()

    def test_default_model_path_points_into_models_directory(self) -> None:
        self.assertEqual(DEFAULT_MODEL_PATH.parent.name, "models")
        self.assertTrue(DEFAULT_MODEL_PATH.exists())


if __name__ == "__main__":
    unittest.main()
