import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vision_robot_arm.app import _source_label, run_app
from vision_robot_arm.cli import parse_args
from vision_robot_arm.core.config import AppConfig
from vision_robot_arm.vision.camera import CameraDevice, open_camera_capture


class CameraSelectionTests(unittest.TestCase):
    def test_explicit_camera_failure_does_not_switch_to_another_camera(self):
        cv2 = Mock()
        capture = cv2.VideoCapture.return_value
        capture.isOpened.return_value = False
        with patch("vision_robot_arm.vision.camera.discover_cameras") as discover:
            result, index, _ = open_camera_capture(7, cv2, 200)
        self.assertIs(result, capture)
        self.assertEqual(index, 7)
        cv2.VideoCapture.assert_called_once_with(7, 200)
        discover.assert_not_called()

    def test_auto_skips_unavailable_devices(self):
        cv2 = Mock()
        unavailable, available = Mock(), Mock()
        unavailable.isOpened.return_value = False
        available.isOpened.return_value = True
        cv2.VideoCapture.side_effect = [unavailable, available]
        devices = [CameraDevice(0, "First"), CameraDevice(2, "Second")]
        with patch("vision_robot_arm.vision.camera.discover_cameras", return_value=devices):
            result, index, _ = open_camera_capture("auto", cv2, 200)
        self.assertIs(result, available)
        self.assertEqual(index, 2)
        unavailable.release.assert_called_once()

    def test_source_label_uses_actual_camera(self):
        self.assertEqual(_source_label(AppConfig(camera="auto"), 2), "CAM 2")
        self.assertEqual(_source_label(AppConfig(video_path=Path("clip.mp4")), 2), "clip.mp4")

    def test_cli_accepts_auto_and_rejects_invalid_camera_name(self):
        self.assertEqual(parse_args(["--camera", "auto"]).camera, "auto")
        with self.assertRaises(SystemExit):
            parse_args(["--camera", "not-a-camera"])

    def test_open_exception_is_not_masked_by_cleanup(self):
        deps = Mock()
        with (
            patch("vision_robot_arm.app.enable_high_dpi_awareness"),
            patch("vision_robot_arm.app.load_runtime_dependencies", return_value=deps),
            patch("vision_robot_arm.app.build_landmark_indices", return_value={}),
            patch("vision_robot_arm.app.build_landmark_names", return_value={}),
            patch("vision_robot_arm.app.open_camera_capture", side_effect=OSError("camera unavailable")),
        ):
            with self.assertRaisesRegex(OSError, "camera unavailable"):
                run_app(AppConfig())
        deps.cv2.destroyAllWindows.assert_called_once()
