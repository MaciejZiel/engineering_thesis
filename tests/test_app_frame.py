import unittest

import numpy as np

from vision_robot_arm.app import _fit_frame
from vision_robot_arm.core.config import AppConfig


class FakeCv2:
    INTER_AREA = 3

    def __init__(self) -> None:
        self.resize_calls: list[tuple[tuple[int, int], int]] = []

    def resize(self, _frame: object, size: tuple[int, int], interpolation: int) -> object:
        self.resize_calls.append((size, interpolation))
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)


class FitFrameTests(unittest.TestCase):
    def test_does_not_upscale_or_stretch_small_camera_frame(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = _fit_frame(cv2, frame, AppConfig(width=1920, height=1080))

        self.assertIs(result, frame)
        self.assertEqual(cv2.resize_calls, [])

    def test_downscales_large_frame_to_fit_target(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        result = _fit_frame(cv2, frame, AppConfig(width=1920, height=1080))

        self.assertEqual(result.shape[:2], (1080, 1920))
        self.assertEqual(cv2.resize_calls, [((1920, 1080), cv2.INTER_AREA)])

    def test_preserves_aspect_ratio_while_downscaling(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((1200, 1920, 3), dtype=np.uint8)

        result = _fit_frame(cv2, frame, AppConfig(width=1920, height=1080))

        self.assertEqual(result.shape[:2], (1080, 1728))
        self.assertEqual(cv2.resize_calls, [((1728, 1080), cv2.INTER_AREA)])


if __name__ == "__main__":
    unittest.main()
