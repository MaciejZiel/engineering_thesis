import unittest

import numpy as np

from pathlib import Path

from vision_robot_arm.app import _fit_frame, _frame_timestamp_ms
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


class FakeCapture:
    """Stands in for cv2.VideoCapture; POS_MSEC restarts at zero after a rewind."""

    POS_MSEC = 0

    def __init__(self, position_ms: float) -> None:
        self.position_ms = position_ms

    def get(self, prop: int) -> float:
        return self.position_ms


class TimestampTests(unittest.TestCase):
    video = AppConfig(video_path=Path("clip.mp4"), loop_video=True)

    def cv2(self) -> object:
        class Cv2:
            CAP_PROP_POS_MSEC = FakeCapture.POS_MSEC

        return Cv2()

    def test_video_time_comes_from_the_clip_position(self) -> None:
        stamp = _frame_timestamp_ms(self.cv2(), FakeCapture(900.0), self.video, 0.0, 867)

        self.assertEqual(stamp, 900)

    def test_a_repeated_position_still_moves_forward(self) -> None:
        stamp = _frame_timestamp_ms(self.cv2(), FakeCapture(900.0), self.video, 0.0, 900)

        self.assertEqual(stamp, 901)

    def test_the_offset_keeps_frame_spacing_across_a_rewind(self) -> None:
        offset = 900 + 33

        first = _frame_timestamp_ms(self.cv2(), FakeCapture(0.0), self.video, 0.0, 900, offset)
        second = _frame_timestamp_ms(self.cv2(), FakeCapture(33.0), self.video, 0.0, first, offset)

        self.assertEqual(first, 933)
        self.assertEqual(second - first, 33)

    def test_camera_time_runs_on_the_wall_clock(self) -> None:
        import time

        started = time.monotonic() - 2.0

        stamp = _frame_timestamp_ms(self.cv2(), FakeCapture(0.0), AppConfig(), started, -1)

        self.assertGreaterEqual(stamp, 2000)
