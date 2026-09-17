import unittest
from pathlib import Path

import numpy as np

from vision_robot_arm.app import (
    _fit_frame,
    _frame_timestamp_ms,
    _release_all,
    _remaining_frame_delay_ms,
    _resize_for_inference,
    configure_camera,
)
from vision_robot_arm.core.config import AppConfig


class FakeCv2:
    INTER_AREA = 3
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FPS = 3
    CAP_PROP_FOURCC = 4

    @staticmethod
    def VideoWriter_fourcc(*code: str) -> int:
        return sum(
            ord(character) << (8 * index) for index, character in enumerate(code)
        )

    def __init__(self) -> None:
        self.resize_calls: list[tuple[tuple[int, int], int]] = []

    def resize(
        self, _frame: object, size: tuple[int, int], interpolation: int
    ) -> object:
        self.resize_calls.append((size, interpolation))
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)


class FitFrameTests(unittest.TestCase):
    def test_inference_uses_a_smaller_copy_without_changing_the_preview(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        inference = _resize_for_inference(cv2, frame, AppConfig())
        self.assertEqual(inference.shape[:2], (540, 960))
        self.assertEqual(frame.shape[:2], (1080, 1920))

    def test_inference_resize_preserves_non_widescreen_aspect_ratio(self) -> None:
        frame = np.zeros((1200, 1600, 3), dtype=np.uint8)
        inference = _resize_for_inference(FakeCv2(), frame, AppConfig())
        self.assertEqual(inference.shape[:2], (540, 720))

    def test_video_wait_only_uses_the_remaining_frame_budget(self):
        self.assertEqual(_remaining_frame_delay_ms(40, 0.025), 15)
        self.assertEqual(_remaining_frame_delay_ms(40, 0.1), 1)
        self.assertEqual(_remaining_frame_delay_ms(1, 0.02), 1)

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


class CameraConfigurationTests(unittest.TestCase):
    class Capture:
        def __init__(self, fps_by_width: dict[int, float]) -> None:
            self.values = {1: 640.0, 2: 480.0, 3: 30.0, 4: 0.0}
            self.fps_by_width = fps_by_width
            self.calls = []

        def set(self, prop: int, value: float) -> bool:
            self.calls.append((prop, value))
            self.values[prop] = value
            if prop in (1, 3):
                self.values[3] = self.fps_by_width.get(
                    round(self.values[1]), value if prop == 3 else 0
                )
            return True

        def get(self, prop: int) -> float:
            return self.values[prop]

    def test_requests_mjpg_before_resolution_and_keeps_full_hd_at_30_fps(self) -> None:
        cv2 = FakeCv2()
        capture = self.Capture({1920: 30.0})
        mode = configure_camera(cv2, capture, AppConfig())
        self.assertEqual(
            capture.calls[0], (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        )
        self.assertEqual(
            (mode.width, mode.height, mode.fps, mode.codec), (1920, 1080, 30.0, "MJPG")
        )

    def test_reduces_resolution_until_requested_fps_is_available(self) -> None:
        mode = configure_camera(
            FakeCv2(), self.Capture({1920: 5.0, 1280: 30.0}), AppConfig()
        )
        self.assertEqual((mode.width, mode.height, mode.fps), (1280, 720, 30.0))


if __name__ == "__main__":
    unittest.main()


class FakeCapture:
    """Stands in for cv2.VideoCapture; POS_MSEC restarts at zero after a rewind."""

    POS_MSEC = 0

    def __init__(self, position_ms: float) -> None:
        self.position_ms = position_ms

    def get(self, prop: int) -> float:
        return self.position_ms


class EmptyFrameTests(unittest.TestCase):
    def test_a_capture_returning_an_empty_mat_does_not_crash(self) -> None:
        """Some webcam drivers hand back ok=True with a zero-size image."""
        cv2 = FakeCv2()
        config = AppConfig(width=1920, height=1080)

        for shape in ((0, 0, 3), (0, 1280, 3), (720, 0, 3)):
            with self.subTest(shape=shape):
                frame = np.zeros(shape, dtype=np.uint8)

                self.assertIs(_fit_frame(cv2, frame, config), frame)


class ReleaseTests(unittest.TestCase):
    def test_every_resource_is_closed_even_when_one_fails(self) -> None:
        closed: list[str] = []

        def failing() -> None:
            raise OSError(28, "No space left on device")

        _release_all(
            ("robot", lambda: closed.append("robot")),
            ("recording", failing),
            ("camera", lambda: closed.append("camera")),
            ("window", None),
        )

        self.assertEqual(closed, ["robot", "camera"])


class TimestampTests(unittest.TestCase):
    video = AppConfig(video_path=Path("clip.mp4"), loop_video=True)

    def cv2(self) -> object:
        class Cv2:
            CAP_PROP_POS_MSEC = FakeCapture.POS_MSEC

        return Cv2()

    def test_video_time_comes_from_the_clip_position(self) -> None:
        stamp = _frame_timestamp_ms(
            self.cv2(), FakeCapture(900.0), self.video, 0.0, 867
        )

        self.assertEqual(stamp, 900)

    def test_a_repeated_position_still_moves_forward(self) -> None:
        stamp = _frame_timestamp_ms(
            self.cv2(), FakeCapture(900.0), self.video, 0.0, 900
        )

        self.assertEqual(stamp, 901)

    def test_the_offset_keeps_frame_spacing_across_a_rewind(self) -> None:
        offset = 900 + 33

        first = _frame_timestamp_ms(
            self.cv2(), FakeCapture(0.0), self.video, 0.0, 900, offset
        )
        second = _frame_timestamp_ms(
            self.cv2(), FakeCapture(33.0), self.video, 0.0, first, offset
        )

        self.assertEqual(first, 933)
        self.assertEqual(second - first, 33)

    def test_two_camera_frames_in_one_millisecond_still_advance(self) -> None:
        import time

        started = time.monotonic()

        first = _frame_timestamp_ms(
            self.cv2(), FakeCapture(0.0), AppConfig(), started, -1
        )
        second = _frame_timestamp_ms(
            self.cv2(), FakeCapture(0.0), AppConfig(), started, first
        )

        self.assertGreater(second, first)

    def test_camera_time_runs_on_the_wall_clock(self) -> None:
        import time

        started = time.monotonic() - 2.0

        stamp = _frame_timestamp_ms(
            self.cv2(), FakeCapture(0.0), AppConfig(), started, -1
        )

        self.assertGreaterEqual(stamp, 2000)
