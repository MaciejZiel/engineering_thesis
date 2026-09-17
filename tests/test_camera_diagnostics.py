import contextlib
from dataclasses import replace
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vision_robot_arm.cli import main
from vision_robot_arm.vision.camera import configure_camera, open_camera_capture
from vision_robot_arm.vision.camera_diagnostics import (
    DiagnosticSettings, apply_diagnostic_mode, collect_diagnostic,
    diagnose_camera, format_diagnostic, measure_capture,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeCapture:
    def __init__(self, clock):
        self.clock = clock
        self.calls = 0
        self.released = False
        self.empty_at = None
        self.fail_at = None
        self.slow_frames = 0
        self.props = {1: 640, 2: 480, 3: 30, 4: 1196444237, 5: 4}
        self.settings = []

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.settings.append((prop, value))
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props[prop]

    def read(self):
        self.calls += 1
        self.clock.now += 0.5 if self.calls <= self.slow_frames else 0.05
        if self.calls == self.fail_at:
            raise OSError("disconnected")
        if self.calls == self.empty_at:
            return True, SimpleNamespace(size=0)
        return True, SimpleNamespace(size=10, shape=(int(self.props[2]), int(self.props[1]), 3))

    def release(self):
        self.released = True


def fake_cv2(capture):
    return SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2,
        CAP_PROP_FPS=3, CAP_PROP_FOURCC=4, CAP_PROP_BUFFERSIZE=5,
        CAP_V4L2=200, CAP_ANY=0,
        VideoWriter_fourcc=lambda *tag: sum(ord(ch) << (8*i) for i,ch in enumerate(tag)),
        VideoCapture=Mock(return_value=capture),
    )


class MeasurementTests(unittest.TestCase):
    def test_measurement_reports_rate_and_latency(self):
        clock = FakeClock()
        result = measure_capture(FakeCapture(clock), 1.0, clock)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.fps, 20)
        self.assertAlmostEqual(result.read_p95_ms, 50)
        self.assertAlmostEqual(result.interval_max_ms, 50)
        self.assertEqual(result.frame_width, 640)

    def test_empty_frames_and_exceptions_are_failures(self):
        for attribute in ("empty_at", "fail_at"):
            with self.subTest(attribute=attribute):
                clock = FakeClock()
                capture = FakeCapture(clock)
                setattr(capture, attribute, 3)
                result = measure_capture(capture, 1.0, clock)
                self.assertFalse(result.valid)
                self.assertEqual(result.frames, 2)
                self.assertEqual(result.failed_reads, 1)

    def test_warmup_is_not_included_and_capture_released(self):
        clock = FakeClock()
        capture = FakeCapture(clock)
        capture.slow_frames = 2
        cv2 = fake_cv2(capture)
        report = collect_diagnostic(
            cv2,
            9,
            DiagnosticSettings(duration_s=1, warmup_s=1, camera_backend="v4l2"),
            clock,
        )
        self.assertTrue(report["valid"])
        self.assertAlmostEqual(report["measurement"]["fps"], 20)
        self.assertEqual(report["warmup"]["frames"], 2)
        self.assertTrue(capture.released)
        cv2.VideoCapture.assert_called_once_with(9, 200)
        self.assertIn("below 80%", format_diagnostic(report))

    def test_open_failure_releases_capture(self):
        clock = FakeClock()
        capture = FakeCapture(clock)
        capture.isOpened = lambda: False
        with self.assertRaisesRegex(OSError, "Could not open"):
            collect_diagnostic(fake_cv2(capture), 9, DiagnosticSettings(), clock)
        self.assertTrue(capture.released)

    def test_warmup_failure_releases_capture(self):
        clock = FakeClock()
        capture = FakeCapture(clock)
        capture.empty_at = 1
        with self.assertRaises(OSError):
            collect_diagnostic(fake_cv2(capture), 0, DiagnosticSettings(), clock)
        self.assertTrue(capture.released)

    def test_failed_measurement_returns_nonzero(self):
        with patch("vision_robot_arm.vision.camera_diagnostics.collect_diagnostic", side_effect=OSError("offline")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(diagnose_camera(Mock(), 9), 1)

    def test_no_single_buffer_for_application_or_diagnostic(self):
        clock = FakeClock()
        capture = FakeCapture(clock)
        cv2 = fake_cv2(capture)
        open_camera_capture(0, cv2, 200)
        configure_camera(cv2, capture, DiagnosticSettings())
        apply_diagnostic_mode(cv2, capture, DiagnosticSettings())
        self.assertFalse(any(prop == cv2.CAP_PROP_BUFFERSIZE for prop, _ in capture.settings))
        self.assertEqual(capture.get(cv2.CAP_PROP_BUFFERSIZE), 4)

    def test_native_size_diagnostic_does_not_set_dimensions(self):
        clock = FakeClock()
        capture = FakeCapture(clock)
        cv2 = fake_cv2(capture)
        apply_diagnostic_mode(cv2, capture, DiagnosticSettings(width=0, height=0))
        self.assertFalse(any(prop in (1, 2) for prop, _ in capture.settings))

    def test_diagnostic_settings_validate_before_open(self):
        for field, values in {
            "duration_s": [0, -1, float("nan"), float("inf"), 61],
            "warmup_s": [-1, float("nan"), 31],
            "camera_fps": [0, float("inf"), float("nan")],
            "width": [-1, True, 1.5, 0],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    cv2 = Mock()
                    with self.assertRaises(ValueError):
                        collect_diagnostic(cv2, 0, replace(DiagnosticSettings(), **{field: value}))
                    cv2.VideoCapture.assert_not_called()


class DiagnosticCliTests(unittest.TestCase):
    def test_diagnostic_forwards_settings_without_loading_models(self):
        with (
            patch("sys.argv", ["main.py", "--diagnose-camera", "8", "--width", "1280", "--height", "720", "--fps", "25", "--camera-format", "yuyv"]),
            patch("vision_robot_arm.cli.load_camera_dependency", return_value="cv2") as load,
            patch("vision_robot_arm.cli.diagnose_camera", return_value=0) as diagnose,
            patch("vision_robot_arm.cli.run_app") as app,
            patch("vision_robot_arm.core.runtime.load_runtime_dependencies") as runtime,
        ):
            self.assertEqual(main(), 0)
            settings = diagnose.call_args.kwargs["settings"]
            self.assertEqual((settings.width, settings.height, settings.camera_fps), (1280, 720, 25))
            self.assertEqual(settings.camera_format, "yuyv")
            load.assert_called_once()
            runtime.assert_not_called()
            app.assert_not_called()

    def test_invalid_flags_do_not_open_camera(self):
        for args in (["--diagnose-camera", "-1"],
                     ["--diagnose-camera", "0", "--diagnostic-seconds", "nan"],
                     ["--diagnose-camera", "0", "--video", "clip.mp4"],
                     ["--diagnose-camera", "0", "--list-cameras"]):
            with self.subTest(args=args), patch("sys.argv", ["main.py", *args]), patch("vision_robot_arm.cli.load_camera_dependency") as load, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()

    def test_listing_only_loads_camera_dependency(self):
        with patch("sys.argv", ["main.py", "--list-cameras"]), patch("vision_robot_arm.cli.load_camera_dependency", return_value="cv2"), patch("vision_robot_arm.cli.discover_cameras", return_value=[]) as discover, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
            discover.assert_called_once_with("cv2")
