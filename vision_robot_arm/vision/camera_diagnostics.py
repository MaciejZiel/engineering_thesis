"""Camera-only measurements; no inference, UI, exposure changes, or image storage."""

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable

from vision_robot_arm.vision.camera import (
    CAMERA_FPS_TOLERANCE,
    CameraMode,
    _read_camera_mode,
    open_camera_capture,
    resolve_cv2_backend,
)


@dataclass(frozen=True)
class DiagnosticSettings:
    width: int = 1920
    height: int = 1080
    camera_fps: float = 30.0
    camera_format: str = "auto"
    camera_backend: str = "auto"
    duration_s: float = 5.0
    warmup_s: float = 2.0

    def validate(self) -> None:
        for name, value in (("width", self.width), ("height", self.height)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (self.width == 0) != (self.height == 0):
            raise ValueError("width and height must both be zero to retain the native size")
        for name, value, low, high in (
            ("fps", self.camera_fps, 1.0, 120.0),
            ("diagnostic-seconds", self.duration_s, 1.0, 60.0),
            ("diagnostic-warmup", self.warmup_s, 0.0, 30.0),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and between {low:g} and {high:g}")
        if self.camera_format not in ("auto", "mjpg", "yuyv", "nv12", "h264"):
            raise ValueError("Unsupported camera format")
        if self.camera_backend not in ("auto", "any", "v4l2", "dshow", "msmf", "avfoundation"):
            raise ValueError("Unsupported camera backend")


@dataclass(frozen=True)
class CaptureMeasurement:
    frames: int
    elapsed_s: float
    failed_reads: int
    fps: float
    read_mean_ms: float | None
    read_p95_ms: float | None
    read_max_ms: float | None
    interval_p95_ms: float | None
    interval_max_ms: float | None
    frame_width: int | None
    frame_height: int | None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.frames >= 2 and self.failed_reads == 0 and self.error is None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[math.ceil(len(values) * 0.95) - 1] * 1000


def measure_capture(
    capture: Any,
    duration_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> CaptureMeasurement:
    """Measure host delivery, not sensor exposure or end-to-end image age.

    The duration is a sampling window, not a timeout on native VideoCapture.read().
    A blocked driver requires process-level supervision by the caller.
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Measurement duration must be finite and positive")
    started = clock()
    previous = None
    reads: list[float] = []
    intervals: list[float] = []
    failed = 0
    error = None
    width = height = None
    while clock() - started < duration_s:
        before = clock()
        try:
            ok, frame = capture.read()
            after = clock()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                failed += 1
                error = "Camera returned an empty or unavailable frame"
                break
            current_height, current_width = frame.shape[:2]
            if current_width <= 0 or current_height <= 0:
                failed += 1
                error = "Camera returned invalid frame dimensions"
                break
            if width is not None and (width, height) != (current_width, current_height):
                failed += 1
                error = "Camera changed frame dimensions during measurement"
                break
            width, height = current_width, current_height
        except Exception as exc:
            failed += 1
            error = f"Camera read failed: {exc}"
            break
        reads.append(after - before)
        if previous is not None:
            intervals.append(after - previous)
        previous = after
    elapsed = max(0.0, clock() - started)
    return CaptureMeasurement(
        frames=len(reads), elapsed_s=elapsed, failed_reads=failed,
        fps=len(reads) / elapsed if elapsed else 0.0,
        read_mean_ms=sum(reads) / len(reads) * 1000 if reads else None,
        read_p95_ms=_p95(reads), read_max_ms=max(reads) * 1000 if reads else None,
        interval_p95_ms=_p95(intervals),
        interval_max_ms=max(intervals) * 1000 if intervals else None,
        frame_width=width, frame_height=height, error=error,
    )


def apply_diagnostic_mode(cv2: Any, capture: Any, settings: DiagnosticSettings) -> dict[str, bool]:
    """Request exactly one mode; report driver acceptance instead of silently probing."""
    settings.validate()
    requested = [
        ("format", cv2.CAP_PROP_FOURCC,
         cv2.VideoWriter_fourcc(*(settings.camera_format.upper() if settings.camera_format != "auto" else "MJPG"))),
    ]
    if settings.width and settings.height:
        requested.extend((("width", cv2.CAP_PROP_FRAME_WIDTH, settings.width),
                          ("height", cv2.CAP_PROP_FRAME_HEIGHT, settings.height)))
    requested.append(("fps", cv2.CAP_PROP_FPS, settings.camera_fps))
    accepted = {}
    for name, prop, value in requested:
        try:
            accepted[name] = bool(capture.set(prop, value))
        except Exception:
            accepted[name] = False
    return accepted


def collect_diagnostic(
    cv2: Any,
    camera_target: int | str,
    settings: DiagnosticSettings,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    settings.validate()
    capture = None
    try:
        capture, index, backend = open_camera_capture(
            camera_target, cv2, resolve_cv2_backend(cv2, settings.camera_backend)
        )
        if not capture.isOpened():
            raise OSError(f"Could not open camera {camera_target}")
        accepted = apply_diagnostic_mode(cv2, capture, settings)
        warmup = measure_capture(capture, settings.warmup_s, clock) if settings.warmup_s else None
        if warmup is not None and not warmup.valid:
            raise OSError(warmup.error or "Not enough frames during warmup")
        measurement = measure_capture(capture, settings.duration_s, clock)
        mode = _read_camera_mode(cv2, capture)
        return {
            "schema_version": 1,
            "camera": index, "backend": backend,
            "requested": asdict(settings), "driver_accepted": accepted,
            "reported_mode": asdict(mode),
            "warmup": asdict(warmup) if warmup else None,
            "measurement": asdict(measurement),
            "valid": measurement.valid,
            "target_met": measurement.valid and measurement.fps >= settings.camera_fps * CAMERA_FPS_TOLERANCE,
        }
    finally:
        if capture is not None:
            capture.release()


def format_diagnostic(report: dict[str, Any]) -> str:
    measurement = report["measurement"]
    mode = CameraMode(**report["reported_mode"])
    lines = [
        f"Camera {report['camera']} [{report['backend']}]",
        f"Driver-reported mode: {mode.describe()}",
        f"Delivered frame size: {measurement['frame_width']}x{measurement['frame_height']}",
        f"Measured FPS: {measurement['fps']:.2f} after {report['requested']['warmup_s']:g}s warmup",
        f"Frames: {measurement['frames']}; failed reads: {measurement['failed_reads']}; elapsed: {measurement['elapsed_s']:.2f}s",
    ]
    for name, key in (("Read mean", "read_mean_ms"), ("Read p95", "read_p95_ms"),
                      ("Read max", "read_max_ms"), ("Frame interval p95", "interval_p95_ms"),
                      ("Frame interval max", "interval_max_ms")):
        value = measurement[key]
        lines.append(f"{name}: {value:.2f} ms" if value is not None else f"{name}: unavailable")
    rejected = [name for name, accepted in report["driver_accepted"].items() if not accepted]
    if rejected:
        lines.append(f"Settings not accepted by driver: {', '.join(rejected)}")
    if not report["valid"]:
        lines.append(f"Measurement failed: {measurement['error'] or 'not enough frames'}")
    elif not report["target_met"]:
        lines.append("Measured delivery is below 80% of requested FPS; no cause or alternative mode is inferred.")
    lines.append("Read timing is host delivery latency, not sensor-to-display latency. No images were saved.")
    return "\n".join(lines)


def diagnose_camera(
    cv2: Any,
    camera_target: int | str,
    test_duration_s: float = 5.0,
    *,
    settings: DiagnosticSettings | None = None,
) -> int:
    try:
        report = collect_diagnostic(cv2, camera_target, settings or DiagnosticSettings(duration_s=test_duration_s))
    except (OSError, ValueError) as error:
        print(f"Camera diagnostic failed: {error}")
        return 1
    print(format_diagnostic(report))
    return 0 if report["valid"] else 1
