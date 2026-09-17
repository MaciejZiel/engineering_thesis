"""Camera device discovery, platform backend selection, and high-performance configuration."""

from __future__ import annotations

import os
import platform
import struct
from dataclasses import dataclass
from typing import Any, Sequence

CAMERA_FPS_TOLERANCE = 0.8

# Standard 16:9 and 4:3 resolutions ordered by pixel count (highest first).
STANDARD_CAMERA_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (3840, 2160),  # 4K UHD
    (2560, 1440),  # 2K QHD
    (1920, 1080),  # 1080p Full HD
    (1600, 1200),  # UXGA
    (1280, 720),   # 720p HD
    (960, 540),    # qHD
    (800, 600),    # SVGA
    (640, 480),    # VGA
    (320, 240),    # QVGA
)

BACKEND_CHOICES: tuple[str, ...] = (
    "auto",
    "v4l2",
    "dshow",
    "msmf",
    "avfoundation",
    "any",
)
FORMAT_CHOICES: tuple[str, ...] = ("auto", "mjpg", "yuyv", "nv12", "h264")


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    path: str | None = None
    is_capture: bool = True
    formats: tuple[str, ...] = ()
    resolutions: tuple[tuple[int, int], ...] = ()
    max_fps: float = 30.0
    backend_name: str = "auto"

    def summary(self) -> str:
        details = []
        if self.resolutions:
            max_w, max_h = self.resolutions[0]
            details.append(f"Max {max_w}x{max_h}")
        if self.max_fps > 0:
            details.append(f"{self.max_fps:g} FPS")
        if self.formats:
            details.append(", ".join(self.formats))
        desc = f" ({'; '.join(details)})" if details else ""
        path_str = f" [{self.path}]" if self.path else ""
        return f"Camera {self.index}: {self.name}{path_str}{desc}"


@dataclass(frozen=True)
class CameraMode:
    width: int
    height: int
    fps: float
    codec: str
    device_name: str = ""
    backend: str = ""

    def describe(self) -> str:
        device_part = f" - {self.device_name}" if self.device_name else ""
        return (
            f"{self.width}x{self.height} @ {self.fps:g} FPS ({self.codec or 'unknown'}){device_part}"
        )


def _query_v4l2_device(dev_path: str, index: int) -> CameraDevice | None:
    """Query Linux V4L2 ioctls for capability, device card name, and supported pixel formats."""
    try:
        import fcntl
    except ImportError:
        return None

    VIDIOC_QUERYCAP = 0x80685600
    VIDIOC_ENUM_FMT = 0xC0405602
    V4L2_CAP_VIDEO_CAPTURE = 0x00000001
    V4L2_CAP_STREAMING = 0x04000000

    try:
        fd = os.open(dev_path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None

    try:
        cap_buf = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap_buf)
        driver, card, bus_info, version, capabilities, device_caps = struct.unpack(
            "16s32s32sIII", cap_buf[:92]
        )
        card_name = card.decode("ascii", errors="ignore").strip("\x00")
        caps = device_caps if (capabilities & 0x80000000) else capabilities
        is_capture = bool(caps & V4L2_CAP_VIDEO_CAPTURE)
        is_streaming = bool(caps & V4L2_CAP_STREAMING)

        if not is_capture:
            return None

        formats: list[str] = []
        fmt_index = 0
        while True:
            fmt_buf = struct.pack("III32sIIIII", fmt_index, 1, 0, b"\x00" * 32, 0, 0, 0, 0, 0)
            try:
                res = fcntl.ioctl(fd, VIDIOC_ENUM_FMT, fmt_buf)
                _, _, _, desc, fourcc, _, _, _, _ = struct.unpack("III32sIIIII", res)
                codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
                if codec and codec not in formats:
                    formats.append(codec)
                fmt_index += 1
            except OSError:
                break

        return CameraDevice(
            index=index,
            name=card_name or f"V4L2 Camera {index}",
            path=dev_path,
            is_capture=is_capture and is_streaming,
            formats=tuple(formats),
            backend_name="v4l2",
        )
    except Exception:
        return None
    finally:
        os.close(fd)


def _discover_linux_v4l2_cameras() -> list[CameraDevice]:
    """Scan /sys/class/video4linux to discover real V4L2 capture cameras."""
    cameras: list[CameraDevice] = []
    v4l_dir = "/sys/class/video4linux"
    if not os.path.exists(v4l_dir):
        return cameras

    for item in sorted(os.listdir(v4l_dir)):
        if not item.startswith("video"):
            continue
        try:
            index = int(item[5:])
        except ValueError:
            continue
        dev_path = f"/dev/{item}"
        device = _query_v4l2_device(dev_path, index)
        if device and device.is_capture:
            cameras.append(device)
    return cameras


def _discover_opencv_cameras(cv2: Any, max_probe_index: int = 6) -> list[CameraDevice]:
    """Probe OpenCV camera indices to discover available cameras across all operating systems."""
    cameras: list[CameraDevice] = []
    backend = _default_cv2_backend(cv2)

    for index in range(max_probe_index):
        cap = None
        try:
            cap = (
                cv2.VideoCapture(index, backend)
                if backend is not None
                else cv2.VideoCapture(index)
            )
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                continue

            width = max(0, round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = max(0, round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = max(0.0, float(cap.get(cv2.CAP_PROP_FPS)))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")

            resolutions = ((width, height),) if width > 0 and height > 0 else ()
            formats = (codec,) if codec else ()

            cameras.append(
                CameraDevice(
                    index=index,
                    name=f"Camera {index}",
                    is_capture=True,
                    formats=formats,
                    resolutions=resolutions,
                    max_fps=fps,
                    backend_name=_backend_name(backend, cv2),
                )
            )
        except Exception:
            continue
        finally:
            if cap is not None:
                cap.release()
    return cameras


def discover_cameras(cv2: Any = None, max_probe_index: int = 6) -> list[CameraDevice]:
    """Find all available video capture devices on the system."""
    if platform.system() == "Linux":
        linux_devices = _discover_linux_v4l2_cameras()
        if linux_devices:
            return linux_devices

    if cv2 is not None:
        return _discover_opencv_cameras(cv2, max_probe_index)
    return []


def _default_cv2_backend(cv2: Any) -> int | None:
    """Select the best OpenCV video backend for the current platform."""
    sys_name = platform.system()
    if sys_name == "Linux" and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    if sys_name == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    if sys_name == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        return cv2.CAP_AVFOUNDATION
    return None


def resolve_cv2_backend(cv2: Any, backend_choice: str = "auto") -> int | None:
    """Resolve backend choice string to OpenCV VideoCapture backend constant."""
    choice = (backend_choice or "auto").lower()
    if choice == "auto":
        return _default_cv2_backend(cv2)
    if choice == "v4l2" and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    if choice == "dshow" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    if choice == "msmf" and hasattr(cv2, "CAP_MSMF"):
        return cv2.CAP_MSMF
    if choice == "avfoundation" and hasattr(cv2, "CAP_AVFOUNDATION"):
        return cv2.CAP_AVFOUNDATION
    if choice == "any" and hasattr(cv2, "CAP_ANY"):
        return cv2.CAP_ANY
    return _default_cv2_backend(cv2)


def _backend_name(backend: int | None, cv2: Any) -> str:
    if backend is None:
        return "any"
    if hasattr(cv2, "CAP_V4L2") and backend == cv2.CAP_V4L2:
        return "v4l2"
    if hasattr(cv2, "CAP_DSHOW") and backend == cv2.CAP_DSHOW:
        return "dshow"
    if hasattr(cv2, "CAP_MSMF") and backend == cv2.CAP_MSMF:
        return "msmf"
    if hasattr(cv2, "CAP_AVFOUNDATION") and backend == cv2.CAP_AVFOUNDATION:
        return "avfoundation"
    return "auto"


def open_camera_capture(
    camera_target: int | str,
    cv2: Any,
    preferred_backend: int | None = None,
) -> tuple[Any, int, str]:
    """Open a VideoCapture for camera_target with auto-fallback to available capture devices."""
    backend = (
        preferred_backend
        if preferred_backend is not None
        else _default_cv2_backend(cv2)
    )

    target_index: int | None = None
    if isinstance(camera_target, int):
        target_index = camera_target
    elif isinstance(camera_target, str) and camera_target.lower() != "auto":
        try:
            target_index = int(camera_target)
        except ValueError:
            target_index = None

    # If target is a specific index, try opening it first
    if target_index is not None:
        cap = (
            cv2.VideoCapture(target_index, backend)
            if backend is not None
            else cv2.VideoCapture(target_index)
        )
        if cap.isOpened():
            return cap, target_index, _backend_name(backend, cv2)
        # An explicit source must never silently switch to a different person/camera.
        return cap, target_index, _backend_name(backend, cv2)

    # Automatic selection scans available capture devices
    devices = discover_cameras(cv2)
    for dev in devices:
        if target_index is not None and dev.index == target_index:
            continue
        cap = (
            cv2.VideoCapture(dev.index, backend)
            if backend is not None
            else cv2.VideoCapture(dev.index)
        )
        if cap.isOpened():
            return cap, dev.index, _backend_name(backend, cv2)
        cap.release()

    # Final attempt: try index 0 with generic backend
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap, 0, "any"

    return cap, 0, "failed"


def configure_camera(
    cv2: Any,
    capture: Any,
    config: Any,
    device_name: str = "",
) -> CameraMode:
    """Configure camera capture for maximum quality, target frame rate, and lowest latency."""
    # Keep driver buffering: a single V4L2 buffer can halve frame delivery rate.
    # 2. Select preferred pixel format (MJPG by default to enable high resolution at 30+ FPS)
    requested_format = getattr(config, "camera_format", "auto")
    codec_tag = "MJPG"
    if requested_format and requested_format.lower() != "auto":
        codec_tag = requested_format.upper()

    try:
        fourcc_code = cv2.VideoWriter_fourcc(*codec_tag)
        capture.set(cv2.CAP_PROP_FOURCC, fourcc_code)
    except Exception:
        pass
    capture.set(cv2.CAP_PROP_FPS, config.camera_fps)

    # 3. Build resolution candidates list
    candidates: list[tuple[int, int]] = []
    if config.width > 0 and config.height > 0:
        candidates.append((config.width, config.height))
        candidates.extend(
            size
            for size in STANDARD_CAMERA_RESOLUTIONS
            if size[0] <= config.width and size[1] <= config.height
        )
    else:
        # Native maximum quality mode (probe from 4K downwards)
        candidates.extend(STANDARD_CAMERA_RESOLUTIONS)

    unique_candidates = list(dict.fromkeys(candidates))
    best_mode = _read_camera_mode(cv2, capture, device_name)
    target_fps = config.camera_fps

    for width, height in unique_candidates:
        if width > 0 and height > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, target_fps)

        mode = _read_camera_mode(cv2, capture, device_name)
        if mode.fps <= 0 or mode.fps >= target_fps * CAMERA_FPS_TOLERANCE:
            best_mode = mode
            break
        best_mode = mode

    return best_mode


def _read_camera_mode(cv2: Any, capture: Any, device_name: str = "") -> CameraMode:
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")
    return CameraMode(
        width=max(0, round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        height=max(0, round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        fps=max(0.0, float(capture.get(cv2.CAP_PROP_FPS))),
        codec=codec,
        device_name=device_name,
    )


def format_camera_list(cameras: Sequence[CameraDevice]) -> str:
    """Format camera devices list into a clear terminal output."""
    if not cameras:
        return "No video capture devices detected."
    lines = ["Available Cameras:"]
    for cam in cameras:
        prefix = f"  [{cam.index}]"
        formats_str = f" [formats: {', '.join(cam.formats)}]" if cam.formats else ""
        lines.append(f"{prefix} {cam.name}{formats_str}")
    return "\n".join(lines)


def diagnose_camera(cv2: Any, camera_target: int | str, test_duration_s: float = 5.0) -> int:
    """Compatibility entry point for the camera-only diagnostic."""
    from vision_robot_arm.vision.camera_diagnostics import diagnose_camera as run_diagnostic

    return run_diagnostic(cv2, camera_target, test_duration_s)
