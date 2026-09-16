from __future__ import annotations

from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class DisplayMetrics:
    width: int
    height: int
    dpi: int = 96

    @property
    def scale(self) -> float:
        return self.dpi / 96.0


def enable_high_dpi_awareness() -> bool:
    """Ask Windows for physical pixels so the OS does not blur the OpenCV window."""
    if sys.platform != "win32":
        return False

    try:
        import ctypes

        per_monitor_v2 = ctypes.c_void_p(-4)
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(per_monitor_v2):
            return True
    except (AttributeError, OSError):
        pass

    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return True
    except (AttributeError, OSError):
        return False


def primary_work_area(fallback: tuple[int, int] = (1920, 1080)) -> DisplayMetrics:
    """Return the usable primary-monitor area in physical pixels."""
    if sys.platform != "win32":
        return DisplayMetrics(*fallback)

    try:
        import ctypes
        from ctypes import wintypes

        class Rect(ctypes.Structure):
            _fields_ = (
                ("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG),
            )

        rect = Rect()
        spi_get_work_area = 0x0030
        if not ctypes.windll.user32.SystemParametersInfoW(
            spi_get_work_area,
            0,
            ctypes.byref(rect),
            0,
        ):
            return DisplayMetrics(*fallback)

        get_dpi = getattr(ctypes.windll.user32, "GetDpiForSystem", None)
        dpi = int(get_dpi()) if get_dpi is not None else 96
        return DisplayMetrics(rect.right - rect.left, rect.bottom - rect.top, dpi)
    except (AttributeError, OSError, ValueError):
        return DisplayMetrics(*fallback)


def preferred_dashboard_size(
    display: DisplayMetrics,
    coverage: float = 0.90,
) -> tuple[int, int]:
    """Choose a readable initial window while leaving room for window chrome."""
    if not 0.5 <= coverage <= 1.0:
        raise ValueError("coverage must be between 0.5 and 1.0")
    width = min(display.width, max(960, round(display.width * coverage)))
    height = min(display.height, max(540, round(display.height * coverage)))
    return width, height
