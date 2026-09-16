"""Small drawing primitives for the dashboard, in OpenCV's BGR colour order."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (22, 20, 19)
SURFACE = (30, 28, 27)
SURFACE_RAISED = (38, 36, 35)
HOVER = (47, 44, 42)
BORDER = (53, 49, 46)
TEXT = (241, 239, 237)
MUTED = (166, 159, 152)
DISABLED = (108, 103, 98)
ACCENT = (112, 169, 238)
ACCENT_INK = (26, 31, 42)
GREEN = (162, 200, 137)
RED = (134, 136, 240)
FONTS = Path(__file__).parent / "assets" / "fonts"


@lru_cache(maxsize=64)
def font(size: int, strong: bool = False) -> ImageFont.FreeTypeFont:
    name = "Lato-Semibold.ttf" if strong else "Lato-Regular.ttf"
    return ImageFont.truetype(str(FONTS / name), size)


@lru_cache(maxsize=512)
def text_mask(text: str, size: int, strong: bool) -> tuple[bytes, tuple[int, int]]:
    face = font(size, strong)
    left, top, right, bottom = face.getbbox(text)
    image = Image.new("L", (max(1, right - left), max(1, bottom - top)))
    ImageDraw.Draw(image).text((-left, -top), text, font=face, fill=255)
    return image.tobytes(), image.size


class Painter:
    """Cache glyph masks and blend only their bounds, not the entire video frame."""

    def __init__(self, cv2: Any, np: Any, canvas: Any, scale: float = 1.0) -> None:
        self.cv2, self.np, self.canvas, self.scale = cv2, np, canvas, scale

    def px(self, value: float) -> int:
        return max(1, round(value * self.scale))

    def measure(self, text: str, size: float = 14, strong: bool = False) -> int:
        return round(font(max(11, self.px(size)), strong).getlength(text))

    def elide(
        self, text: str, width: int, size: float = 14, strong: bool = False
    ) -> str:
        if self.measure(text, size, strong) <= width:
            return text
        if self.measure("…", size, strong) > width:
            return ""
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.measure(text[:mid] + "…", size, strong) <= width:
                low = mid
            else:
                high = mid - 1
        return text[:low].rstrip() + "…"

    def text(
        self,
        text: str,
        x: int,
        y: int,
        *,
        size: float = 14,
        color: tuple = TEXT,
        strong: bool = False,
        width: int | None = None,
        align: str = "left",
    ) -> None:
        """Draw text from its top edge; elide to the supplied available width."""
        if width is not None:
            text = self.elide(text, width, size, strong)
        if not text:
            return
        data, (w, h) = text_mask(text, max(11, self.px(size)), strong)
        if align == "right":
            x -= w
        elif align == "center":
            x -= w // 2
        left, top = max(0, x), max(0, y)
        right, bottom = (
            min(self.canvas.shape[1], x + w),
            min(self.canvas.shape[0], y + h),
        )
        if right <= left or bottom <= top:
            return
        alpha = self.np.frombuffer(data, dtype=self.np.uint8).reshape(h, w)
        alpha = (
            alpha[top - y : bottom - y, left - x : right - x, None].astype(
                self.np.float32
            )
            / 255
        )
        region = self.canvas[top:bottom, left:right]
        region[:] = region * (1 - alpha) + self.np.asarray(color) * alpha

    def wrap(self, text: str, width: int, size: float = 12) -> tuple[str, ...]:
        lines: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if current and self.measure(candidate, size) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return tuple(self.elide(line, width, size) for line in lines)

    def box(
        self,
        rect: Any,
        fill: tuple = SURFACE,
        *,
        radius: float = 12,
        border: tuple | None = BORDER,
    ) -> None:
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        if w <= 0 or h <= 0:
            return
        r = min(self.px(radius), w // 2, h // 2)
        self._rounded(x, y, w, h, r, border or fill)
        if border:
            self._rounded(x + 1, y + 1, w - 2, h - 2, max(0, r - 1), fill)

    def _rounded(self, x: int, y: int, w: int, h: int, r: int, color: tuple) -> None:
        cv = self.cv2
        cv.rectangle(self.canvas, (x + r, y), (x + w - 1 - r, y + h - 1), color, -1)
        cv.rectangle(self.canvas, (x, y + r), (x + w - 1, y + h - 1 - r), color, -1)
        for cx, cy in (
            (x + r, y + r),
            (x + w - 1 - r, y + r),
            (x + r, y + h - 1 - r),
            (x + w - 1 - r, y + h - 1 - r),
        ):
            cv.circle(self.canvas, (cx, cy), r, color, -1, cv.LINE_AA)

    def line(
        self, start: tuple, end: tuple, color: tuple = BORDER, width: float = 1
    ) -> None:
        self.cv2.line(self.canvas, start, end, color, self.px(width), self.cv2.LINE_AA)

    def dot(self, x: int, y: int, color: tuple = GREEN, radius: float = 3) -> None:
        self.cv2.circle(
            self.canvas, (x, y), self.px(radius), color, -1, self.cv2.LINE_AA
        )
