from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vision_robot_arm.core.config import ANGLE_MODE, BOTH_MODE, LANDMARK_MODE


Color = tuple[int, int, int]

ACTION_CALIBRATE = "calibrate"
ACTION_FULLSCREEN = "fullscreen"
ACTION_MODE = "mode"
ACTION_QUIT = "quit"
ACTION_RECORD = "record"

BACKGROUND: Color = (14, 18, 24)
PANEL: Color = (23, 29, 37)
PANEL_ALT: Color = (29, 37, 47)
BORDER: Color = (55, 66, 78)
ORANGE: Color = (0, 142, 255)
ORANGE_SOFT: Color = (30, 178, 255)
GREEN: Color = (80, 210, 130)
RED: Color = (80, 90, 235)
TEXT: Color = (238, 242, 247)
MUTED: Color = (148, 160, 174)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom


@dataclass(frozen=True)
class DashboardButton:
    action: str
    label: str
    shortcut: str
    rect: Rect
    active: bool = False
    danger: bool = False


def fit_inside(source_size: tuple[int, int], target: Rect) -> Rect:
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0 or target.width <= 0 or target.height <= 0:
        return Rect(target.x, target.y, 0, 0)
    scale = min(target.width / source_width, target.height / source_height)
    width = max(1, round(source_width * scale))
    height = max(1, round(source_height * scale))
    return Rect(
        target.x + (target.width - width) // 2,
        target.y + (target.height - height) // 2,
        width,
        height,
    )


def cycle_output_mode(mode: str) -> str:
    order = (ANGLE_MODE, LANDMARK_MODE, BOTH_MODE)
    try:
        return order[(order.index(mode) + 1) % len(order)]
    except ValueError:
        return ANGLE_MODE


class DashboardUi:
    """Single-window OpenCV dashboard with a camera view and embedded robot preview."""

    def __init__(self, cv2: Any, np: Any, window_name: str) -> None:
        self._cv2 = cv2
        self._np = np
        self.window_name = window_name
        self._buttons: tuple[DashboardButton, ...] = ()
        self._pending_action: str | None = None
        self._fullscreen = False

    @property
    def fullscreen(self) -> bool:
        return self._fullscreen

    @property
    def buttons(self) -> tuple[DashboardButton, ...]:
        return self._buttons

    def open(self, width: int, height: int) -> None:
        self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)
        self._cv2.resizeWindow(self.window_name, min(width, 1600), min(height, 900))
        self._cv2.setMouseCallback(self.window_name, self._on_mouse)

    def consume_action(self) -> str | None:
        action = self._pending_action
        self._pending_action = None
        return action

    def toggle_fullscreen(self) -> bool:
        self._fullscreen = not self._fullscreen
        value = self._cv2.WINDOW_FULLSCREEN if self._fullscreen else self._cv2.WINDOW_NORMAL
        self._cv2.setWindowProperty(self.window_name, self._cv2.WND_PROP_FULLSCREEN, value)
        return self._fullscreen

    def render(
        self,
        camera_frame: Any,
        simulation_frame: Any,
        *,
        mode: str,
        person_detected: bool,
        calibrated: bool,
        recording: bool,
        robot_label: str,
        gestures: tuple[str, ...],
        status_lines: tuple[str, ...],
        tracking_quality: float,
        fps: float,
        source_label: str,
    ) -> Any:
        camera_height, camera_width = camera_frame.shape[:2]
        width = max(1280, camera_width)
        height = max(720, round(width * 9 / 16))
        if height > 1080:
            height = 1080
            width = 1920

        canvas = self._np.full((height, width, 3), BACKGROUND, dtype=self._np.uint8)
        scale = height / 1080.0
        margin = max(10, round(20 * scale))
        gap = max(8, round(16 * scale))
        header_height = max(58, round(84 * scale))
        footer_height = max(74, round(108 * scale))
        body_top = header_height
        body_height = height - header_height - footer_height
        sidebar_width = max(360, round(width * 0.31))
        camera_width_area = width - sidebar_width - 3 * margin

        self._draw_header(
            canvas,
            source_label=source_label,
            fps=fps,
            person_detected=person_detected,
            robot_label=robot_label,
            scale=scale,
        )

        camera_panel = Rect(margin, body_top, camera_width_area, body_height - margin)
        sidebar = Rect(camera_panel.right + gap, body_top, sidebar_width, body_height - margin)
        self._panel(canvas, camera_panel)
        self._text(
            canvas,
            "LIVE CAMERA",
            (camera_panel.x + margin, camera_panel.y + round(31 * scale)),
            ORANGE_SOFT,
            0.54,
            2,
        )
        self._text(
            canvas,
            "MediaPipe pose + hand tracking",
            (camera_panel.right - round(292 * scale), camera_panel.y + round(31 * scale)),
            MUTED,
            0.42,
        )
        camera_target = Rect(
            camera_panel.x + 2,
            camera_panel.y + max(38, round(48 * scale)),
            camera_panel.width - 4,
            camera_panel.height - max(40, round(50 * scale)),
        )
        self._place_image(canvas, camera_frame, camera_target)
        self._corner_accents(canvas, camera_panel, scale)

        sim_height = max(260, round(sidebar.height * 0.50))
        simulation_panel = Rect(sidebar.x, sidebar.y, sidebar.width, sim_height)
        status_panel = Rect(
            sidebar.x,
            simulation_panel.bottom + gap,
            sidebar.width,
            sidebar.height - sim_height - gap,
        )
        self._draw_simulation_panel(canvas, simulation_frame, simulation_panel, scale)
        self._draw_status_panel(
            canvas,
            status_panel,
            person_detected=person_detected,
            calibrated=calibrated,
            recording=recording,
            robot_label=robot_label,
            gestures=gestures,
            status_lines=status_lines,
            tracking_quality=tracking_quality,
            scale=scale,
        )
        self._draw_footer(canvas, mode, recording, height - footer_height, scale)
        return canvas

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != self._cv2.EVENT_LBUTTONUP:
            return
        for button in self._buttons:
            if button.rect.contains(x, y):
                self._pending_action = button.action
                return

    def _draw_header(
        self,
        canvas: Any,
        *,
        source_label: str,
        fps: float,
        person_detected: bool,
        robot_label: str,
        scale: float,
    ) -> None:
        _, width = canvas.shape[:2]
        title_y = max(34, round(48 * scale))
        self._text(canvas, "MOTION TWIN", (round(22 * scale), title_y), TEXT, 0.86, 2)
        self._text(
            canvas,
            "DUAL UR7e  /  ORBBEC GEMINI 335Lg  /  JETSON ORIN AGX",
            (round(235 * scale), title_y),
            MUTED,
            0.45,
        )
        badges = (
            (f"CAMERA  {source_label}", True),
            ("PERSON  TRACKED" if person_detected else "PERSON  SEARCHING", person_detected),
            (f"ROBOT  {robot_label.upper()}", robot_label != "off"),
            (f"{fps:4.1f} FPS", fps >= 15.0),
        )
        x = width - round(18 * scale)
        for label, active in reversed(badges):
            badge_width = max(round(102 * scale), round((len(label) * 8 + 24) * scale))
            x -= badge_width
            self._badge(canvas, label, Rect(x, round(17 * scale), badge_width, round(40 * scale)), active)
            x -= round(8 * scale)
        divider_y = round(72 * scale)
        self._cv2.line(canvas, (0, divider_y), (width, divider_y), BORDER, 1, self._cv2.LINE_AA)

    def _draw_simulation_panel(self, canvas: Any, simulation: Any, panel: Rect, scale: float) -> None:
        self._panel(canvas, panel)
        self._text(
            canvas,
            "ROBOT DIGITAL TWIN",
            (panel.x + round(14 * scale), panel.y + round(28 * scale)),
            ORANGE_SOFT,
            0.5,
            2,
        )
        target = Rect(
            panel.x + 2,
            panel.y + max(35, round(42 * scale)),
            panel.width - 4,
            panel.height - max(37, round(44 * scale)),
        )
        self._place_image(canvas, simulation, target)

    def _draw_status_panel(
        self,
        canvas: Any,
        panel: Rect,
        *,
        person_detected: bool,
        calibrated: bool,
        recording: bool,
        robot_label: str,
        gestures: tuple[str, ...],
        status_lines: tuple[str, ...],
        tracking_quality: float,
        scale: float,
    ) -> None:
        self._panel(canvas, panel, PANEL_ALT)
        x = panel.x + round(16 * scale)
        y = panel.y + round(29 * scale)
        line = max(25, round(31 * scale))
        self._text(canvas, "SYSTEM STATUS", (x, y), TEXT, 0.52, 2)
        y += line
        rows = (
            (
                "VISION",
                f"TRACKING {tracking_quality:.0%}" if person_detected else "NO PERSON",
                person_detected,
            ),
            ("CALIBRATION", "READY" if calibrated else "PENDING", calibrated),
            ("RECORDING", "ACTIVE" if recording else "IDLE", recording),
            ("ROBOT LINK", robot_label.upper(), robot_label != "off"),
        )
        for name, value, active in rows:
            self._status_row(canvas, name, value, (x, y), active, scale)
            y += line

        if gestures and y < panel.bottom - line:
            self._text(canvas, "GESTURES", (x, y), MUTED, 0.4)
            y += line
            gesture_text = "  /  ".join(gesture.replace("_", " ").upper() for gesture in gestures[:3])
            self._text(canvas, gesture_text, (x, y), ORANGE_SOFT, 0.4, 1)
            y += line

        for status in status_lines[:2]:
            if y >= panel.bottom - round(12 * scale):
                break
            self._text(canvas, status[:58], (x, y), MUTED, 0.36)
            y += line

    def _draw_footer(self, canvas: Any, mode: str, recording: bool, top: int, scale: float) -> None:
        height, width = canvas.shape[:2]
        self._cv2.rectangle(canvas, (0, top), (width, height), PANEL, -1)
        self._cv2.line(canvas, (0, top), (width, top), BORDER, 1, self._cv2.LINE_AA)
        margin = max(10, round(20 * scale))
        gap = max(7, round(10 * scale))
        button_height = max(48, round(62 * scale))
        button_top = top + (height - top - button_height) // 2
        labels = (
            (ACTION_MODE, f"OUTPUT: {mode.upper()}", "1/2/3", False, False),
            (ACTION_CALIBRATE, "CALIBRATE", "C", False, False),
            (ACTION_RECORD, "STOP RECORDING" if recording else "START RECORDING", "R", recording, False),
            (ACTION_FULLSCREEN, "FULLSCREEN", "F", self._fullscreen, False),
            (ACTION_QUIT, "QUIT", "Q / ESC", False, True),
        )
        available = width - 2 * margin - gap * (len(labels) - 1)
        button_width = available // len(labels)
        buttons: list[DashboardButton] = []
        for index, (action, label, shortcut, active, danger) in enumerate(labels):
            rect = Rect(margin + index * (button_width + gap), button_top, button_width, button_height)
            button = DashboardButton(action, label, shortcut, rect, active, danger)
            buttons.append(button)
            self._button(canvas, button, scale)
        self._buttons = tuple(buttons)

    def _panel(self, canvas: Any, rect: Rect, color: Color = PANEL) -> None:
        self._cv2.rectangle(canvas, (rect.x, rect.y), (rect.right, rect.bottom), color, -1)
        self._cv2.rectangle(canvas, (rect.x, rect.y), (rect.right, rect.bottom), BORDER, 1, self._cv2.LINE_AA)

    def _badge(self, canvas: Any, label: str, rect: Rect, active: bool) -> None:
        color = GREEN if active else MUTED
        self._cv2.rectangle(canvas, (rect.x, rect.y), (rect.right, rect.bottom), PANEL_ALT, -1)
        self._cv2.rectangle(canvas, (rect.x, rect.y), (rect.right, rect.bottom), BORDER, 1, self._cv2.LINE_AA)
        center_y = rect.y + rect.height // 2
        self._cv2.circle(canvas, (rect.x + 13, center_y), 4, color, -1, self._cv2.LINE_AA)
        self._text(canvas, label, (rect.x + 24, center_y + 5), TEXT, 0.38)

    def _status_row(
        self,
        canvas: Any,
        name: str,
        value: str,
        origin: tuple[int, int],
        active: bool,
        scale: float,
    ) -> None:
        x, y = origin
        self._cv2.circle(
            canvas,
            (x + round(4 * scale), y - round(4 * scale)),
            max(2, round(4 * scale)),
            GREEN if active else MUTED,
            -1,
            self._cv2.LINE_AA,
        )
        self._text(canvas, name, (x + round(16 * scale), y), MUTED, 0.39)
        self._text(canvas, value, (x + round(150 * scale), y), TEXT if active else MUTED, 0.39, 1)

    def _button(self, canvas: Any, button: DashboardButton, scale: float) -> None:
        fill = (49, 57, 67)
        border = BORDER
        if button.active:
            fill, border = (0, 88, 155), ORANGE
        if button.danger:
            fill, border = (52, 38, 43), RED
        rect = button.rect
        self._cv2.rectangle(canvas, (rect.x, rect.y), (rect.right, rect.bottom), fill, -1)
        self._cv2.rectangle(canvas, (rect.x, rect.y), (rect.right, rect.bottom), border, 1, self._cv2.LINE_AA)
        self._text(
            canvas,
            button.label,
            (rect.x + round(14 * scale), rect.y + round(27 * scale)),
            TEXT,
            0.43,
            1,
        )
        self._text(
            canvas,
            button.shortcut,
            (rect.x + round(14 * scale), rect.bottom - round(9 * scale)),
            MUTED,
            0.31,
        )

    def _place_image(self, canvas: Any, image: Any, target: Rect) -> None:
        source_height, source_width = image.shape[:2]
        fitted = fit_inside((source_width, source_height), target)
        if fitted.width <= 0 or fitted.height <= 0:
            return
        resized = self._cv2.resize(image, (fitted.width, fitted.height), interpolation=self._cv2.INTER_AREA)
        canvas[fitted.y:fitted.bottom, fitted.x:fitted.right] = resized

    def _corner_accents(self, canvas: Any, rect: Rect, scale: float) -> None:
        length = max(12, round(25 * scale))
        thick = max(1, round(3 * scale))
        corners = (
            ((rect.x, rect.y), (rect.x + length, rect.y), (rect.x, rect.y + length)),
            ((rect.right, rect.y), (rect.right - length, rect.y), (rect.right, rect.y + length)),
            ((rect.x, rect.bottom), (rect.x + length, rect.bottom), (rect.x, rect.bottom - length)),
            (
                (rect.right, rect.bottom),
                (rect.right - length, rect.bottom),
                (rect.right, rect.bottom - length),
            ),
        )
        for corner, horizontal, vertical in corners:
            self._cv2.line(canvas, corner, horizontal, ORANGE, thick, self._cv2.LINE_AA)
            self._cv2.line(canvas, corner, vertical, ORANGE, thick, self._cv2.LINE_AA)

    def _text(
        self,
        canvas: Any,
        text: str,
        origin: tuple[int, int],
        color: Color,
        scale: float,
        thickness: int = 1,
    ) -> None:
        factor = canvas.shape[0] / 1080.0
        self._cv2.putText(
            canvas,
            text,
            origin,
            self._cv2.FONT_HERSHEY_SIMPLEX,
            max(0.28, scale * factor),
            color,
            thickness,
            self._cv2.LINE_AA,
        )
