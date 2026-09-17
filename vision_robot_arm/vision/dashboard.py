from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vision_robot_arm.core.config import ANGLE_MODE, BOTH_MODE, LANDMARK_MODE
from vision_robot_arm.core.display import preferred_dashboard_size, primary_work_area
from vision_robot_arm.vision.ui_style import (
    ACCENT,
    ACCENT_INK,
    BACKGROUND,
    BORDER,
    DISABLED,
    GREEN,
    HOVER,
    MUTED,
    RED,
    SURFACE,
    SURFACE_RAISED,
    TEXT,
    Painter,
)

ACTION_CALIBRATE = "calibrate"
ACTION_FULLSCREEN = "fullscreen"
ACTION_MODE = "mode"
ACTION_QUIT = "quit"
ACTION_RECORD = "record"
ACTION_DETAILS = "details"
ACTION_CONTROL = "control"
ACTION_JOG_NEGATIVE = "jog_negative"
ACTION_JOG_POSITIVE = "jog_positive"
ACTION_STOP = "stop"
ACTION_VIEW = "view"


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
    enabled: bool = True
    primary: bool = False


@dataclass(frozen=True)
class DashboardLayout:
    scale: float
    margin: int
    header: int
    camera: Rect
    preview: Rect
    status: Rect
    footer: Rect


def preview_target_rect(rect: Rect, px: Any) -> Rect:
    """Area inside the arm preview panel that shows the simulation image."""
    pad = px(20)
    return Rect(
        rect.x + pad, rect.y + px(82), rect.width - 2 * pad, rect.height - px(127)
    )


def preview_target(layout: DashboardLayout) -> Rect:
    return preview_target_rect(
        layout.preview, lambda value: max(1, round(value * layout.scale))
    )


def dashboard_layout(width: int, height: int) -> DashboardLayout:
    """Allocate panels from one scale; width as well as height limits density."""
    scale = max(0.65, min(width / 1440, height / 900, 2.0))
    px = lambda value: round(value * scale)
    margin, gap = px(24), px(20)
    header, body_top, footer_height = px(64), px(152), px(100)
    body_height = height - body_top - footer_height - margin
    sidebar_width = min(px(400), max(px(340), round(width * 0.28)))
    camera = Rect(
        margin, body_top, width - margin * 2 - gap - sidebar_width, body_height
    )
    preview_height = max(px(190), round(body_height * 0.56))
    preview = Rect(camera.right + gap, body_top, sidebar_width, preview_height)
    status = Rect(
        preview.x,
        preview.bottom + gap,
        sidebar_width,
        body_height - preview_height - gap,
    )
    return DashboardLayout(
        scale,
        margin,
        header,
        camera,
        preview,
        status,
        Rect(0, height - footer_height, width, footer_height),
    )


def fit_inside(source_size: tuple[int, int], target: Rect) -> Rect:
    source_width, source_height = source_size
    if (
        source_width <= 0
        or source_height <= 0
        or target.width <= 0
        or target.height <= 0
    ):
        return Rect(target.x, target.y, 0, 0)
    scale = min(target.width / source_width, target.height / source_height)
    width, height = (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
    )
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


def backend_label(backend: str) -> str:
    # Backend selection does not prove a robot connection or hardware feedback.
    return {
        "sim": "Simulation",
        "off": "Preview only",
        "none": "Preview only",
        "debug": "Debug output",
        "ur": "URScript output",
        "serial": "Serial output",
    }.get(backend, backend.replace("_", " ").capitalize())


class DashboardUi:
    """A quiet camera-first workspace; diagnostics are available on demand."""

    def __init__(self, cv2: Any, np: Any, window_name: str) -> None:
        self._cv2, self._np, self.window_name = cv2, np, window_name
        self._buttons: tuple[DashboardButton, ...] = ()
        self._pending_action: str | None = None
        self._fullscreen = False
        self._canvas_size: tuple[int, int] | None = None
        self._pointer = (-1, -1)
        self._pressed: str | None = None
        self._focus: str | None = None
        self._details = False
        self._workspace_focus = False

    @property
    def fullscreen(self) -> bool:
        return self._fullscreen

    @property
    def buttons(self) -> tuple[DashboardButton, ...]:
        return self._buttons

    def open(self) -> tuple[int, int]:
        size = preferred_dashboard_size(primary_work_area())
        flags = self._cv2.WINDOW_NORMAL | getattr(self._cv2, "WINDOW_FREERATIO", 0)
        self._cv2.namedWindow(self.window_name, flags)
        self._cv2.resizeWindow(self.window_name, *size)
        self._cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._canvas_size = size
        return size

    def sync_window_size(self) -> tuple[int, int] | None:
        cv_error = getattr(self._cv2, "error", Exception)
        try:
            _x, _y, width, height = self._cv2.getWindowImageRect(self.window_name)
        except (AttributeError, cv_error):
            return self._canvas_size
        if width >= 640 and height >= 480:
            self._canvas_size = (width, height)
        return self._canvas_size

    def consume_action(self) -> str | None:
        action, self._pending_action = self._pending_action, None
        return action

    @property
    def held_action(self) -> str | None:
        """Return a motion action only while its enabled button remains held."""
        if self._pressed not in (ACTION_JOG_NEGATIVE, ACTION_JOG_POSITIVE):
            return None
        button = next(
            (candidate for candidate in self._buttons if candidate.action == self._pressed),
            None,
        )
        if button is None or not button.enabled or not button.rect.contains(*self._pointer):
            return None
        return self._pressed

    @property
    def workspace_focus(self) -> bool:
        return self._workspace_focus

    def toggle_workspace_focus(self) -> bool:
        self._workspace_focus = not self._workspace_focus
        return self._workspace_focus

    def toggle_fullscreen(self) -> bool:
        self._fullscreen = not self._fullscreen
        value = (
            self._cv2.WINDOW_FULLSCREEN if self._fullscreen else self._cv2.WINDOW_NORMAL
        )
        self._cv2.setWindowProperty(
            self.window_name, self._cv2.WND_PROP_FULLSCREEN, value
        )
        return self._fullscreen

    def handle_key(self, key: int) -> None:
        enabled = [button.action for button in self._buttons if button.enabled]
        if key == 9 and enabled:
            current = enabled.index(self._focus) if self._focus in enabled else -1
            self._focus = enabled[(current + 1) % len(enabled)]
        elif key in (10, 13, 32) and self._focus in enabled:
            self._activate(self._focus)
        elif key == ord("d"):
            self._activate(ACTION_DETAILS)

    def _activate(self, action: str) -> None:
        if action == ACTION_DETAILS:
            self._details = not self._details
        else:
            self._pending_action = action

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        self._pointer = (x, y)
        hit = next(
            (b for b in self._buttons if b.enabled and b.rect.contains(x, y)), None
        )
        if event == self._cv2.EVENT_LBUTTONDOWN:
            self._pressed = hit.action if hit else None
            self._focus = self._pressed
        elif event == self._cv2.EVENT_LBUTTONUP:
            if hit and hit.action == self._pressed:
                self._activate(hit.action)
            self._pressed = None

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
        robot_state_available: bool = True,
        can_calibrate: bool | None = None,
        control_label: str | None = None,
        commissioning_joint: str | None = None,
        alert: str | None = None,
    ) -> Any:
        camera_height, camera_width = camera_frame.shape[:2]
        width, height = self._canvas_size or (
            max(1280, camera_width),
            max(720, camera_height),
        )
        layout = dashboard_layout(width, height)
        canvas = self._np.full((height, width, 3), BACKGROUND, dtype=self._np.uint8)
        p = Painter(self._cv2, self._np, canvas, layout.scale)
        self._buttons = ()
        self._header(p, layout, robot_label)
        x, top = layout.margin, layout.header + p.px(22)
        p.text("Motion workspace", x, top, size=28, strong=True)
        subtitle = (
            "Live movement, one workspace."
            if person_detected
            else "Step into view with your shoulders and hands visible."
        )
        p.text(
            alert or subtitle,
            x,
            top + p.px(36),
            color=RED if alert else MUTED,
            size=14,
            width=width - 2 * x,
        )
        if recording:
            p.dot(width - x - p.px(125), top + p.px(12), RED)
            p.text(
                "Recording session",
                width - x,
                top + p.px(6),
                color=RED,
                align="right",
                size=13,
            )
        if self._workspace_focus:
            self._preview(
                p, layout.camera, simulation_frame, robot_label, robot_state_available
            )
            self._camera(
                p, layout.preview, camera_frame, source_label, fps, person_detected
            )
        else:
            self._camera(
                p, layout.camera, camera_frame, source_label, fps, person_detected
            )
            self._preview(
                p, layout.preview, simulation_frame, robot_label, robot_state_available
            )
        self._status(
            p,
            layout.status,
            person_detected,
            calibrated,
            gestures,
            tracking_quality,
            status_lines,
            recording,
        )
        calibration_ready = (
            person_detected
            if can_calibrate is None
            else person_detected and can_calibrate
        )
        self._footer(
            p,
            layout,
            mode,
            recording,
            calibration_ready,
            calibrated,
            control_label,
            commissioning_joint,
        )
        return canvas

    def _header(self, p: Painter, layout: DashboardLayout, robot: str) -> None:
        x, y = layout.margin, p.px(20)
        p.box(Rect(x, y - p.px(3), p.px(30), p.px(30)), ACCENT, radius=8, border=None)
        p.line((x + p.px(7), y + p.px(17)), (x + p.px(13), y + p.px(6)), ACCENT_INK, 2)
        p.line((x + p.px(13), y + p.px(6)), (x + p.px(22), y + p.px(14)), ACCENT_INK, 2)
        p.dot(x + p.px(22), y + p.px(14), ACCENT_INK, 2)
        p.text("Motion Twin", x + p.px(42), y + p.px(3), size=20, strong=True)
        p.line((x + p.px(180), y), (x + p.px(180), y + p.px(22)))
        p.text("Dual UR7e workspace", x + p.px(198), y + p.px(6), size=13, color=MUTED)
        label = backend_label(robot)
        badge_width = p.measure(label, 13) + p.px(38)
        right = p.canvas.shape[1] - layout.margin
        badge = Rect(right - badge_width, y - p.px(1), badge_width, p.px(27))
        p.box(badge, SURFACE_RAISED, radius=7)
        p.dot(
            badge.x + p.px(13),
            badge.y + badge.height // 2,
            ACCENT if robot == "sim" else MUTED,
            3,
        )
        p.text(label, badge.x + p.px(25), badge.y + p.px(7), size=13)
        p.line((layout.margin, layout.header), (right, layout.header))

    def _camera(
        self,
        p: Painter,
        rect: Rect,
        camera: Any,
        source: str,
        fps: float,
        detected: bool,
    ) -> None:
        p.box(rect)
        pad = p.px(20)
        p.text("Camera", rect.x + pad, rect.y + p.px(19), size=16, strong=True)
        state = "Pose detected" if detected else "Waiting for pose"
        label_width = p.measure(state, 12)
        p.dot(
            rect.right - pad - label_width - p.px(10),
            rect.y + p.px(27),
            GREEN if detected else MUTED,
        )
        p.text(
            state,
            rect.right - pad,
            rect.y + p.px(21),
            size=12,
            color=GREEN if detected else MUTED,
            align="right",
        )
        target = Rect(
            rect.x + 1, rect.y + p.px(55), rect.width - 2, rect.height - p.px(94)
        )
        self._cv2.rectangle(
            p.canvas,
            (target.x, target.y),
            (target.right - 1, target.bottom - 1),
            (16, 15, 14),
            -1,
        )
        self._place_image(p.canvas, camera, target)
        if not detected:
            label = "Waiting for a person"
            w = p.measure(label, 13) + p.px(28)
            notice = Rect(
                target.x + (target.width - w) // 2, target.y + p.px(16), w, p.px(32)
            )
            p.box(notice, SURFACE, radius=8)
            p.text(label, notice.x + p.px(14), notice.y + p.px(9), size=13, color=MUTED)
        bottom = rect.bottom - p.px(25)
        h, w = camera.shape[:2]
        metrics = f"{w} × {h}   ·   {max(0, fps):.0f} fps"
        p.text(
            source,
            rect.x + pad,
            bottom,
            size=12,
            color=MUTED,
            width=max(0, rect.width - 2 * pad - p.measure(metrics, 12) - p.px(16)),
        )
        p.text(metrics, rect.right - pad, bottom, size=12, color=MUTED, align="right")

    def _preview(
        self, p: Painter, rect: Rect, simulation: Any, robot: str, available: bool
    ) -> None:
        p.box(rect)
        pad = p.px(20)
        p.text("3D workspace", rect.x + pad, rect.y + p.px(19), size=16, strong=True)
        p.text(
            "2 × UR7e · BODY XYZ",
            rect.right - pad,
            rect.y + p.px(22),
            size=12,
            color=MUTED,
            align="right",
        )
        target = preview_target_rect(rect, p.px)
        if available:
            self._place_image(p.canvas, simulation, target)
        else:
            p.text(
                "No arm data",
                rect.x + rect.width // 2,
                target.y + target.height // 2 - p.px(6),
                color=DISABLED,
                size=13,
                align="center",
            )
        y = rect.bottom - p.px(25)
        label = "Simulated" if robot == "sim" else "Commanded"
        p.line(
            (rect.x + pad, y + p.px(5)),
            (rect.x + pad + p.px(14), y + p.px(5)),
            ACCENT,
            2,
        )
        p.text(label, rect.x + pad + p.px(22), y, size=11, color=MUTED)
        x = rect.x + pad + p.px(22) + p.measure(label, 11) + p.px(20)
        p.line((x, y + p.px(5)), (x + p.px(14), y + p.px(5)), DISABLED, 2)
        p.text("Target", x + p.px(22), y, size=11, color=MUTED)

    def _status(
        self,
        p: Painter,
        rect: Rect,
        detected: bool,
        calibrated: bool,
        gestures: tuple,
        quality: float,
        lines: tuple,
        recording: bool,
    ) -> None:
        p.box(rect)
        pad = p.px(20)
        x, right, y = rect.x + pad, rect.right - pad, rect.y + p.px(20)
        p.text("Session", x, y, size=16, strong=True)
        button = DashboardButton(
            ACTION_DETAILS,
            "Back" if self._details else "Details",
            "D",
            Rect(right - p.px(77), y - p.px(7), p.px(82), p.px(28)),
            active=self._details,
        )
        self._buttons += (button,)
        self._button(p, button, compact=True)
        y += p.px(43)
        if self._details:
            p.text("Backend diagnostics", x, y, size=12, color=MUTED)
            y += p.px(26)
            for message in lines or ("No backend messages.",):
                for line in p.wrap(message, right - x):
                    if y + p.px(14) > rect.bottom - pad:
                        return
                    p.text(line, x, y, size=12, color=TEXT, width=right - x)
                    y += p.px(21)
                y += p.px(8)
            return
        p.text("Landmarks visible", x, y, size=13, color=MUTED)
        value = f"{max(0, min(1, quality)):.0%}" if detected else "—"
        p.text(
            value, right, y, size=13, align="right", color=TEXT if detected else MUTED
        )
        y += p.px(25)
        bar = Rect(x, y, right - x, p.px(4))
        p.box(bar, SURFACE_RAISED, radius=2, border=None)
        if detected and quality > 0:
            p.box(
                Rect(x, y, max(1, round(bar.width * min(1, quality))), bar.height),
                GREEN,
                radius=2,
                border=None,
            )
        y += p.px(25)
        p.text("Calibration", x, y, size=13, color=MUTED)
        p.text(
            "Calibrated" if calibrated else "Not set",
            right,
            y,
            size=13,
            color=GREEN if calibrated else MUTED,
            align="right",
        )
        y += p.px(30)
        p.text("Recording", x, y, size=13, color=MUTED)
        p.text(
            "In progress" if recording else "Idle",
            right,
            y,
            size=13,
            color=RED if recording else MUTED,
            align="right",
        )
        y += p.px(32)
        if y + p.px(40) <= rect.bottom - pad:
            p.line((x, y), (right, y))
            y += p.px(19)
            p.text("Gestures", x, y, size=12, color=MUTED)
            y += p.px(25)
            labels = tuple(g.replace("_", " ").capitalize() for g in gestures) or (
                "No gesture detected",
            )
            for label in labels:
                if y + p.px(14) > rect.bottom - pad:
                    break
                p.text(
                    label,
                    x,
                    y,
                    size=13,
                    color=TEXT if gestures else DISABLED,
                    width=right - x,
                )
                y += p.px(23)

    def _footer(
        self,
        p: Painter,
        layout: DashboardLayout,
        mode: str,
        recording: bool,
        detected: bool,
        calibrated: bool,
        control_label: str | None = None,
        commissioning_joint: str | None = None,
    ) -> None:
        rect, pad = layout.footer, layout.margin
        p.line((pad, rect.y), (rect.right - pad, rect.y))
        hint = (
            "Ready to calibrate your neutral pose."
            if detected and not calibrated
            else "Session controls"
        )
        if not detected:
            hint = "Show your shoulders and arms to enable calibration."
        nav_hint = "Tab to navigate · Enter to select"
        hint_width = rect.width - 2 * pad - p.measure(nav_hint, 11) - p.px(24)
        p.text(hint, pad, rect.y + p.px(14), size=12, color=MUTED, width=hint_width)
        p.text(
            nav_hint,
            rect.right - pad,
            rect.y + p.px(14),
            size=11,
            color=DISABLED,
            align="right",
        )
        top, height, gap = rect.y + p.px(39), p.px(43), p.px(10)
        if commissioning_joint:
            label = commissioning_joint.replace("_", " ").title()
            hint = f"Hold a jog button to move {label}; releasing it stops motion."
            p.text(
                hint,
                pad,
                rect.y + p.px(14),
                size=12,
                color=MUTED,
                width=hint_width,
            )
            definitions = (
                (ACTION_JOG_NEGATIVE, f"{label}  −", "[", 180, False, True, False),
                (ACTION_JOG_POSITIVE, f"{label}  +", "]", 180, False, True, False),
                (ACTION_STOP, "Stop motion", "P", 165, False, True, False),
                (
                    ACTION_CONTROL,
                    control_label or "Disarm",
                    "H",
                    190,
                    False,
                    True,
                    True,
                ),
                (ACTION_QUIT, "Quit", "Esc", 90, False, True, False),
            )
        else:
            definitions = (
            (
                ACTION_CALIBRATE,
                "Recalibrate" if calibrated else "Calibrate",
                "C",
                162,
                False,
                detected,
                True,
            ),
            (
                ACTION_RECORD,
                "Stop recording" if recording else "Record session",
                "R",
                180,
                recording,
                True,
                False,
            ),
            (
                ACTION_CONTROL if control_label else ACTION_MODE,
                control_label or f"Console: {mode}",
                "H" if control_label else "1–3",
                178,
                False,
                True,
                bool(control_label),
            ),
            (
                ACTION_FULLSCREEN,
                "Exit full screen" if self._fullscreen else "Full screen",
                "F",
                160,
                self._fullscreen,
                True,
                False,
            ),
            (
                ACTION_VIEW,
                "Camera focus" if self._workspace_focus else "3D focus",
                "V",
                145,
                self._workspace_focus,
                True,
                False,
            ),
            (ACTION_QUIT, "Quit", "Esc", 90, False, True, False),
            )
        widths = [p.px(d[3]) for d in definitions]
        x = pad
        for i, (action, label, shortcut, _, active, enabled, primary) in enumerate(
            definitions
        ):
            if not commissioning_joint and i == 2:
                x = max(
                    x,
                    rect.right
                    - pad
                    - sum(widths[2:])
                    - gap * max(0, len(widths) - 3),
                )
            button = DashboardButton(
                action,
                label,
                shortcut,
                Rect(x, top, widths[i], height),
                active=active,
                enabled=enabled,
                primary=primary,
            )
            self._buttons += (button,)
            self._button(p, button)
            x += widths[i] + gap

    def _button(
        self, p: Painter, button: DashboardButton, compact: bool = False
    ) -> None:
        rect = button.rect
        hover = button.enabled and rect.contains(*self._pointer)
        pressed = hover and self._pressed == button.action
        fill, ink, border = SURFACE, TEXT, BORDER
        if button.action in (
            ACTION_MODE,
            ACTION_FULLSCREEN,
            ACTION_QUIT,
            ACTION_DETAILS,
            ACTION_VIEW,
        ):
            fill, border = BACKGROUND if not compact else SURFACE, None
        if button.action == ACTION_STOP:
            fill, ink, border = RED, (255, 255, 255), None
        if hover:
            fill = HOVER
        if button.primary and button.enabled:
            fill, ink, border = ACCENT, ACCENT_INK, None
            if hover:
                fill = (130, 186, 251)
        if button.active:
            fill, ink = (
                SURFACE_RAISED,
                RED if button.action == ACTION_RECORD else ACCENT,
            )
        if not button.enabled:
            ink, fill = DISABLED, SURFACE
        if pressed or (self._focus == button.action and button.enabled):
            border = ACCENT
        p.box(rect, fill, radius=8, border=border)
        size = 12 if compact else 14
        padding = p.px(10 if compact else 14)
        shortcut_width = p.measure(button.shortcut, 10) + p.px(10)
        show_shortcut = (
            not compact
            and rect.width
            > p.measure(button.label, size) + shortcut_width + 3 * padding
        )
        room = (
            rect.width
            - 2 * padding
            - (shortcut_width + p.px(6) if show_shortcut else 0)
        )
        p.text(
            button.label,
            rect.x + padding,
            rect.y + (rect.height - p.px(size)) // 2,
            size=size,
            color=ink,
            strong=button.primary,
            width=room,
        )
        if show_shortcut:
            p.text(
                button.shortcut,
                rect.right - padding,
                rect.y + (rect.height - p.px(10)) // 2,
                size=10,
                color=ink if button.primary else MUTED,
                align="right",
            )

    def simulation_target_size(self) -> tuple[int, int]:
        """Pixel size of the arm preview area, so the simulation renders without rescaling."""
        width, height = self._canvas_size or (1280, 720)
        layout = dashboard_layout(width, height)
        rect = layout.camera if self._workspace_focus else layout.preview
        target = preview_target_rect(
            rect, lambda value: max(1, round(value * layout.scale))
        )
        return max(2, target.width), max(2, target.height)

    def _place_image(self, canvas: Any, image: Any, target: Rect) -> None:
        source_height, source_width = image.shape[:2]
        if (source_width, source_height) == (target.width, target.height):
            canvas[target.y : target.bottom, target.x : target.right] = image
            return
        fitted = fit_inside((source_width, source_height), target)
        if fitted.width <= 0 or fitted.height <= 0:
            return
        interpolation = (
            self._cv2.INTER_AREA
            if fitted.width <= source_width
            else self._cv2.INTER_LINEAR
        )
        resized = self._cv2.resize(
            image, (fitted.width, fitted.height), interpolation=interpolation
        )
        canvas[fitted.y : fitted.bottom, fitted.x : fitted.right] = resized
