import math
from typing import Any

from vision_robot_arm.core.hud import fill_translucent, put_text
from vision_robot_arm.robot.targets import (
    ARM_LEFT,
    ARM_RIGHT,
    GRIPPER_CLOSE,
    JOINT_ELBOW,
    JOINT_NAMES,
    JOINT_SHOULDER,
    JOINT_WRIST,
    ArmState,
    RobotState,
)

Point = tuple[int, int]
Color = tuple[int, int, int]

BACKGROUND: Color = (24, 24, 24)
PANEL_BORDER: Color = (90, 90, 90)
CURRENT_ARM: Color = (80, 220, 80)
TARGET_ARM: Color = (120, 120, 120)
GRIPPER_COLOR: Color = (255, 200, 60)
TITLE_COLOR: Color = (235, 235, 235)
VALUE_COLOR: Color = (200, 200, 200)
MUTED_COLOR: Color = (130, 130, 130)
TEXT_SCALE = 0.45
LINE_HEIGHT = 18
PADDING = 10
HAND_LINK_RATIO = 0.55
DISPLAY_ORDER = (ARM_LEFT, ARM_RIGHT)


def arm_points(
    shoulder_deg: float,
    elbow_deg: float,
    wrist_deg: float,
    base: Point,
    link_length: float,
    mirror: bool = False,
) -> tuple[Point, Point, Point, Point]:
    sign = -1.0 if mirror else 1.0
    upper_direction = math.radians(shoulder_deg - 90.0)
    forearm_direction = upper_direction + math.radians(180.0 - elbow_deg)
    hand_direction = forearm_direction + math.radians(180.0 - wrist_deg)

    def advance(start: tuple[float, float], direction: float, length: float) -> tuple[float, float]:
        return (
            start[0] + sign * length * math.cos(direction),
            start[1] - length * math.sin(direction),
        )

    elbow = advance(base, upper_direction, link_length)
    wrist = advance(elbow, forearm_direction, link_length)
    tip = advance(wrist, hand_direction, link_length * HAND_LINK_RATIO)
    return base, _rounded(elbow), _rounded(wrist), _rounded(tip)


def draw_simulation(cv2: Any, canvas: Any, state: RobotState | None) -> None:
    canvas[:] = BACKGROUND
    height, width = canvas.shape[:2]
    margin = max(8, round(width * 0.02))
    footer = round(LINE_HEIGHT * 1.6 * _scale_for(min(width, height)))
    panel_width = (width - 3 * margin) // 2
    panel_height = height - 2 * margin - footer

    for column, arm in enumerate(DISPLAY_ORDER):
        left = margin + column * (panel_width + margin)
        draw_arm_panel(
            cv2,
            canvas,
            state.arm(arm) if state is not None else None,
            (left, margin),
            (panel_width, panel_height),
            title=f"{arm} arm",
            mirror=arm == ARM_LEFT,
        )

    lift = "on" if state is not None and state.lift_mode else "off"
    footer_scale = TEXT_SCALE * _scale_for(min(width, height))
    put_text(
        cv2,
        canvas,
        f"lift mode: {lift}    grey = target, green = current, blue = gripper",
        (margin, height - margin - round(6 * footer_scale / TEXT_SCALE)),
        MUTED_COLOR,
        footer_scale,
    )


def draw_arm_panel(
    cv2: Any,
    frame: Any,
    state: ArmState | None,
    origin: Point,
    size: tuple[int, int],
    title: str = "robot arm",
    mirror: bool = False,
) -> None:
    left, top = origin
    width, height = size
    right, bottom = left + width, top + height
    factor = _scale_for(min(width, height))
    padding = round(PADDING * factor)
    line_height = round(LINE_HEIGHT * factor)
    text_scale = TEXT_SCALE * factor
    thick_text = 2 if factor >= 1.4 else 1

    fill_translucent(cv2, frame, (left, top), (right, bottom))
    cv2.rectangle(frame, (left, top), (right, bottom), PANEL_BORDER, 1, cv2.LINE_AA)

    lines = [title] + [_joint_line(joint, state) for joint in JOINT_NAMES]
    lines.append(f"gripper {state.gripper if state is not None else 'n/a'}")
    for row, text in enumerate(lines):
        put_text(
            cv2,
            frame,
            text,
            (left + padding, top + padding + (row + 1) * line_height - round(4 * factor)),
            TITLE_COLOR if row == 0 else VALUE_COLOR,
            text_scale,
            2 if row == 0 else thick_text,
        )

    text_bottom = top + padding + len(lines) * line_height
    sketch_height = bottom - text_bottom
    if sketch_height <= 0:
        return
    base = (left + width // 2, text_bottom + round(sketch_height * 0.5))
    link_length = min(width, sketch_height) * 0.3

    targets = state.targets if state is not None else {}
    joints = state.joints if state is not None else {}
    _draw_arm(cv2, frame, targets, base, link_length, TARGET_ARM, max(1, round(2 * factor)), mirror)
    _, _, wrist, tip = _draw_arm(
        cv2, frame, joints, base, link_length, CURRENT_ARM, max(2, round(4 * factor)), mirror
    )
    closed = state is not None and state.gripper == GRIPPER_CLOSE
    _draw_gripper(cv2, frame, wrist, tip, closed, max(4, round(link_length * 0.3)))


def _draw_arm(
    cv2: Any,
    frame: Any,
    angles: dict[str, float],
    base: Point,
    link_length: float,
    color: Color,
    thickness: int,
    mirror: bool,
) -> tuple[Point, Point, Point, Point]:
    points = arm_points(
        angles.get(JOINT_SHOULDER, 90.0),
        angles.get(JOINT_ELBOW, 180.0),
        angles.get(JOINT_WRIST, 180.0),
        base,
        link_length,
        mirror,
    )
    for start, end in zip(points, points[1:]):
        cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
    for point in points[:3]:
        cv2.circle(frame, point, thickness + 2, color, -1, cv2.LINE_AA)
    return points


def _draw_gripper(
    cv2: Any,
    frame: Any,
    wrist: Point,
    tip: Point,
    closed: bool,
    jaw_length: int,
) -> None:
    dx, dy = tip[0] - wrist[0], tip[1] - wrist[1]
    length = math.hypot(dx, dy) or 1.0
    direction = (dx / length, dy / length)
    normal = (-direction[1], direction[0])
    spread = 2 if closed else max(4, jaw_length // 2)
    for side in (-1, 1):
        start = (
            round(tip[0] + side * spread * normal[0]),
            round(tip[1] + side * spread * normal[1]),
        )
        end = (
            round(start[0] + jaw_length * direction[0]),
            round(start[1] + jaw_length * direction[1]),
        )
        cv2.line(frame, start, end, GRIPPER_COLOR, 2, cv2.LINE_AA)


def _joint_line(joint: str, state: ArmState | None) -> str:
    if state is None:
        return f"{joint} n/a"
    current = state.joints.get(joint)
    target = state.targets.get(joint)
    if current is None or target is None:
        return f"{joint} n/a"
    return f"{joint} {current:5.1f} -> {target:5.1f}"


def _scale_for(extent: int) -> float:
    return max(0.6, extent / 240.0)


def _rounded(point: tuple[float, float]) -> Point:
    return round(point[0]), round(point[1])
