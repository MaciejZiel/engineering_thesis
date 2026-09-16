import math
from typing import Any

from vision_robot_arm.core.hud import fill_translucent, put_text
from vision_robot_arm.robot.targets import GRIPPER_CLOSE, JOINT_ELBOW, JOINT_SHOULDER, ArmState

Point = tuple[int, int]
Color = tuple[int, int, int]

PANEL_BORDER: Color = (90, 90, 90)
CURRENT_ARM: Color = (80, 220, 80)
TARGET_ARM: Color = (120, 120, 120)
GRIPPER_COLOR: Color = (60, 200, 255)
TITLE_COLOR: Color = (235, 235, 235)
VALUE_COLOR: Color = (200, 200, 200)
TEXT_SCALE = 0.45
LINE_HEIGHT = 18
PADDING = 10


def arm_points(
    shoulder_deg: float,
    elbow_deg: float,
    base: Point,
    link_length: float,
) -> tuple[Point, Point, Point]:
    upper_direction = math.radians(shoulder_deg - 90.0)
    elbow = (
        base[0] + link_length * math.cos(upper_direction),
        base[1] - link_length * math.sin(upper_direction),
    )
    forearm_direction = upper_direction + math.radians(180.0 - elbow_deg)
    wrist = (
        elbow[0] + link_length * math.cos(forearm_direction),
        elbow[1] - link_length * math.sin(forearm_direction),
    )
    return base, (round(elbow[0]), round(elbow[1])), (round(wrist[0]), round(wrist[1]))


def draw_arm_panel(
    cv2: Any,
    frame: Any,
    state: ArmState,
    origin: Point,
    size: int = 240,
) -> None:
    left, top = origin
    right, bottom = left + size, top + size
    factor = size / 240.0
    padding = round(PADDING * factor)
    line_height = round(LINE_HEIGHT * factor)
    text_scale = TEXT_SCALE * factor
    fill_translucent(cv2, frame, (left, top), (right, bottom))
    cv2.rectangle(frame, (left, top), (right, bottom), PANEL_BORDER, 1, cv2.LINE_AA)

    lines = [
        "robot arm",
        _joint_line(JOINT_SHOULDER, state),
        _joint_line(JOINT_ELBOW, state),
        f"gripper {state.gripper}   lift {'on' if state.lift_mode else 'off'}",
    ]
    for row, text in enumerate(lines):
        put_text(
            cv2,
            frame,
            text,
            (left + padding, top + padding + (row + 1) * line_height - round(4 * factor)),
            TITLE_COLOR if row == 0 else VALUE_COLOR,
            text_scale,
            2 if row == 0 or factor >= 1.4 else 1,
        )

    text_bottom = top + padding + len(lines) * line_height
    sketch_height = bottom - text_bottom
    base = (left + size // 2, text_bottom + sketch_height // 2)
    link_length = min(size, sketch_height) * 0.24

    _draw_arm(
        cv2,
        frame,
        state.targets.get(JOINT_SHOULDER, 90.0),
        state.targets.get(JOINT_ELBOW, 180.0),
        base,
        link_length,
        TARGET_ARM,
        thickness=max(1, round(2 * factor)),
    )
    _, elbow, wrist = _draw_arm(
        cv2,
        frame,
        state.joints.get(JOINT_SHOULDER, 90.0),
        state.joints.get(JOINT_ELBOW, 180.0),
        base,
        link_length,
        CURRENT_ARM,
        thickness=max(2, round(4 * factor)),
    )
    _draw_gripper(cv2, frame, elbow, wrist, state.gripper == GRIPPER_CLOSE, int(size * 0.06))


def _draw_arm(
    cv2: Any,
    frame: Any,
    shoulder_deg: float,
    elbow_deg: float,
    base: Point,
    link_length: float,
    color: Color,
    thickness: int,
) -> tuple[Point, Point, Point]:
    points = arm_points(shoulder_deg, elbow_deg, base, link_length)
    base_point, elbow, wrist = points
    cv2.line(frame, base_point, elbow, color, thickness, cv2.LINE_AA)
    cv2.line(frame, elbow, wrist, color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, thickness + 2, color, -1, cv2.LINE_AA)
    return points


def _draw_gripper(
    cv2: Any,
    frame: Any,
    elbow: Point,
    wrist: Point,
    closed: bool,
    jaw_length: int,
) -> None:
    dx, dy = wrist[0] - elbow[0], wrist[1] - elbow[1]
    length = math.hypot(dx, dy) or 1.0
    direction = (dx / length, dy / length)
    normal = (-direction[1], direction[0])
    spread = 2 if closed else max(4, jaw_length // 2)
    for side in (-1, 1):
        start = (
            round(wrist[0] + side * spread * normal[0]),
            round(wrist[1] + side * spread * normal[1]),
        )
        end = (
            round(start[0] + jaw_length * direction[0]),
            round(start[1] + jaw_length * direction[1]),
        )
        cv2.line(frame, start, end, GRIPPER_COLOR, 2, cv2.LINE_AA)


def _joint_line(joint: str, state: ArmState) -> str:
    current = state.joints.get(joint)
    target = state.targets.get(joint)
    if current is None or target is None:
        return f"{joint} n/a"
    return f"{joint} {current:5.1f} -> {target:5.1f}"
