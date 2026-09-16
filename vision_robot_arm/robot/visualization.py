import math
from typing import Any

from vision_robot_arm.robot.targets import GRIPPER_CLOSE, JOINT_ELBOW, JOINT_SHOULDER, ArmState

Point = tuple[int, int]
Color = tuple[int, int, int]

PANEL_BACKGROUND: Color = (30, 30, 30)
PANEL_BORDER: Color = (120, 120, 120)
CURRENT_ARM: Color = (80, 220, 80)
TARGET_ARM: Color = (110, 110, 110)
GRIPPER_COLOR: Color = (60, 200, 255)
TEXT_COLOR: Color = (240, 240, 240)


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
    cv2.rectangle(frame, (left, top), (right, bottom), PANEL_BACKGROUND, -1)
    cv2.rectangle(frame, (left, top), (right, bottom), PANEL_BORDER, 1, cv2.LINE_AA)

    base = (left + int(size * 0.5), top + int(size * 0.5))
    link_length = size * 0.2

    _draw_arm(
        cv2,
        frame,
        state.targets.get(JOINT_SHOULDER, 90.0),
        state.targets.get(JOINT_ELBOW, 180.0),
        base,
        link_length,
        TARGET_ARM,
        thickness=2,
    )
    _, elbow, wrist = _draw_arm(
        cv2,
        frame,
        state.joints.get(JOINT_SHOULDER, 90.0),
        state.joints.get(JOINT_ELBOW, 180.0),
        base,
        link_length,
        CURRENT_ARM,
        thickness=4,
    )
    _draw_gripper(cv2, frame, elbow, wrist, state.gripper == GRIPPER_CLOSE, int(size * 0.06))

    lines = [
        "robot arm",
        _joint_line(JOINT_SHOULDER, state),
        _joint_line(JOINT_ELBOW, state),
        f"gripper {state.gripper} | lift {'on' if state.lift_mode else 'off'}",
    ]
    for row, text in enumerate(lines):
        _draw_text(cv2, frame, text, (left + 8, top + 18 + row * 18), row == 0)


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


def _draw_text(cv2: Any, frame: Any, text: str, position: Point, bold: bool) -> None:
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        2 if bold else 1,
        cv2.LINE_AA,
    )
