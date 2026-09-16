import math
from typing import Any

from vision_robot_arm.core.hud import put_text, text_size
from vision_robot_arm.robot.targets import (
    ARM_LEFT,
    ARM_RIGHT,
    GRIPPER_CLOSE,
    JOINT_ELBOW,
    JOINT_SHOULDER,
    JOINT_WRIST_1,
    MAPPED_JOINTS,
    ROBOT_MODEL,
    UR_HOME_DEG,
    ArmState,
    RobotState,
)

Point = tuple[int, int]
FloatPoint = tuple[float, float]
Color = tuple[int, int, int]

BACKGROUND: Color = (23, 29, 37)
PANEL: Color = (29, 37, 47)
PANEL_BORDER: Color = (55, 66, 78)
GRID: Color = (38, 47, 58)
PEDESTAL: Color = (70, 82, 96)
LINK_FILL: Color = (0, 142, 255)
LINK_EDGE: Color = (0, 90, 170)
JOINT_FILL: Color = (240, 244, 248)
JOINT_RING: Color = (0, 90, 170)
TARGET_ARM: Color = (120, 132, 146)
GRIPPER_OPEN_COLOR: Color = (80, 210, 130)
GRIPPER_CLOSED_COLOR: Color = (80, 90, 235)
TEXT: Color = (238, 242, 247)
MUTED: Color = (148, 160, 174)
ACCENT: Color = (30, 178, 255)

HAND_LINK_RATIO = 0.6
DISPLAY_ORDER = (ARM_LEFT, ARM_RIGHT)
JOINT_LABELS = {JOINT_SHOULDER: "S", JOINT_ELBOW: "E", JOINT_WRIST_1: "W1"}
REFERENCE_PANEL_WIDTH = 300.0


def arm_points(
    shoulder_deg: float,
    elbow_deg: float,
    wrist_deg: float,
    base: Point,
    link_length: float,
    mirror: bool = False,
) -> tuple[Point, Point, Point, Point]:
    sign = -1.0 if mirror else 1.0
    upper_direction = math.radians(shoulder_deg + 90.0)
    forearm_direction = upper_direction + math.radians(elbow_deg)
    hand_direction = forearm_direction + math.radians(wrist_deg)

    def advance(start: FloatPoint, direction: float, length: float) -> FloatPoint:
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
    factor = _scale_for(width / 2)
    margin = max(6, round(8 * factor))
    footer = round(22 * factor)
    panel_width = (width - 3 * margin) // 2
    panel_height = height - 2 * margin - footer
    if panel_width <= 0 or panel_height <= 0:
        return

    for column, arm in enumerate(DISPLAY_ORDER):
        left = margin + column * (panel_width + margin)
        draw_arm_panel(
            cv2,
            canvas,
            state.arm(arm) if state is not None else None,
            (left, margin),
            (panel_width, panel_height),
            title=f"{arm.upper()} {ROBOT_MODEL}",
            mirror=arm == ARM_LEFT,
        )

    lift = "ON" if state is not None and state.lift_mode else "OFF"
    footer_scale = 0.36 * factor
    baseline = height - margin - round(6 * factor)
    put_text(cv2, canvas, f"LIFT {lift}", (margin, baseline), ACCENT if lift == "ON" else MUTED, footer_scale)
    legend = "dashed = target   solid = current   base / wrist 2 / wrist 3 held"
    legend_width = text_size(cv2, legend, footer_scale)[0]
    put_text(cv2, canvas, legend, (max(margin, width - margin - legend_width), baseline), MUTED, footer_scale)


def draw_arm_panel(
    cv2: Any,
    frame: Any,
    state: ArmState | None,
    origin: Point,
    size: tuple[int, int],
    title: str = ROBOT_MODEL,
    mirror: bool = False,
) -> None:
    left, top = origin
    width, height = size
    right, bottom = left + width, top + height
    factor = _scale_for(width)
    pad = round(10 * factor)
    title_height = round(28 * factor)
    readout_rows = len(MAPPED_JOINTS) + 1
    row_height = round(17 * factor)
    readout_height = readout_rows * row_height + pad

    cv2.rectangle(frame, (left, top), (right, bottom), PANEL, -1)
    cv2.rectangle(frame, (left, top), (right, bottom), PANEL_BORDER, 1, cv2.LINE_AA)

    put_text(cv2, frame, title, (left + pad, top + round(19 * factor)), TEXT, 0.46 * factor, 2)
    _draw_gripper_badge(cv2, frame, state, (right - pad, top + round(19 * factor)), factor)

    sketch_top = top + title_height
    sketch_bottom = bottom - readout_height
    sketch_height = sketch_bottom - sketch_top
    if sketch_height > 20:
        _draw_sketch(cv2, frame, state, (left, sketch_top), (width, sketch_height), factor, mirror)

    _draw_readout(cv2, frame, state, (left + pad, sketch_bottom + pad), width - 2 * pad, row_height, factor)


def _draw_sketch(
    cv2: Any,
    frame: Any,
    state: ArmState | None,
    origin: Point,
    size: tuple[int, int],
    factor: float,
    mirror: bool,
) -> None:
    left, top = origin
    width, height = size
    bottom = top + height
    grid_step = max(12, round(24 * factor))
    for x in range(left + grid_step, left + width, grid_step):
        cv2.line(frame, (x, top), (x, bottom), GRID, 1)
    for y in range(top + grid_step, bottom, grid_step):
        cv2.line(frame, (left, y), (left + width, y), GRID, 1)

    link_length = min(width * 0.32, height * 0.24)
    base_x = left + round(width * (0.78 if mirror else 0.22))
    base = (base_x, top + round(height * 0.52))

    pedestal_width = round(link_length * 0.5)
    pedestal_height = round(link_length * 0.25)
    cv2.rectangle(
        frame,
        (base[0] - pedestal_width // 2, base[1] + pedestal_height // 2),
        (base[0] + pedestal_width // 2, base[1] + pedestal_height),
        PEDESTAL,
        -1,
    )

    targets = state.targets if state is not None else {}
    joints = state.joints if state is not None else {}
    thickness = max(4, round(9 * factor))

    target_points = _points_for(targets, base, link_length, mirror)
    for start, end in zip(target_points, target_points[1:]):
        _dashed_line(cv2, frame, start, end, TARGET_ARM, max(1, round(2 * factor)), round(6 * factor))
    for point in target_points[1:3]:
        cv2.circle(frame, point, max(2, round(3 * factor)), TARGET_ARM, -1, cv2.LINE_AA)

    points = _points_for(joints, base, link_length, mirror)
    for start, end in zip(points, points[:3][1:]):
        cv2.line(frame, start, end, LINK_EDGE, thickness + 4, cv2.LINE_AA)
    for start, end in zip(points, points[:3][1:]):
        cv2.line(frame, start, end, LINK_FILL, thickness, cv2.LINE_AA)

    joint_radius = max(4, round(7 * factor))
    for point, joint in zip(points[:3], MAPPED_JOINTS):
        cv2.circle(frame, point, joint_radius + 2, JOINT_RING, -1, cv2.LINE_AA)
        cv2.circle(frame, point, joint_radius, JOINT_FILL, -1, cv2.LINE_AA)
        label = JOINT_LABELS[joint]
        label_width = text_size(cv2, label, 0.34 * factor)[0]
        label_x = point[0] - joint_radius - label_width - 4 if mirror else point[0] + joint_radius + 4
        put_text(cv2, frame, label, (label_x, point[1] - joint_radius - 2), MUTED, 0.34 * factor)

    closed = state is not None and state.gripper == GRIPPER_CLOSE
    _draw_gripper(cv2, frame, points[2], points[3], closed, link_length * HAND_LINK_RATIO, factor)


def _points_for(
    angles: dict[str, float],
    base: Point,
    link_length: float,
    mirror: bool,
) -> tuple[Point, Point, Point, Point]:
    return arm_points(
        angles.get(JOINT_SHOULDER, UR_HOME_DEG[JOINT_SHOULDER]),
        angles.get(JOINT_ELBOW, UR_HOME_DEG[JOINT_ELBOW]),
        angles.get(JOINT_WRIST_1, UR_HOME_DEG[JOINT_WRIST_1]),
        base,
        link_length,
        mirror,
    )


def _draw_gripper(
    cv2: Any,
    frame: Any,
    wrist: Point,
    tip: Point,
    closed: bool,
    hand_length: float,
    factor: float,
) -> None:
    dx, dy = tip[0] - wrist[0], tip[1] - wrist[1]
    length = math.hypot(dx, dy) or 1.0
    direction = (dx / length, dy / length)
    normal = (-direction[1], direction[0])
    color = GRIPPER_CLOSED_COLOR if closed else GRIPPER_OPEN_COLOR
    thickness = max(2, round(4 * factor))

    palm_start = _offset(wrist, direction, hand_length * 0.55)
    palm_half = hand_length * (0.18 if closed else 0.42)
    jaw_length = hand_length * 0.5
    cv2.line(frame, wrist, palm_start, LINK_EDGE, thickness + 2, cv2.LINE_AA)
    cv2.line(frame, wrist, palm_start, LINK_FILL, thickness - 1, cv2.LINE_AA)
    palm_a = _offset(palm_start, normal, palm_half)
    palm_b = _offset(palm_start, normal, -palm_half)
    cv2.line(frame, palm_a, palm_b, color, thickness, cv2.LINE_AA)
    for side_point in (palm_a, palm_b):
        jaw_end = _offset(side_point, direction, jaw_length)
        cv2.line(frame, side_point, jaw_end, color, thickness, cv2.LINE_AA)
        pad_length = jaw_length * 0.35
        pad_inward = palm_half * 0.35
        pad_dir = -1.0 if side_point == palm_a else 1.0
        pad_start = _offset(jaw_end, direction, -pad_length)
        cv2.line(
            frame,
            pad_start,
            _offset(pad_start, normal, pad_dir * pad_inward),
            color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )


def _draw_gripper_badge(cv2: Any, frame: Any, state: ArmState | None, anchor_right: Point, factor: float) -> None:
    if state is None:
        label, color = "NO DATA", MUTED
    elif state.gripper == GRIPPER_CLOSE:
        label, color = "GRIP CLOSED", GRIPPER_CLOSED_COLOR
    else:
        label, color = "GRIP OPEN", GRIPPER_OPEN_COLOR
    scale = 0.36 * factor
    width, height = text_size(cv2, label, scale)
    pad = round(6 * factor)
    right, baseline = anchor_right
    left, top = right - width - 2 * pad, baseline - height - pad // 2
    cv2.rectangle(frame, (left, top), (right, baseline + pad // 2), PANEL_BORDER, -1)
    cv2.circle(frame, (left + pad, baseline - height // 2 + 1), max(2, round(3 * factor)), color, -1, cv2.LINE_AA)
    put_text(cv2, frame, label, (left + 2 * pad + round(3 * factor), baseline), TEXT, scale)


def _draw_readout(
    cv2: Any,
    frame: Any,
    state: ArmState | None,
    origin: Point,
    width: int,
    row_height: int,
    factor: float,
) -> None:
    x, y = origin
    scale = 0.36 * factor
    current_x = x + round(width * 0.36)
    target_x = x + round(width * 0.68)
    baseline = y + row_height - round(4 * factor)
    put_text(cv2, frame, "JOINT", (x, baseline), MUTED, scale)
    put_text(cv2, frame, "NOW", (current_x, baseline), MUTED, scale)
    put_text(cv2, frame, "TARGET", (target_x, baseline), MUTED, scale)
    for row, joint in enumerate(MAPPED_JOINTS, start=1):
        baseline = y + (row + 1) * row_height - round(4 * factor)
        put_text(cv2, frame, joint.replace("_", " "), (x, baseline), TEXT, scale)
        current = state.joints.get(joint) if state is not None else None
        target = state.targets.get(joint) if state is not None else None
        put_text(cv2, frame, _format_angle(current), (current_x, baseline), TEXT, scale)
        moving = current is not None and target is not None and abs(current - target) > 0.5
        put_text(cv2, frame, _format_angle(target), (target_x, baseline), ACCENT if moving else MUTED, scale)


def _format_angle(value: float | None) -> str:
    return "n/a" if value is None else f"{value:6.1f}"


def _dashed_line(cv2: Any, frame: Any, start: Point, end: Point, color: Color, thickness: int, dash: int) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    steps = max(1, int(length // max(dash, 1)))
    for step in range(0, steps, 2):
        t0 = step / steps
        t1 = min(1.0, (step + 1) / steps)
        a = (round(start[0] + dx * t0), round(start[1] + dy * t0))
        b = (round(start[0] + dx * t1), round(start[1] + dy * t1))
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)


def _offset(point: Point, direction: FloatPoint, distance: float) -> Point:
    return round(point[0] + direction[0] * distance), round(point[1] + direction[1] * distance)


def _scale_for(panel_width: float) -> float:
    return max(0.6, min(2.5, panel_width / REFERENCE_PANEL_WIDTH))


def _rounded(point: FloatPoint) -> Point:
    return round(point[0]), round(point[1])
