import math
from typing import Any

from vision_robot_arm.core.hud import fill_translucent, put_text
from vision_robot_arm.robot.targets import (
    ARM_LEFT,
    ARM_RIGHT,
    GRIPPER_CLOSE,
    HELD_JOINTS,
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
Color = tuple[int, int, int]

BACKGROUND: Color = (24, 24, 24)
PANEL_BORDER: Color = (90, 90, 90)
CURRENT_ARM: Color = (0, 145, 255)
TARGET_ARM: Color = (82, 94, 108)
GRIPPER_COLOR: Color = (40, 190, 255)
TITLE_COLOR: Color = (235, 235, 235)
VALUE_COLOR: Color = (200, 200, 200)
MUTED_COLOR: Color = (130, 130, 130)
TEXT_SCALE = 0.45
LINE_HEIGHT = 18
PADDING = 10
HAND_LINK_RATIO = 0.7
GRIPPER_RATIO = 0.42
COMPACT_BACKGROUND: Color = (30, 28, 27)
COMPACT_LINK: Color = (112, 169, 238)
COMPACT_LINK_EDGE: Color = (52, 86, 134)
COMPACT_JOINT: Color = (241, 239, 237)
COMPACT_TARGET: Color = (108, 103, 98)
COMPACT_PEDESTAL: Color = (72, 67, 62)
COMPACT_GRIP_OPEN: Color = (162, 200, 137)
COMPACT_GRIP_CLOSED: Color = (134, 136, 240)
COMPACT_FIT_MARGIN = 1.1
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
    upper_direction = math.radians(shoulder_deg + 90.0)
    forearm_direction = upper_direction + math.radians(elbow_deg)
    hand_direction = forearm_direction + math.radians(wrist_deg)

    def advance(start: tuple[float, float], direction: float, length: float) -> tuple[float, float]:
        return (
            start[0] + sign * length * math.cos(direction),
            start[1] - length * math.sin(direction),
        )

    elbow = advance(base, upper_direction, link_length)
    wrist = advance(elbow, forearm_direction, link_length)
    tip = advance(wrist, hand_direction, link_length * HAND_LINK_RATIO)
    return base, _rounded(elbow), _rounded(wrist), _rounded(tip)


def draw_simulation(cv2: Any, canvas: Any, state: RobotState | None, *, compact: bool = False) -> None:
    if compact:
        draw_compact_simulation(cv2, canvas, state)
        return
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
            title=f"{arm} {ROBOT_MODEL}",
            mirror=arm == ARM_LEFT,
        )

    lift = "on" if state is not None and state.lift_mode else "off"
    footer_scale = TEXT_SCALE * _scale_for(min(width, height))
    put_text(
        cv2,
        canvas,
        f"lift mode: {lift}    grey = target, orange = current    angles in UR joint degrees",
        (margin, height - margin - round(6 * footer_scale / TEXT_SCALE)),
        MUTED_COLOR,
        footer_scale,
    )


def draw_compact_simulation(cv2: Any, canvas: Any, state: RobotState | None) -> None:
    """Draw only the schematic arms; the dashboard owns labels and telemetry.

    Both arms share one link scale so their sizes stay comparable. Each arm is
    clipped to its own viewport and the fit leaves room for the gripper jaws.
    No pose is drawn when the backend has not supplied an arm state.
    """
    canvas[:] = COMPACT_BACKGROUND
    height, width = canvas.shape[:2]
    geometry: dict[str, tuple[float, float, float, float]] = {}
    for name in DISPLAY_ORDER:
        arm = state.arm(name) if state is not None else None
        if arm is None:
            continue
        points = []
        for pose in (arm.joints, arm.targets):
            points.extend(_points_for(pose, (0, 0), 1000.0, name == ARM_LEFT))
        xs = [point[0] / 1000.0 for point in points]
        ys = [point[1] / 1000.0 for point in points]
        geometry[name] = (min(xs), min(ys), max(xs), max(ys))
    if not geometry:
        return

    half_width = width // 2
    link = min(
        min(
            half_width * 0.84 / (right - left + COMPACT_FIT_MARGIN),
            height * 0.84 / (bottom - top + COMPACT_FIT_MARGIN),
        )
        for left, top, right, bottom in geometry.values()
    )
    if link < 4:
        return

    for column, name in enumerate(DISPLAY_ORDER):
        arm = state.arm(name) if state is not None else None
        if arm is None:
            continue
        left, right = column * half_width, (column + 1) * half_width
        view = canvas[:, left:right]
        mirror = name == ARM_LEFT
        min_x, min_y, max_x, max_y = geometry[name]
        base = (
            round((right - left) / 2 - (min_x + max_x) * link / 2),
            round(height / 2 - (min_y + max_y) * link / 2),
        )
        thickness = max(3, round(link * 0.16))
        _draw_pedestal(cv2, view, base, link, thickness)
        target_points = _points_for(arm.targets, base, link, mirror)
        for start_point, end_point in zip(target_points, target_points[1:]):
            _dashed_line(cv2, view, start_point, end_point, COMPACT_TARGET, max(1, thickness // 3), max(4, round(link * 0.12)))
        points = _points_for(arm.joints, base, link, mirror)
        _draw_compact_links(cv2, view, points, thickness)
        _draw_jaw_gripper(
            cv2,
            view,
            points[2],
            points[3],
            arm.gripper == GRIPPER_CLOSE,
            link * GRIPPER_RATIO,
            max(2, round(thickness * 0.5)),
        )


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


def _draw_pedestal(cv2: Any, view: Any, base: Point, link: float, thickness: int) -> None:
    half = max(5, round(link * 0.28))
    top = base[1] + max(3, thickness // 2)
    cv2.rectangle(
        view,
        (base[0] - half, top),
        (base[0] + half, top + max(3, round(link * 0.14))),
        COMPACT_PEDESTAL,
        -1,
    )


def _draw_compact_links(cv2: Any, view: Any, points: tuple[Point, Point, Point, Point], thickness: int) -> None:
    widths = (thickness, thickness, max(2, thickness - 2))
    for (start_point, end_point), width_px in zip(zip(points, points[1:]), widths):
        cv2.line(view, start_point, end_point, COMPACT_LINK_EDGE, width_px + 3, cv2.LINE_AA)
    for (start_point, end_point), width_px in zip(zip(points, points[1:]), widths):
        cv2.line(view, start_point, end_point, COMPACT_LINK, width_px, cv2.LINE_AA)
    radius = max(3, round(thickness * 0.75))
    for point in points[:3]:
        cv2.circle(view, point, radius + 2, COMPACT_LINK_EDGE, -1, cv2.LINE_AA)
        cv2.circle(view, point, radius, COMPACT_JOINT, -1, cv2.LINE_AA)


def _draw_jaw_gripper(
    cv2: Any,
    view: Any,
    wrist: Point,
    tip: Point,
    closed: bool,
    gripper_length: float,
    thickness: int,
) -> None:
    dx, dy = tip[0] - wrist[0], tip[1] - wrist[1]
    length = math.hypot(dx, dy) or 1.0
    direction = (dx / length, dy / length)
    normal = (-direction[1], direction[0])
    color = COMPACT_GRIP_CLOSED if closed else COMPACT_GRIP_OPEN
    palm_half = gripper_length * (0.22 if closed else 0.55)
    mount_radius = max(2, thickness)
    cv2.circle(view, tip, mount_radius + 2, COMPACT_LINK_EDGE, -1, cv2.LINE_AA)
    cv2.circle(view, tip, mount_radius, color, -1, cv2.LINE_AA)
    palm_a = _offset(tip, normal, palm_half)
    palm_b = _offset(tip, normal, -palm_half)
    cv2.line(view, palm_a, palm_b, color, thickness, cv2.LINE_AA)
    for side_point, inward in ((palm_a, -1.0), (palm_b, 1.0)):
        jaw_end = _offset(side_point, direction, gripper_length)
        cv2.line(view, side_point, jaw_end, color, thickness, cv2.LINE_AA)
        pad_start = _offset(jaw_end, direction, -gripper_length * 0.35)
        cv2.line(view, pad_start, _offset(pad_start, normal, inward * palm_half * 0.4), color, max(1, thickness - 1), cv2.LINE_AA)


def _dashed_line(cv2: Any, view: Any, start: Point, end: Point, color: Color, thickness: int, dash: int) -> None:
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
        cv2.line(view, a, b, color, thickness, cv2.LINE_AA)


def _offset(point: Point, direction: tuple[float, float], distance: float) -> Point:
    return round(point[0] + direction[0] * distance), round(point[1] + direction[1] * distance)


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
    factor = _scale_for(min(width, height))
    padding = round(PADDING * factor)
    line_height = round(LINE_HEIGHT * factor)
    text_scale = TEXT_SCALE * factor
    thick_text = 2 if factor >= 1.4 else 1

    fill_translucent(cv2, frame, (left, top), (right, bottom))
    cv2.rectangle(frame, (left, top), (right, bottom), PANEL_BORDER, 1, cv2.LINE_AA)

    lines = [title] + [_joint_line(joint, state) for joint in MAPPED_JOINTS]
    lines.append(f"gripper {state.gripper if state is not None else 'n/a'}")
    lines.append(_held_line(state))
    for row, text in enumerate(lines):
        put_text(
            cv2,
            frame,
            text,
            (left + padding, top + padding + (row + 1) * line_height - round(4 * factor)),
            TITLE_COLOR if row == 0 else (MUTED_COLOR if row == len(lines) - 1 else VALUE_COLOR),
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
        angles.get(JOINT_SHOULDER, UR_HOME_DEG[JOINT_SHOULDER]),
        angles.get(JOINT_ELBOW, UR_HOME_DEG[JOINT_ELBOW]),
        angles.get(JOINT_WRIST_1, UR_HOME_DEG[JOINT_WRIST_1]),
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
    color: Color = GRIPPER_COLOR,
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
        cv2.line(frame, start, end, color, 2, cv2.LINE_AA)


def _joint_line(joint: str, state: ArmState | None) -> str:
    if state is None:
        return f"{joint} n/a"
    current = state.joints.get(joint)
    target = state.targets.get(joint)
    if current is None or target is None:
        return f"{joint} n/a"
    return f"{joint} {current:6.1f} -> {target:6.1f}"


def _held_line(state: ArmState | None) -> str:
    values = []
    for joint in HELD_JOINTS:
        value = state.joints.get(joint, UR_HOME_DEG[joint]) if state is not None else UR_HOME_DEG[joint]
        values.append(f"{joint} {value:.0f}")
    return "held: " + "  ".join(values)


def _scale_for(extent: int) -> float:
    return max(0.6, extent / 240.0)


def _rounded(point: tuple[float, float]) -> Point:
    return round(point[0]), round(point[1])
