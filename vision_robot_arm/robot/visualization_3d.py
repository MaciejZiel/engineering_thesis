"""Perspective 3D workspace preview for two UR7e arms and tracked human arms."""

import math
from dataclasses import dataclass
from typing import Any

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.cartesian_mapping import BASE_SEPARATION_M
from vision_robot_arm.robot.kinematics import ur7e_joint_points
from vision_robot_arm.robot.targets import (
    ARM_LEFT,
    ARM_RIGHT,
    UR_HOME_DEG,
    RobotState,
)

Point3 = tuple[float, float, float]
Color = tuple[int, int, int]

BACKGROUND: Color = (29, 27, 26)
GRID: Color = (45, 42, 40)
GRID_MAJOR: Color = (78, 73, 68)
FLOOR_EDGE: Color = (96, 90, 84)
LEFT_ARM: Color = (235, 165, 91)
RIGHT_ARM: Color = (104, 178, 241)
TARGET_ARM: Color = (118, 111, 104)
TARGET_TCP: Color = (85, 220, 245)
HUMAN_ARM: Color = (101, 211, 165)
HUMAN_HAND: Color = (113, 225, 186)
DROP_LINE: Color = (70, 66, 62)
JOINT: Color = (238, 236, 233)
BASE: Color = (88, 82, 77)
LABEL: Color = (176, 169, 161)
AXIS_X: Color = (90, 110, 238)
AXIS_Y: Color = (104, 205, 126)
AXIS_Z: Color = (74, 184, 246)

HUMAN_OFFSET: Point3 = (0.0, -0.62, 0.92)

# The view frames this fixed volume rather than the live pose, so the preview never
# zooms or pans while you move. It holds the space both arms and the body actually
# work in; a rare fully extended reach may touch the edge, which costs far less than
# a camera that breathes on every frame.
CONTENT_BOX: tuple[tuple[float, float], ...] = (
    (-0.74, 0.74),
    (-0.76, 0.22),
    (0.0, 1.12),
)
# The floor is a reference, not content: it may run off the panel edges.
GRID_BOX: tuple[tuple[float, float], ...] = ((-1.0, 1.0), (-0.9, 0.5))
# A panel narrower than this belongs to the dashboard's own labels; drawing Hershey
# text into it only buries the scene. See docs/ARCHITECTURE.md.
LABEL_MIN_WIDTH = 720
GRID_STEP_M = 0.25
GRID_MAJOR_EVERY = 2

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class Camera3D:
    position: Point3 = (1.40, -2.05, 0.98)
    target: Point3 = (0.0, -0.18, 0.52)
    margin: float = 0.05


@dataclass(frozen=True)
class Segment3D:
    start: Point3
    end: Point3
    color: Color
    thickness: int = 1
    dashed: bool = False


def draw_workspace_3d(
    cv2: Any,
    np: Any,
    canvas: Any,
    robot_state: RobotState | None,
    pose_state: PoseState | None,
    indices: dict[str, int],
    camera: Camera3D = Camera3D(),
) -> None:
    canvas[:] = BACKGROUND
    height, width = canvas.shape[:2]
    scale = _scale(width, height)
    project = _projector(np, width, height, camera)
    segments: list[Segment3D] = []

    _append_grid(segments)
    _append_robots(segments, robot_state, scale)
    _append_tracked_arms(segments, pose_state, indices, scale)
    _append_drop_lines(segments, robot_state, pose_state, indices)

    visible = []
    for segment in segments:
        projected = project(segment.start), project(segment.end)
        if projected[0] is None or projected[1] is None:
            continue
        depth = (projected[0][2] + projected[1][2]) / 2
        visible.append((depth, segment, projected[0][:2], projected[1][:2]))
    for _, segment, start, end in sorted(
        visible, reverse=True, key=lambda item: item[0]
    ):
        if segment.dashed:
            _dashed_line(cv2, canvas, start, end, segment.color, segment.thickness)
        else:
            cv2.line(canvas, start, end, segment.color, segment.thickness, cv2.LINE_AA)

    _draw_bases(cv2, canvas, project, scale)
    _draw_joint_markers(cv2, canvas, project, robot_state, scale)
    _draw_tcp_targets(cv2, canvas, project, robot_state, scale)
    _draw_axis_gizmo(cv2, canvas, camera, width, height, scale)
    if width >= LABEL_MIN_WIDTH:
        _draw_tracking_coordinates(cv2, canvas, pose_state, indices, width, height)


def _scale(width: int, height: int) -> float:
    """Stroke weights follow the panel, so the small preview is not drawn in hairlines."""
    return max(0.62, min(2.6, min(width, height) / 300))


def _append_grid(segments: list[Segment3D]) -> None:
    (x_min, x_max), (y_min, y_max) = GRID_BOX
    steps_x = round((x_max - x_min) / GRID_STEP_M)
    steps_y = round((y_max - y_min) / GRID_STEP_M)
    for step in range(steps_x + 1):
        x = x_min + step * GRID_STEP_M
        major = step % GRID_MAJOR_EVERY == 0
        segments.append(
            Segment3D((x, y_min, 0), (x, y_max, 0), GRID_MAJOR if major else GRID)
        )
    for step in range(steps_y + 1):
        y = y_min + step * GRID_STEP_M
        major = step % GRID_MAJOR_EVERY == 0
        segments.append(
            Segment3D((x_min, y, 0), (x_max, y, 0), GRID_MAJOR if major else GRID)
        )
    # A brighter rim reads as a floor instead of as another layer of mesh.
    for a, b in (
        ((x_min, y_min, 0), (x_max, y_min, 0)),
        ((x_max, y_min, 0), (x_max, y_max, 0)),
        ((x_max, y_max, 0), (x_min, y_max, 0)),
        ((x_min, y_max, 0), (x_min, y_min, 0)),
    ):
        segments.append(Segment3D(a, b, FLOOR_EDGE))


def _append_robots(
    segments: list[Segment3D], state: RobotState | None, scale: float
) -> None:
    link = max(3, round(5 * scale))
    for side, x, color in (
        (ARM_LEFT, -BASE_SEPARATION_M / 2, LEFT_ARM),
        (ARM_RIGHT, BASE_SEPARATION_M / 2, RIGHT_ARM),
    ):
        arm = state.arm(side) if state is not None else None
        current = arm.joints if arm is not None else UR_HOME_DEG
        target = arm.targets if arm is not None else current
        base = (x, 0.0, 0.0)
        target_points = ur7e_joint_points(target, base)
        current_points = ur7e_joint_points(current, base)
        if target != current:
            segments.extend(
                Segment3D(a, b, TARGET_ARM, max(1, round(1.5 * scale)), True)
                for a, b in zip(target_points, target_points[1:])
            )
        segments.extend(
            Segment3D(a, b, color, link)
            for a, b in zip(current_points, current_points[1:])
        )


def _append_tracked_arms(
    segments: list[Segment3D],
    state: PoseState | None,
    indices: dict[str, int],
    scale: float,
) -> None:
    if state is None or state.world_landmarks is None:
        return
    for side in ("left", "right"):
        points = _arm_points(state, indices, side)
        segments.extend(
            Segment3D(a, b, HUMAN_ARM, max(2, round(3 * scale)))
            for a, b in zip(points, points[1:])
        )
        hand_points = _hand_points(state, side)
        segments.extend(
            Segment3D(hand_points[a], hand_points[b], HUMAN_HAND, max(1, round(scale)))
            for a, b in HAND_CONNECTIONS
            if max(a, b) < len(hand_points)
        )


def _append_drop_lines(
    segments: list[Segment3D],
    state: RobotState | None,
    pose_state: PoseState | None,
    indices: dict[str, int],
) -> None:
    """Without a line to the floor a perspective view cannot show how high a hand is."""
    tips: list[Point3] = []
    for side, x in (
        (ARM_LEFT, -BASE_SEPARATION_M / 2),
        (ARM_RIGHT, BASE_SEPARATION_M / 2),
    ):
        arm = state.arm(side) if state is not None else None
        joints = arm.joints if arm is not None else UR_HOME_DEG
        tips.append(ur7e_joint_points(joints, (x, 0.0, 0.0))[-1])
    if pose_state is not None and pose_state.world_landmarks is not None:
        for side in ("left", "right"):
            points = _arm_points(pose_state, indices, side)
            if points:
                tips.append(points[-1])
    for tip in tips:
        if all(math.isfinite(value) for value in tip) and tip[2] > 0.05:
            segments.append(Segment3D(tip, (tip[0], tip[1], 0.0), DROP_LINE, 1, True))


def _arm_points(state: PoseState, indices: dict[str, int], side: str) -> list[Point3]:
    body_coordinates = state.body_landmarks is not None
    world = state.body_landmarks or state.world_landmarks
    if world is None:
        return []
    points: list[Point3] = []
    for joint in ("SHOULDER", "ELBOW", "WRIST"):
        index = indices.get(f"{side.upper()}_{joint}")
        if index is None or not 0 <= index < len(world):
            return []
        point = world[index]
        points.append(
            _body_to_scene(point) if body_coordinates else _pose_to_scene(point)
        )
    return points


def _hand_points(state: PoseState, side: str) -> list[Point3]:
    body_coordinates = state.body_landmarks is not None
    hand = (
        state.hand_body_landmarks.get(side, ())
        if body_coordinates
        else state.hand_world_landmarks.get(side, ())
    )
    return [
        _body_to_scene(point) if body_coordinates else _pose_to_scene(point)
        for point in hand
    ]


def _draw_bases(cv2: Any, canvas: Any, project: Any, scale: float) -> None:
    """The ring carries the arm's colour, so no legend is needed to tell them apart."""
    for x, color in (
        (-BASE_SEPARATION_M / 2, LEFT_ARM),
        (BASE_SEPARATION_M / 2, RIGHT_ARM),
    ):
        point = project((x, 0.0, 0.0))
        if point is None:
            continue
        radius = max(5, round(9 * scale))
        cv2.circle(canvas, point[:2], radius, BASE, -1, cv2.LINE_AA)
        cv2.circle(canvas, point[:2], radius, color, max(1, round(1.6 * scale)), cv2.LINE_AA)
        cv2.circle(canvas, point[:2], max(2, round(3 * scale)), JOINT, -1, cv2.LINE_AA)


def _draw_joint_markers(
    cv2: Any, canvas: Any, project: Any, state: RobotState | None, scale: float
) -> None:
    radius = max(3, round(5 * scale))
    for side, x, color in (
        (ARM_LEFT, -BASE_SEPARATION_M / 2, LEFT_ARM),
        (ARM_RIGHT, BASE_SEPARATION_M / 2, RIGHT_ARM),
    ):
        arm = state.arm(side) if state is not None else None
        points = ur7e_joint_points(
            arm.joints if arm is not None else UR_HOME_DEG, (x, 0, 0)
        )
        for point in points[1:]:
            projected = project(point)
            if projected is not None:
                cv2.circle(canvas, projected[:2], radius, color, -1, cv2.LINE_AA)
                cv2.circle(
                    canvas, projected[:2], max(1, radius // 2), JOINT, -1, cv2.LINE_AA
                )


def _draw_tcp_targets(
    cv2: Any, canvas: Any, project: Any, state: RobotState | None, scale: float
) -> None:
    if state is None:
        return
    radius = max(5, round(8 * scale))
    thickness = max(1, round(scale))
    for side in (ARM_LEFT, ARM_RIGHT):
        arm = state.arm(side)
        if arm is None or arm.tcp_target is None:
            continue
        point = project(arm.tcp_target)
        if point is None:
            continue
        x, y = point[:2]
        reach = radius + max(2, round(3 * scale))
        cv2.circle(canvas, (x, y), radius, TARGET_TCP, thickness, cv2.LINE_AA)
        cv2.line(
            canvas, (x - reach, y), (x + reach, y), TARGET_TCP, thickness, cv2.LINE_AA
        )
        cv2.line(
            canvas, (x, y - reach), (x, y + reach), TARGET_TCP, thickness, cv2.LINE_AA
        )


def _draw_axis_gizmo(
    cv2: Any, canvas: Any, camera: Camera3D, width: int, height: int, scale: float
) -> None:
    """Pinned to a corner: inside the scene the gizmo only tangles with the arms."""
    size = max(13, min(34, round(min(width, height) * 0.1)))
    origin = (round(size * 0.9), height - round(size * 0.9))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, min(0.42, 0.34 * scale))
    for (dx, dy), color, label in zip(
        _screen_basis(camera), (AXIS_X, AXIS_Y, AXIS_Z), ("X", "Y", "Z")
    ):
        end = (round(origin[0] + dx * size), round(origin[1] + dy * size))
        cv2.line(canvas, origin, end, color, max(1, round(1.6 * scale)), cv2.LINE_AA)
        tip = (round(origin[0] + dx * size * 1.34), round(origin[1] + dy * size * 1.34))
        cv2.putText(canvas, label, tip, font, font_scale, color, 1, cv2.LINE_AA)
    cv2.circle(canvas, origin, max(2, round(2 * scale)), LABEL, -1, cv2.LINE_AA)


def _screen_basis(camera: Camera3D) -> tuple[tuple[float, float], ...]:
    forward = _unit(tuple(t - p for t, p in zip(camera.target, camera.position)))
    right = _unit(_cross(forward, (0.0, 0.0, 1.0)))
    up = _cross(right, forward)
    return tuple(
        (_dot(axis, right), -_dot(axis, up))
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )


def _draw_tracking_coordinates(
    cv2: Any,
    canvas: Any,
    state: PoseState | None,
    indices: dict[str, int],
    width: int,
    height: int,
) -> None:
    if state is None or state.body_landmarks is None:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.34, min(width, height) / 1400)
    step = max(15, round(height * 0.035))
    y = step
    source = (
        state.body_frame.source.replace("shoulders_", "") if state.body_frame else ""
    )
    cv2.putText(
        canvas,
        f"body XYZ in metres  -  origin shoulders  -  vertical {source}",
        (10, y),
        font,
        scale,
        LABEL,
        1,
        cv2.LINE_AA,
    )
    y += step
    for side in ("left", "right"):
        values = []
        for joint in ("ELBOW", "WRIST"):
            index = indices.get(f"{side.upper()}_{joint}")
            if index is None or not 0 <= index < len(state.body_landmarks):
                continue
            point = state.body_landmarks[index]
            if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
                continue
            values.append(f"{joint[0]} {point.x:+.2f} {point.y:+.2f} {point.z:+.2f}")
        if values:
            cv2.putText(
                canvas,
                f"{side[0].upper()}  " + "   ".join(values),
                (10, y),
                font,
                scale,
                HUMAN_ARM,
                1,
                cv2.LINE_AA,
            )
            y += step


def _pose_to_scene(point: Any) -> Point3:
    return (
        float(point.x) + HUMAN_OFFSET[0],
        float(point.z) + HUMAN_OFFSET[1],
        -float(point.y) + HUMAN_OFFSET[2],
    )


def _body_to_scene(point: Any) -> Point3:
    """Body coordinates already use X right, Y forward/depth, Z up."""
    return (
        float(point.x) + HUMAN_OFFSET[0],
        float(point.y) + HUMAN_OFFSET[1],
        float(point.z) + HUMAN_OFFSET[2],
    )


def _cross(a: Point3, b: Point3) -> Point3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Point3, b: Point3) -> float:
    return sum(x * y for x, y in zip(a, b))


def _unit(vector: Point3) -> Point3:
    length = math.sqrt(_dot(vector, vector)) or 1.0
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _projector(np: Any, width: int, height: int, camera: Camera3D) -> Any:
    position = np.asarray(camera.position, dtype=float)
    target = np.asarray(camera.target, dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    def normalized(point: Point3) -> tuple[float, float, float] | None:
        if not all(math.isfinite(value) for value in point):
            return None
        relative = np.asarray(point, dtype=float) - position
        depth = float(np.dot(relative, forward))
        if not math.isfinite(depth) or depth <= 0.05:
            return None
        return (
            float(np.dot(relative, right)) / depth,
            -float(np.dot(relative, up)) / depth,
            depth,
        )

    focal, offset_x, offset_y = _fit(normalized, width, height, camera.margin)

    def project(point: Point3) -> tuple[int, int, float] | None:
        view = normalized(point)
        if view is None:
            return None
        return (
            round(offset_x + focal * view[0]),
            round(offset_y + focal * view[1]),
            view[2],
        )

    return project


def _fit(
    normalized: Any, width: int, height: int, margin: float
) -> tuple[float, float, float]:
    """Scale and centre the fixed content box to fill the panel, whatever its shape."""
    seen = [
        corner
        for corner in (
            normalized((x, y, z))
            for x in CONTENT_BOX[0]
            for y in CONTENT_BOX[1]
            for z in CONTENT_BOX[2]
        )
        if corner is not None
    ]
    if not seen:
        return float(min(width, height)), width / 2, height / 2
    left = min(corner[0] for corner in seen)
    right = max(corner[0] for corner in seen)
    top = min(corner[1] for corner in seen)
    bottom = max(corner[1] for corner in seen)
    inset = margin * min(width, height)
    focal = min(
        (width - 2 * inset) / max(right - left, 1e-6),
        (height - 2 * inset) / max(bottom - top, 1e-6),
    )
    return (
        focal,
        width / 2 - focal * (left + right) / 2,
        height / 2 - focal * (top + bottom) / 2,
    )


def _dashed_line(
    cv2: Any,
    canvas: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    color: Color,
    thickness: int,
) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    steps = max(1, round(length / 8))
    for step in range(0, steps, 2):
        a = (round(start[0] + dx * step / steps), round(start[1] + dy * step / steps))
        next_step = min(steps, step + 1)
        b = (
            round(start[0] + dx * next_step / steps),
            round(start[1] + dy * next_step / steps),
        )
        cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)
