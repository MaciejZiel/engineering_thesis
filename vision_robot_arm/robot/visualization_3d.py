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
GRID: Color = (52, 49, 47)
GRID_MAJOR: Color = (72, 67, 63)
LEFT_ARM: Color = (235, 165, 91)
RIGHT_ARM: Color = (104, 178, 241)
TARGET_ARM: Color = (112, 105, 98)
HUMAN_ARM: Color = (101, 211, 165)
HUMAN_HAND: Color = (113, 225, 186)
JOINT: Color = (238, 236, 233)
BASE: Color = (88, 82, 77)
AXIS_X: Color = (90, 110, 238)
AXIS_Y: Color = (104, 205, 126)
AXIS_Z: Color = (74, 184, 246)

HUMAN_OFFSET: Point3 = (0.0, -0.62, 0.92)
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
    position: Point3 = (1.45, -2.15, 1.45)
    target: Point3 = (0.0, 0.18, 0.42)
    focal_scale: float = 1.05


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
    project = _projector(np, width, height, camera)
    segments: list[Segment3D] = []

    _append_grid(segments)
    _append_axes(segments)
    _append_robots(segments, robot_state)
    _append_tracked_arms(segments, pose_state, indices)

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

    for side, base in (
        (ARM_LEFT, (-BASE_SEPARATION_M / 2, 0.0, 0.0)),
        (ARM_RIGHT, (BASE_SEPARATION_M / 2, 0.0, 0.0)),
    ):
        point = project(base)
        if point is not None:
            cv2.circle(
                canvas,
                point[:2],
                max(4, round(min(width, height) * 0.026)),
                BASE,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                point[:2],
                max(2, round(min(width, height) * 0.012)),
                JOINT,
                -1,
                cv2.LINE_AA,
            )

    _draw_joint_markers(cv2, canvas, project, robot_state, width, height)
    _draw_labels(cv2, canvas, project, width, height)
    _draw_tracking_coordinates(cv2, canvas, pose_state, indices, width, height)


def _append_grid(segments: list[Segment3D]) -> None:
    for step in range(-10, 11):
        coordinate = step * 0.1
        color = GRID_MAJOR if step % 5 == 0 else GRID
        segments.append(Segment3D((coordinate, -0.45, 0), (coordinate, 1.0, 0), color))
    for step in range(-4, 11):
        coordinate = step * 0.1
        color = GRID_MAJOR if step % 5 == 0 else GRID
        segments.append(Segment3D((-1.0, coordinate, 0), (1.0, coordinate, 0), color))


def _append_axes(segments: list[Segment3D]) -> None:
    origin = (-0.92, -0.38, 0.015)
    segments.extend(
        (
            Segment3D(origin, (origin[0] + 0.26, origin[1], origin[2]), AXIS_X, 2),
            Segment3D(origin, (origin[0], origin[1] + 0.26, origin[2]), AXIS_Y, 2),
            Segment3D(origin, (origin[0], origin[1], origin[2] + 0.26), AXIS_Z, 2),
        )
    )


def _append_robots(segments: list[Segment3D], state: RobotState | None) -> None:
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
                Segment3D(a, b, TARGET_ARM, 1, True)
                for a, b in zip(target_points, target_points[1:])
            )
        segments.extend(
            Segment3D(a, b, color, 4)
            for a, b in zip(current_points, current_points[1:])
        )


def _append_tracked_arms(
    segments: list[Segment3D], state: PoseState | None, indices: dict[str, int]
) -> None:
    if state is None or state.world_landmarks is None:
        return
    body_coordinates = state.body_landmarks is not None
    world = state.body_landmarks or state.world_landmarks
    for side in ("left", "right"):
        names = (
            f"{side.upper()}_SHOULDER",
            f"{side.upper()}_ELBOW",
            f"{side.upper()}_WRIST",
        )
        points = []
        for name in names:
            index = indices.get(name)
            if index is None or not 0 <= index < len(world):
                points = []
                break
            points.append(
                _body_to_scene(world[index])
                if body_coordinates
                else _pose_to_scene(world[index])
            )
        segments.extend(
            Segment3D(a, b, HUMAN_ARM, 3) for a, b in zip(points, points[1:])
        )
        hand = (
            state.hand_body_landmarks.get(side, ())
            if body_coordinates
            else state.hand_world_landmarks.get(side, ())
        )
        if hand:
            hand_points = [
                _body_to_scene(point) if body_coordinates else _pose_to_scene(point)
                for point in hand
            ]
            segments.extend(
                Segment3D(hand_points[a], hand_points[b], HUMAN_HAND, 1)
                for a, b in HAND_CONNECTIONS
                if max(a, b) < len(hand_points)
            )


def _draw_joint_markers(
    cv2: Any,
    canvas: Any,
    project: Any,
    state: RobotState | None,
    width: int,
    height: int,
) -> None:
    radius = max(2, round(min(width, height) * 0.014))
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
                cv2.circle(canvas, projected[:2], radius + 1, color, -1, cv2.LINE_AA)
                cv2.circle(
                    canvas, projected[:2], max(1, radius // 2), JOINT, -1, cv2.LINE_AA
                )


def _draw_labels(cv2: Any, canvas: Any, project: Any, width: int, height: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.32, min(width, height) / 900)
    for label, endpoint, color in (
        ("X", (-0.66, -0.38, 0.015), AXIS_X),
        ("Y depth", (-0.92, -0.12, 0.015), AXIS_Y),
        ("Z up", (-0.92, -0.38, 0.275), AXIS_Z),
    ):
        point = project(endpoint)
        if point is not None:
            cv2.putText(canvas, label, point[:2], font, scale, color, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "body XYZ: X right / Y forward / Z up",
        (8, max(14, round(height * 0.08))),
        font,
        scale,
        HUMAN_ARM,
        1,
        cv2.LINE_AA,
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
    scale = max(0.28, min(width, height) / 1100)
    y = max(30, round(height * 0.14))
    source = state.body_frame.source.replace("shoulders_", "") if state.body_frame else ""
    cv2.putText(
        canvas,
        f"origin: shoulders / vertical: {source}",
        (8, y),
        font,
        scale,
        GRID_MAJOR,
        1,
        cv2.LINE_AA,
    )
    y += max(13, round(height * 0.055))
    for side in ("left", "right"):
        values = []
        for joint in ("ELBOW", "WRIST"):
            index = indices.get(f"{side.upper()}_{joint}")
            if index is None or not 0 <= index < len(state.body_landmarks):
                continue
            point = state.body_landmarks[index]
            if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
                continue
            values.append(
                f"{joint[0]} {point.x:+.2f} {point.y:+.2f} {point.z:+.2f}"
            )
        if values:
            cv2.putText(
                canvas,
                f"{side[0].upper()}  " + "  |  ".join(values) + " m",
                (8, y),
                font,
                scale,
                HUMAN_ARM,
                1,
                cv2.LINE_AA,
            )
            y += max(13, round(height * 0.055))
    cv2.putText(
        canvas,
        "0.5 m base spacing",
        (8, height - 8),
        font,
        scale,
        GRID_MAJOR,
        1,
        cv2.LINE_AA,
    )


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


def _projector(np: Any, width: int, height: int, camera: Camera3D) -> Any:
    position = np.asarray(camera.position, dtype=float)
    target = np.asarray(camera.target, dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    focal = min(width, height) * camera.focal_scale

    def project(point: Point3) -> tuple[int, int, float] | None:
        if not all(math.isfinite(value) for value in point):
            return None
        relative = np.asarray(point, dtype=float) - position
        depth = float(np.dot(relative, forward))
        if not math.isfinite(depth) or depth <= 0.05:
            return None
        x = width / 2 + focal * float(np.dot(relative, right)) / depth
        y = height / 2 - focal * float(np.dot(relative, up)) / depth
        return round(x), round(y), depth

    return project

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
