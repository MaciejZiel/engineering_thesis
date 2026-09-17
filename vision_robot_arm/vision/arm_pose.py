"""Arm elevation measured against the torso, so a seated person still drives the robot.

`metrics.ANGLE_DEFINITIONS` measures the shoulder as elbow-shoulder-hip. At a desk the
hips are outside the frame, so that angle is missing on almost every frame. Elevation
needs only the shoulder and the elbow: 0 = arm hanging down, 90 = horizontal,
180 = raised straight up.
"""

import math
from typing import Any

from vision_robot_arm.vision.landmarks import is_reliable

SIDES = ("left", "right")
ELEVATION_SUFFIX = "shoulder_elevation"
ELEVATION_ANGLE_NAMES = tuple(f"{side}_{ELEVATION_SUFFIX}" for side in SIDES)
IMAGE_DOWN = (0.0, 1.0)
# An upper arm pointing at the camera projects to almost nothing, and the angle of a
# two-pixel vector is landmark noise. Below this length the joint is left unmeasured.
MIN_ARM_LENGTH = 0.04
MIN_ARM_LENGTH_PER_SHOULDER_WIDTH = 0.25

Vector = tuple[float, ...]


def arm_elevation_angles(
    landmarks: list[Any],
    indices: dict[str, int],
    min_visibility: float = 0.55,
    aspect_ratio: float = 1.0,
    world_landmarks: list[Any] | None = None,
) -> dict[str, float]:
    if world_landmarks is not None:
        return _world_arm_elevation_angles(
            landmarks, world_landmarks, indices, min_visibility
        )
    down = torso_down_vector(landmarks, indices, min_visibility, aspect_ratio)
    minimum_length = _minimum_arm_length(
        landmarks, indices, min_visibility, aspect_ratio
    )
    angles: dict[str, float] = {}
    for side in SIDES:
        shoulder = _point(
            landmarks, indices, f"{side.upper()}_SHOULDER", min_visibility
        )
        elbow = _point(landmarks, indices, f"{side.upper()}_ELBOW", min_visibility)
        if shoulder is None or elbow is None:
            continue
        arm = _vector(shoulder, elbow, aspect_ratio)
        if _norm(arm) < minimum_length:
            continue
        elevation = vector_angle(arm, down)
        if elevation is not None:
            angles[f"{side}_{ELEVATION_SUFFIX}"] = elevation
    return angles


def _world_arm_elevation_angles(
    image_landmarks: list[Any],
    world_landmarks: list[Any],
    indices: dict[str, int],
    min_visibility: float,
) -> dict[str, float]:
    """3D arm elevation relative to the torso; never substitute an image-plane axis."""
    names = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")
    image_points = [
        _point(image_landmarks, indices, name, min_visibility) for name in names
    ]
    world_points = [_world_point(world_landmarks, indices, name) for name in names]
    shoulders_valid = all(
        point is not None for point in (*image_points[:2], *world_points[:2])
    )
    if not shoulders_valid:
        return {}
    hips_valid = all(
        point is not None for point in (*image_points[2:], *world_points[2:])
    )
    if hips_valid:
        left_shoulder, right_shoulder, left_hip, right_hip = world_points
        shoulder_center = _midpoint3(left_shoulder, right_shoulder)
        hip_center = _midpoint3(left_hip, right_hip)
        down = _vector3(shoulder_center, hip_center)
    else:
        # World Y is the camera-aligned vertical axis. This keeps depth in the arm
        # vector instead of falling back to a 2D angle when hips leave a desk frame.
        down = (0.0, 1.0, 0.0)
    angles = {}
    for side in SIDES:
        shoulder_name, elbow_name = f"{side.upper()}_SHOULDER", f"{side.upper()}_ELBOW"
        if _point(image_landmarks, indices, shoulder_name, min_visibility) is None:
            continue
        if _point(image_landmarks, indices, elbow_name, min_visibility) is None:
            continue
        shoulder = _world_point(world_landmarks, indices, shoulder_name)
        elbow = _world_point(world_landmarks, indices, elbow_name)
        if shoulder is None or elbow is None:
            continue
        elevation = vector_angle(_vector3(shoulder, elbow), down)
        if elevation is not None:
            angles[f"{side}_{ELEVATION_SUFFIX}"] = elevation
    return angles


def _minimum_arm_length(
    landmarks: list[Any],
    indices: dict[str, int],
    min_visibility: float,
    aspect_ratio: float,
) -> float:
    """Scale the noise floor with the person: someone further away is smaller on screen."""
    left = _point(landmarks, indices, "LEFT_SHOULDER", min_visibility)
    right = _point(landmarks, indices, "RIGHT_SHOULDER", min_visibility)
    if left is None or right is None:
        return MIN_ARM_LENGTH
    shoulder_width = _norm(_vector(left, right, aspect_ratio))
    return max(MIN_ARM_LENGTH, shoulder_width * MIN_ARM_LENGTH_PER_SHOULDER_WIDTH)


def torso_down_vector(
    landmarks: list[Any],
    indices: dict[str, int],
    min_visibility: float = 0.55,
    aspect_ratio: float = 1.0,
) -> Vector:
    """Shoulders to hips when the hips are visible, otherwise straight down the image."""
    corners = [
        _point(landmarks, indices, name, min_visibility)
        for name in ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")
    ]
    if any(corner is None for corner in corners):
        return IMAGE_DOWN
    left_shoulder, right_shoulder, left_hip, right_hip = corners
    shoulder_center = _midpoint(left_shoulder, right_shoulder)
    hip_center = _midpoint(left_hip, right_hip)
    down = (
        (hip_center[0] - shoulder_center[0]) * aspect_ratio,
        hip_center[1] - shoulder_center[1],
    )
    return IMAGE_DOWN if _norm(down) < 1e-6 else down


def vector_angle(first: Vector, second: Vector) -> float | None:
    first_length, second_length = _norm(first), _norm(second)
    if first_length < 1e-6 or second_length < 1e-6:
        return None
    if not all(math.isfinite(value) for value in (*first, *second)):
        return None
    cosine = sum(a * b for a, b in zip(first, second)) / (first_length * second_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _point(
    landmarks: list[Any],
    indices: dict[str, int],
    name: str,
    min_visibility: float,
) -> Any | None:
    index = indices.get(name)
    if index is None or not 0 <= index < len(landmarks):
        return None
    landmark = landmarks[index]
    return landmark if is_reliable(landmark, min_visibility) else None


def _vector(start: Any, end: Any, aspect_ratio: float) -> Vector:
    return (
        (float(end.x) - float(start.x)) * aspect_ratio,
        float(end.y) - float(start.y),
    )


def _midpoint(first: Any, second: Any) -> Vector:
    return (
        (float(first.x) + float(second.x)) / 2.0,
        (float(first.y) + float(second.y)) / 2.0,
    )


def _norm(vector: Vector) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _world_point(
    landmarks: list[Any], indices: dict[str, int], name: str
) -> Any | None:
    index = indices.get(name)
    if index is None or not 0 <= index < len(landmarks):
        return None
    point = landmarks[index]
    return (
        point
        if all(math.isfinite(float(getattr(point, axis))) for axis in ("x", "y", "z"))
        else None
    )


def _midpoint3(first: Any, second: Any) -> tuple[float, float, float]:
    return tuple(
        (float(getattr(first, axis)) + float(getattr(second, axis))) / 2.0
        for axis in ("x", "y", "z")
    )


def _vector3(start: Any, end: Any) -> tuple[float, float, float]:
    def coordinate(point: Any, axis: str, index: int) -> float:
        return (
            float(getattr(point, axis)) if hasattr(point, axis) else float(point[index])
        )

    return tuple(
        coordinate(end, axis, index) - coordinate(start, axis, index)
        for index, axis in enumerate(("x", "y", "z"))
    )
