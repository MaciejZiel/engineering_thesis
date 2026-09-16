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
IMAGE_DOWN = (0.0, 1.0)

Vector = tuple[float, float]


def arm_elevation_angles(
    landmarks: list[Any],
    indices: dict[str, int],
    min_visibility: float = 0.55,
    aspect_ratio: float = 1.0,
) -> dict[str, float]:
    down = torso_down_vector(landmarks, indices, min_visibility, aspect_ratio)
    angles: dict[str, float] = {}
    for side in SIDES:
        shoulder = _point(landmarks, indices, f"{side.upper()}_SHOULDER", min_visibility)
        elbow = _point(landmarks, indices, f"{side.upper()}_ELBOW", min_visibility)
        if shoulder is None or elbow is None:
            continue
        elevation = vector_angle(_vector(shoulder, elbow, aspect_ratio), down)
        if elevation is not None:
            angles[f"{side}_{ELEVATION_SUFFIX}"] = elevation
    return angles


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
    down = ((hip_center[0] - shoulder_center[0]) * aspect_ratio, hip_center[1] - shoulder_center[1])
    return IMAGE_DOWN if _norm(down) < 1e-6 else down


def vector_angle(first: Vector, second: Vector) -> float | None:
    first_length, second_length = _norm(first), _norm(second)
    if first_length < 1e-6 or second_length < 1e-6:
        return None
    if not all(math.isfinite(value) for value in (*first, *second)):
        return None
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)
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
    return ((float(end.x) - float(start.x)) * aspect_ratio, float(end.y) - float(start.y))


def _midpoint(first: Any, second: Any) -> Vector:
    return ((float(first.x) + float(second.x)) / 2.0, (float(first.y) + float(second.y)) / 2.0)


def _norm(vector: Vector) -> float:
    return math.hypot(vector[0], vector[1])
