import math
from typing import Any

from vision_robot_arm.vision.landmarks import is_reliable


AngleDefinitions = dict[str, tuple[str, str, str]]


ANGLE_DEFINITIONS: AngleDefinitions = {
    "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_shoulder": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "right_shoulder": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "left_wrist": ("LEFT_ELBOW", "LEFT_WRIST", "LEFT_INDEX"),
    "right_wrist": ("RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_INDEX"),
    "left_hip": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "right_hip": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "left_knee": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "left_ankle": ("LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    "right_ankle": ("RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
}


def calculate_angle(
    a: Any, b: Any, c: Any, *, aspect_ratio: float = 1.0, use_depth: bool = False
) -> float | None:
    """Angle in a consistent coordinate space; coincident points are not a joint."""
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        return None
    def vector(point: Any) -> tuple[float, float, float]:
        return ((point.x-b.x)*aspect_ratio, point.y-b.y,
                (point.z-b.z)*aspect_ratio if use_depth else 0.0)
    first, second = vector(a), vector(c)
    if not all(math.isfinite(v) for v in (*first, *second)):
        return None
    lengths = math.hypot(*first) * math.hypot(*second)
    if lengths < 1e-10:
        return None
    cosine = sum(a*b for a, b in zip(first, second)) / lengths
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def calculate_angles(
    landmarks: list[Any],
    indices: dict[str, int],
    min_visibility: float,
    angle_definitions: AngleDefinitions = ANGLE_DEFINITIONS,
    *,
    aspect_ratio: float = 1.0,
    world_landmarks: list[Any] | None = None,
) -> dict[str, float | None]:
    angles: dict[str, float | None] = {}
    for name, (first, middle, third) in angle_definitions.items():
        keys = (indices.get(first), indices.get(middle), indices.get(third))
        if any(index is None or index < 0 or index >= len(landmarks) for index in keys):
            angles[name] = None
            continue
        first_landmark, middle_landmark, third_landmark = (landmarks[index] for index in keys)
        if not (
            is_reliable(first_landmark, min_visibility)
            and is_reliable(middle_landmark, min_visibility)
            and is_reliable(third_landmark, min_visibility)
        ):
            angles[name] = None
            continue
        if world_landmarks is not None:
            # World points are metres, not normalized image coordinates. Still
            # gate them on current IMAGE visibility; an occluded 3D guess is not a measurement.
            if any(index >= len(world_landmarks) for index in keys):
                angles[name] = None
                continue
            points = [world_landmarks[index] for index in keys]
            angles[name] = calculate_angle(*points, use_depth=True)
        else:
            angles[name] = calculate_angle(first_landmark, middle_landmark, third_landmark,
                                           aspect_ratio=aspect_ratio)
    return angles


def format_angles(angles: dict[str, float | None]) -> str:
    parts = []
    for name, value in angles.items():
        if value is None:
            parts.append(f"{name}=n/a")
        else:
            parts.append(f"{name}={value:5.1f}")
    return " ".join(parts)
