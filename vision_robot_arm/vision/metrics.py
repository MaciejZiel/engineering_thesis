import math
from typing import Any

from vision_robot_arm.vision.landmarks import is_reliable


AngleDefinitions = dict[str, tuple[str, str, str]]


ANGLE_DEFINITIONS: AngleDefinitions = {
    "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_shoulder": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "right_shoulder": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "left_hip": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "right_hip": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "left_knee": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "left_ankle": ("LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    "right_ankle": ("RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
}


def calculate_angle(a: Any, b: Any, c: Any) -> float:
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(math.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def calculate_angles(
    landmarks: list[Any],
    indices: dict[str, int],
    min_visibility: float,
    angle_definitions: AngleDefinitions = ANGLE_DEFINITIONS,
) -> dict[str, float | None]:
    angles: dict[str, float | None] = {}
    for name, (first, middle, third) in angle_definitions.items():
        first_landmark = landmarks[indices[first]]
        middle_landmark = landmarks[indices[middle]]
        third_landmark = landmarks[indices[third]]
        if not (
            is_reliable(first_landmark, min_visibility)
            and is_reliable(middle_landmark, min_visibility)
            and is_reliable(third_landmark, min_visibility)
        ):
            angles[name] = None
            continue
        angles[name] = calculate_angle(first_landmark, middle_landmark, third_landmark)
    return angles


def format_angles(angles: dict[str, float | None]) -> str:
    parts = []
    for name, value in angles.items():
        if value is None:
            parts.append(f"{name}=n/a")
        else:
            parts.append(f"{name}={value:5.1f}")
    return " ".join(parts)
