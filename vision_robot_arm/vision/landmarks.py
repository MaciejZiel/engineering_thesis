import math
from typing import Any


def build_landmark_indices(vision: Any) -> dict[str, int]:
    return {landmark.name: landmark.value for landmark in vision.PoseLandmark}


def build_landmark_names(vision: Any) -> dict[int, str]:
    return {
        landmark.value: landmark.name.lower()
        for landmark in vision.PoseLandmark
    }


def is_reliable(landmark: Any, min_visibility: float) -> bool:
    visibility = getattr(landmark, "visibility", 1.0)
    return (
        math.isfinite(visibility)
        and visibility >= min_visibility
        and math.isfinite(landmark.x)
        and math.isfinite(landmark.y)
        and math.isfinite(landmark.z)
        and -0.15 <= landmark.x <= 1.15
        and -0.15 <= landmark.y <= 1.15
    )
