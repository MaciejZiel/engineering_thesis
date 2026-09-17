"""Transform MediaPipe world landmarks into a stable, person-relative XYZ frame."""

from __future__ import annotations

import math
from typing import Any, Iterable

from vision_robot_arm.core.pose_state import BodyFrame3D, LandmarkPoint
from vision_robot_arm.vision.landmarks import is_reliable

Vector3 = tuple[float, float, float]
MIN_AXIS_LENGTH_M = 1e-4


def build_body_frame(
    image_landmarks: list[Any],
    world_landmarks: list[Any] | None,
    indices: dict[str, int],
    min_visibility: float,
) -> BodyFrame3D | None:
    """Build a right-handed X-right, Y-forward, Z-up frame at shoulder centre.

    Hips provide the preferred vertical reference. When they are outside a desk
    camera's view, camera-up is projected perpendicular to the shoulder line. The
    fallback keeps depth usable without claiming a camera-to-robot calibration.
    """
    if world_landmarks is None:
        return None
    left = _reliable_pair(
        image_landmarks, world_landmarks, indices.get("LEFT_SHOULDER"), min_visibility
    )
    right = _reliable_pair(
        image_landmarks, world_landmarks, indices.get("RIGHT_SHOULDER"), min_visibility
    )
    if left is None or right is None:
        return None

    origin = _scale(_add(left, right), 0.5)
    right_axis = _unit(_subtract(right, left))
    if right_axis is None:
        return None

    left_hip = _reliable_pair(
        image_landmarks, world_landmarks, indices.get("LEFT_HIP"), min_visibility
    )
    right_hip = _reliable_pair(
        image_landmarks, world_landmarks, indices.get("RIGHT_HIP"), min_visibility
    )
    if left_hip is not None and right_hip is not None:
        hip_centre = _scale(_add(left_hip, right_hip), 0.5)
        up_reference = _subtract(origin, hip_centre)
        source = "shoulders_hips"
    else:
        up_reference = (0.0, -1.0, 0.0)
        source = "shoulders_camera_up"

    # Gram-Schmidt removes shoulder tilt from the vertical axis. This prevents an
    # arm moving sideways from leaking into Z and keeps all three axes orthogonal.
    up_axis = _unit(_reject(up_reference, right_axis))
    if up_axis is None and source == "shoulders_hips":
        up_axis = _unit(_reject((0.0, -1.0, 0.0), right_axis))
        source = "shoulders_camera_up"
    if up_axis is None:
        return None
    forward_axis = _unit(_cross(up_axis, right_axis))
    if forward_axis is None:
        return None
    # Recompute up to eliminate accumulated floating-point non-orthogonality.
    up_axis = _unit(_cross(right_axis, forward_axis))
    if up_axis is None:
        return None
    return BodyFrame3D(origin, right_axis, forward_axis, up_axis, source)


def transform_pose_to_body(
    frame: BodyFrame3D,
    image_landmarks: list[Any],
    world_landmarks: list[Any],
) -> list[LandmarkPoint]:
    """Transform every finite pose point and retain image-model visibility."""
    transformed = []
    for index, point in enumerate(world_landmarks):
        visibility = (
            float(getattr(image_landmarks[index], "visibility", 1.0))
            if index < len(image_landmarks)
            else 0.0
        )
        transformed.append(_transform_point(frame, point, visibility))
    return transformed


def transform_hands_to_body(
    frame: BodyFrame3D,
    hands: dict[str, Iterable[Any]],
) -> dict[str, tuple[LandmarkPoint, ...]]:
    transformed: dict[str, tuple[LandmarkPoint, ...]] = {}
    for side, points in hands.items():
        converted = tuple(_transform_point(frame, point, 1.0) for point in points)
        if converted and all(_finite_point(point) for point in converted):
            transformed[side] = converted
    return transformed


def _transform_point(
    frame: BodyFrame3D, point: Any, visibility: float
) -> LandmarkPoint:
    coordinate = _coordinates(point)
    if coordinate is None or not math.isfinite(visibility):
        return LandmarkPoint(float("nan"), float("nan"), float("nan"), 0.0)
    relative = _subtract(coordinate, frame.origin)
    return LandmarkPoint(
        _dot(relative, frame.right),
        _dot(relative, frame.forward),
        _dot(relative, frame.up),
        visibility,
    )


def _reliable_pair(
    image: list[Any], world: list[Any], index: int | None, min_visibility: float
) -> Vector3 | None:
    if index is None or not 0 <= index < len(image) or index >= len(world):
        return None
    if not is_reliable(image[index], min_visibility):
        return None
    return _coordinates(world[index])


def _coordinates(point: Any) -> Vector3 | None:
    try:
        values = tuple(float(getattr(point, axis)) for axis in ("x", "y", "z"))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _finite_point(point: LandmarkPoint) -> bool:
    return all(math.isfinite(value) for value in (point.x, point.y, point.z))


def _add(first: Vector3, second: Vector3) -> Vector3:
    return tuple(a + b for a, b in zip(first, second))


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(first, second))


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return tuple(value * scalar for value in vector)


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(a * b for a, b in zip(first, second))


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _reject(vector: Vector3, axis: Vector3) -> Vector3:
    return _subtract(vector, _scale(axis, _dot(vector, axis)))


def _unit(vector: Vector3) -> Vector3 | None:
    length = math.sqrt(_dot(vector, vector))
    if not math.isfinite(length) or length < MIN_AXIS_LENGTH_M:
        return None
    return _scale(vector, 1.0 / length)
