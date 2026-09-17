"""Map tracked human arms into conservative, mirrored UR workspace targets."""

from __future__ import annotations

import math

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.targets import ARM_LEFT, ARM_RIGHT

Point3 = tuple[float, float, float]

BASE_SEPARATION_M = 0.5
ROBOT_BASES: dict[str, Point3] = {
    ARM_LEFT: (-BASE_SEPARATION_M / 2.0, 0.0, 0.0),
    ARM_RIGHT: (BASE_SEPARATION_M / 2.0, 0.0, 0.0),
}

# The shoulder-to-wrist vector controls direction. This makes the mapping immune
# to the operator leaning or moving as a whole. Human depth is deliberately
# negated: reaching toward the camera moves the cobot TCP toward the operator.
LATERAL_SCALE = 0.75
DEPTH_SCALE = 0.65
VERTICAL_SCALE = 0.75
REST_DEPTH_M = -0.34
REST_HEIGHT_M = 0.62
MIN_POINT_VISIBILITY = 0.5


def tracked_tcp_target(state: PoseState, side: str) -> Point3 | None:
    """Return scene-space TCP target with X/Z preserved and depth mirrored."""
    base = ROBOT_BASES.get(side)
    shoulder = state.body_points.get(f"{side}_shoulder")
    wrist = state.body_points.get(f"{side}_wrist")
    if base is None or not _reliable(shoulder) or not _reliable(wrist):
        return None
    assert shoulder is not None and wrist is not None
    arm = (
        wrist.x - shoulder.x,
        wrist.y - shoulder.y,
        wrist.z - shoulder.z,
    )
    target = (
        base[0] + LATERAL_SCALE * arm[0],
        REST_DEPTH_M - DEPTH_SCALE * arm[1],
        REST_HEIGHT_M + VERTICAL_SCALE * arm[2],
    )
    return (
        _clamp(target[0], base[0] - 0.55, base[0] + 0.55),
        _clamp(target[1], -0.72, 0.18),
        _clamp(target[2], 0.18, 1.08),
    )


def _reliable(point: object | None) -> bool:
    return point is not None and getattr(point, "visibility", 0.0) >= MIN_POINT_VISIBILITY and all(
        math.isfinite(float(getattr(point, axis))) for axis in ("x", "y", "z")
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
