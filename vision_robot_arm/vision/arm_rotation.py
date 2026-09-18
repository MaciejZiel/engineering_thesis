"""Rotation angles of the arm that the three pitch joints cannot express.

The UR7e's shoulder, elbow and wrist 1 share parallel horizontal axes (UR DH:
alpha 0 for joints 2 and 3), so the angles in `arm_pose.py` and `metrics.py`
only ever move the robot up and down in one vertical plane. The other three
joints are rotations: the base turns the whole arm about the vertical, wrist 2
yaws the tool and wrist 3 rolls it. This module measures their human
counterparts in the metric body frame from `body_tracking.py`:

* ``shoulder_azimuth``  horizontal swing of the upper arm       -> base
* ``wrist_deviation``   radial/ulnar bend of the hand            -> wrist 2
* ``forearm_roll``      pronation/supination of the forearm     -> wrist 3

All three need 3D world landmarks. There is no image-plane substitute, which is
why the 2D tracking space leaves the rotation joints held. Angles are anatomical:
the same motion of the left and the right arm gives the same sign, so the
operator has one mental model; mounting differences belong to the robot config.
Each angle is kept continuous across +/-180 while the joint stays visible, the
way `planar_tracking.py` does, because a low-pass filter must never see a jump
from 179 to -179.
"""

from __future__ import annotations

import math
from typing import Any

from vision_robot_arm.vision.body_tracking import (
    build_body_frame,
    transform_hands_to_body,
    transform_pose_to_body,
)

Vector3 = tuple[float, float, float]

SIDES = ("left", "right")
AZIMUTH_SUFFIX = "shoulder_azimuth"
DEVIATION_SUFFIX = "wrist_deviation"
ROLL_SUFFIX = "forearm_roll"
ROTATION_ANGLE_SUFFIXES = (AZIMUTH_SUFFIX, DEVIATION_SUFFIX, ROLL_SUFFIX)

# MediaPipe hand topology.
HAND_WRIST = 0
HAND_INDEX_MCP = 5
HAND_MIDDLE_MCP = 9
HAND_PINKY_MCP = 17
HAND_POINTS_REQUIRED = 18

# An upper arm hanging down has no meaningful azimuth: its horizontal shadow is a
# few centimetres of landmark noise. Below this share of the arm's length the
# joint is left unmeasured rather than spun by noise.
MIN_HORIZONTAL_FRACTION = 0.3
MIN_FOREARM_LENGTH_M = 0.05
MIN_PALM_AREA_M2 = 1e-4
MIN_AXIS_LENGTH = 1e-6


class ArmRotationTracker:
    """Measure the three rotation angles per arm and keep them continuous."""

    def __init__(self) -> None:
        self._previous: dict[str, float] = {}

    def reset(self) -> None:
        self._previous.clear()

    def measure(
        self,
        image_landmarks: list[Any],
        world_landmarks: list[Any] | None,
        hand_world_landmarks: dict[str, list[Any]],
        indices: dict[str, int],
        min_visibility: float,
    ) -> dict[str, float]:
        if world_landmarks is None:
            return {}
        frame = build_body_frame(image_landmarks, world_landmarks, indices, min_visibility)
        if frame is None:
            return {}
        body = transform_pose_to_body(frame, image_landmarks, world_landmarks)
        hands = transform_hands_to_body(frame, hand_world_landmarks)

        result: dict[str, float] = {}
        for side in SIDES:
            shoulder = _point(body, indices, f"{side.upper()}_SHOULDER", min_visibility)
            elbow = _point(body, indices, f"{side.upper()}_ELBOW", min_visibility)
            wrist = _point(body, indices, f"{side.upper()}_WRIST", min_visibility)

            if shoulder is not None and elbow is not None:
                azimuth = shoulder_azimuth(shoulder, elbow, side)
                if azimuth is not None:
                    result[f"{side}_{AZIMUTH_SUFFIX}"] = self._continuous(
                        f"{side}_{AZIMUTH_SUFFIX}", azimuth
                    )

            hand = hands.get(side)
            if elbow is None or wrist is None or hand is None or len(hand) < HAND_POINTS_REQUIRED:
                continue
            forearm = _sub(_xyz(wrist), _xyz(elbow))
            if _norm(forearm) < MIN_FOREARM_LENGTH_M:
                continue
            normal = palm_normal(hand, side)
            if normal is None:
                continue
            roll = forearm_roll(forearm, normal, side)
            if roll is not None:
                result[f"{side}_{ROLL_SUFFIX}"] = self._continuous(f"{side}_{ROLL_SUFFIX}", roll)
            deviation = wrist_deviation(forearm, hand, normal)
            if deviation is not None:
                result[f"{side}_{DEVIATION_SUFFIX}"] = self._continuous(
                    f"{side}_{DEVIATION_SUFFIX}", deviation
                )
        return result

    def _continuous(self, name: str, angle: float) -> float:
        previous = self._previous.get(name, angle)
        angle = previous + (angle - previous + 180.0) % 360.0 - 180.0
        self._previous[name] = angle
        return angle


def shoulder_azimuth(shoulder: Any, elbow: Any, side: str) -> float | None:
    """Horizontal swing of the upper arm: 0 = out to the side (T-pose), +90 = forward."""
    arm = _sub(_xyz(elbow), _xyz(shoulder))
    length = _norm(arm)
    horizontal = (arm[0], arm[1])
    if length < MIN_AXIS_LENGTH or math.hypot(*horizontal) < MIN_HORIZONTAL_FRACTION * length:
        return None
    lateral = arm[0] if side == "right" else -arm[0]
    return math.degrees(math.atan2(arm[1], lateral))


def palm_normal(hand: Any, side: str) -> Vector3 | None:
    """Unit normal leaving the back of the hand, the same way for both hands."""
    wrist = _xyz(hand[HAND_WRIST])
    index = _sub(_xyz(hand[HAND_INDEX_MCP]), wrist)
    pinky = _sub(_xyz(hand[HAND_PINKY_MCP]), wrist)
    normal = _cross(index, pinky)
    if _norm(normal) < MIN_PALM_AREA_M2:
        return None
    # MediaPipe numbers both hands identically, so this cross product points
    # out of the back of a right hand and into the palm of a left one.
    if side == "left":
        normal = _scale(normal, -1.0)
    return _unit(normal)


def forearm_roll(forearm: Vector3, normal: Vector3, side: str) -> float | None:
    """Rotation of the palm about the forearm axis: 0 = palm down, pronation positive."""
    axis = _unit(forearm)
    if axis is None:
        return None
    reference = _unit(_reject((0.0, 0.0, -1.0), axis))
    if reference is None:
        # Forearm vertical: measure against the body's forward direction instead.
        reference = _unit(_reject((0.0, 1.0, 0.0), axis))
        if reference is None:
            return None
    second = _cross(axis, reference)
    projected = _reject(normal, axis)
    if _norm(projected) < MIN_AXIS_LENGTH:
        return None
    roll = math.degrees(math.atan2(_dot(projected, second), _dot(projected, reference)))
    # Mirror-symmetric motion of the two arms rotates in opposite directions in
    # a fixed frame; flip the left side so pronation reads the same on both.
    return -roll if side == "left" else roll


def wrist_deviation(forearm: Vector3, hand: Any, normal: Vector3) -> float | None:
    """Bend of the hand within the palm plane: toward the index finger positive."""
    wrist = _xyz(hand[HAND_WRIST])
    direction = _sub(_xyz(hand[HAND_MIDDLE_MCP]), wrist)
    toward_index = _sub(_xyz(hand[HAND_INDEX_MCP]), _xyz(hand[HAND_PINKY_MCP]))
    forearm_in_plane = _unit(_reject(forearm, normal))
    hand_in_plane = _unit(_reject(direction, normal))
    if forearm_in_plane is None or hand_in_plane is None:
        return None
    turn = _cross(forearm_in_plane, hand_in_plane)
    angle = math.degrees(math.atan2(_dot(turn, normal), _dot(forearm_in_plane, hand_in_plane)))
    # Positive toward the index-finger side, which is the radial side of either hand.
    index_turn = _dot(_cross(forearm_in_plane, _reject(toward_index, normal)), normal)
    if abs(index_turn) < MIN_AXIS_LENGTH:
        return None
    return angle if index_turn > 0 else -angle


def _point(
    points: list[Any], indices: dict[str, int], name: str, min_visibility: float
) -> Any | None:
    """A visible, finite body-frame point.

    These are metres, not normalized image coordinates, so the image-range test
    in `landmarks.is_reliable` must not be applied: a left wrist sits at
    x = -0.7 m on every frame. Visibility was copied from the image model by
    `transform_pose_to_body`, so it still says whether the joint was seen.
    """
    index = indices.get(name)
    if index is None or not 0 <= index < len(points):
        return None
    point = points[index]
    visibility = float(getattr(point, "visibility", 1.0))
    if not math.isfinite(visibility) or visibility < min_visibility:
        return None
    if not all(math.isfinite(float(getattr(point, axis))) for axis in ("x", "y", "z")):
        return None
    return point


def _xyz(point: Any) -> Vector3:
    return (float(point.x), float(point.y), float(point.z))


def _sub(first: Vector3, second: Vector3) -> Vector3:
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _dot(first: Vector3, second: Vector3) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _reject(vector: Vector3, axis: Vector3) -> Vector3:
    return _sub(vector, _scale(axis, _dot(vector, axis)))


def _unit(vector: Vector3) -> Vector3 | None:
    length = _norm(vector)
    if not math.isfinite(length) or length < MIN_AXIS_LENGTH:
        return None
    return _scale(vector, 1.0 / length)
