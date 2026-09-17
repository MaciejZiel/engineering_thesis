import math
from itertools import product
from typing import Any

from vision_robot_arm.core.pose_state import LandmarkPoint
from vision_robot_arm.vision.landmarks import is_reliable
from vision_robot_arm.vision.metrics import calculate_angle

HAND_WRIST = 0
HAND_MIDDLE_MCP = 9
FINGER_PIPS = (6, 10, 14, 18)
FINGER_TIPS = (8, 12, 16, 20)
EXTENDED_RATIO = 1.1
OPEN_HAND_MIN_FINGERS = 3
MAX_WRIST_DISTANCE = 0.15
WRIST_DISTANCE_PER_HAND_SPAN = 1.2

HAND_OPEN = "open"
HAND_FIST = "fist"

SIDES = ("left", "right")


def _finger_states(hand: list[Any], aspect_ratio: float) -> tuple[str | None, ...]:
    if len(hand) < 21 or not all(_finite(point) for point in hand):
        return ()
    if _norm(_vector(hand[0], hand[9], aspect_ratio)) < 1e-5:
        return ()
    states = []
    for base in (5, 9, 13, 17):
        mcp, pip, dip, tip = hand[base : base + 4]
        pip_angle = calculate_angle(
            mcp, pip, dip, aspect_ratio=aspect_ratio, use_depth=True
        )
        dip_angle = calculate_angle(
            pip, dip, tip, aspect_ratio=aspect_ratio, use_depth=True
        )
        if pip_angle is None or dip_angle is None:
            states.append(None)
            continue
        tip_distance = _norm(_vector(hand[0], tip, aspect_ratio))
        pip_distance = _norm(_vector(hand[0], pip, aspect_ratio))
        if (
            pip_angle >= 155
            and dip_angle >= 145
            and tip_distance > pip_distance * EXTENDED_RATIO
        ):
            states.append(HAND_OPEN)
        elif pip_angle < 135 or tip_distance < pip_distance * 1.03:
            states.append(HAND_FIST)
        else:
            states.append(None)
    return tuple(states)


def count_extended_fingers(hand: list[Any], aspect_ratio: float = 1.0) -> int:
    return _finger_states(hand, aspect_ratio).count(HAND_OPEN)


def classify_hand(hand: list[Any], aspect_ratio: float = 1.0) -> str | None:
    states = _finger_states(hand, aspect_ratio)
    if states.count(HAND_OPEN) >= OPEN_HAND_MIN_FINGERS:
        return HAND_OPEN
    if len(states) == 4 and states.count(HAND_FIST) >= 3 and HAND_OPEN not in states:
        return HAND_FIST
    return None


def assign_hand_sides(
    hands: list[list[Any]],
    pose_landmarks: list[Any],
    indices: dict[str, int],
    max_distance: float = MAX_WRIST_DISTANCE,
    *,
    aspect_ratio: float = 1.0,
    min_visibility: float = 0.55,
    ambiguity_margin: float = 0.02,
) -> dict[str, list[Any]]:
    return {
        side: hands[index]
        for side, index in assign_hand_indices(
            hands,
            pose_landmarks,
            indices,
            max_distance,
            aspect_ratio=aspect_ratio,
            min_visibility=min_visibility,
            ambiguity_margin=ambiguity_margin,
        ).items()
    }


def assign_hand_indices(
    hands: list[list[Any]],
    pose_landmarks: list[Any],
    indices: dict[str, int],
    max_distance: float = MAX_WRIST_DISTANCE,
    *,
    aspect_ratio: float = 1.0,
    min_visibility: float = 0.55,
    ambiguity_margin: float = 0.02,
) -> dict[str, int]:
    """Match detections once so image and world landmarks keep the same identity."""
    costs = {}
    for hand_number, hand in enumerate(hands):
        if not hand or not _finite(hand[0]):
            continue
        allowed = max_distance
        if len(hand) > HAND_MIDDLE_MCP and _finite(hand[HAND_MIDDLE_MCP]):
            span = _norm(_vector(hand[HAND_WRIST], hand[HAND_MIDDLE_MCP], aspect_ratio))
            # A hand close to the camera is large and its two wrist estimates drift apart.
            allowed = max(max_distance, span * WRIST_DISTANCE_PER_HAND_SPAN)
        for side in SIDES:
            wrist_index = indices.get(f"{side.upper()}_WRIST")
            if wrist_index is None or not 0 <= wrist_index < len(pose_landmarks):
                continue
            wrist = pose_landmarks[wrist_index]
            if not is_reliable(wrist, min_visibility):
                continue
            distance = math.hypot(
                (hand[0].x - wrist.x) * aspect_ratio, hand[0].y - wrist.y
            )
            if distance <= allowed:
                costs[side, hand_number] = distance
    candidates = []
    for assignment in product((None, *range(len(hands))), repeat=2):
        used = [i for i in assignment if i is not None]
        if len(set(used)) != len(used):
            continue
        pairs = [(side, i) for side, i in zip(SIDES, assignment) if i is not None]
        if all(pair in costs for pair in pairs):
            candidates.append((len(pairs), sum(costs[pair] for pair in pairs), pairs))
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    if not candidates:
        return {}
    count, cost, pairs = candidates[0]
    # Crossed/overlapping wrists may have two equally plausible assignments.
    # Do not send a gesture to the wrong gripper on an arbitrary tie-break.
    ambiguous = set()
    for other_count, other_cost, other_pairs in candidates[1:]:
        if other_count != count or other_cost - cost >= ambiguity_margin:
            break
        other = dict(other_pairs)
        ambiguous.update(side for side, i in pairs if other.get(side) != i)
    return {side: i for side, i in pairs if side not in ambiguous}


def anchor_hand_world_landmarks(
    hand_world_by_side: dict[str, list[Any]],
    pose_world_landmarks: list[Any] | None,
    indices: dict[str, int],
) -> dict[str, list[LandmarkPoint]]:
    """Place hand-local metric landmarks at the pose model's metric wrist.

    MediaPipe returns the hand shape in a metric coordinate system centered on the
    hand. Translation is therefore taken from the pose wrist while every hand offset
    remains relative to the hand model's own wrist. This produces one body-relative
    3D scene without mixing normalized image depth with metres.
    """
    if pose_world_landmarks is None:
        return {}
    anchored: dict[str, list[LandmarkPoint]] = {}
    for side, hand in hand_world_by_side.items():
        pose_index = indices.get(f"{side.upper()}_WRIST")
        if (
            pose_index is None
            or not 0 <= pose_index < len(pose_world_landmarks)
            or len(hand) <= HAND_WRIST
            or not _finite(hand[HAND_WRIST])
            or not _finite(pose_world_landmarks[pose_index])
        ):
            continue
        pose_wrist = pose_world_landmarks[pose_index]
        hand_wrist = hand[HAND_WRIST]
        points = []
        for point in hand:
            if not _finite(point):
                points = []
                break
            points.append(
                LandmarkPoint(
                    x=float(pose_wrist.x) + float(point.x) - float(hand_wrist.x),
                    y=float(pose_wrist.y) + float(point.y) - float(hand_wrist.y),
                    z=float(pose_wrist.z) + float(point.z) - float(hand_wrist.z),
                )
            )
        if points:
            anchored[side] = points
    return anchored


def detect_hand_gestures(
    hands: list[list[Any]],
    pose_landmarks: list[Any],
    indices: dict[str, int],
    *,
    aspect_ratio: float = 1.0,
    min_visibility: float = 0.55,
) -> tuple[str, ...]:
    return gestures_from_sides(
        assign_hand_sides(
            hands,
            pose_landmarks,
            indices,
            aspect_ratio=aspect_ratio,
            min_visibility=min_visibility,
        ),
        aspect_ratio=aspect_ratio,
    )


def gestures_from_sides(
    sides: dict[str, list[Any]],
    *,
    aspect_ratio: float = 1.0,
) -> tuple[str, ...]:
    """Classify hands that were already matched to a side, so matching happens once."""
    gestures: list[str] = []
    for side, hand in sides.items():
        shape = classify_hand(hand, aspect_ratio)
        if shape == HAND_OPEN:
            gestures.append(f"{side}_hand_open")
        elif shape == HAND_FIST:
            gestures.append(f"{side}_fist")
    return tuple(sorted(gestures))


Vector = tuple[float, float, float]

DEFAULT_ASPECT_RATIO = 16.0 / 9.0
MAX_WRIST_DEVIATION_DEG = 90.0
STRAIGHT_WRIST_DEG = 180.0
# Shorter than this on screen and the direction of a vector is noise, not a measurement.
MIN_PROJECTION = 0.02
# The two arms are mirror images, so the same bend turns opposite ways on screen.
SIDE_ROTATION = {"left": 1.0, "right": -1.0}
# A wrist cannot cross its whole range in one frame; anything faster is a measurement jump.
MAX_WRIST_RATE_DEG_S = 240.0


def refine_pose_wrists(
    pose_landmarks: list[Any],
    hands_by_side: dict[str, list[Any]],
    indices: dict[str, int],
) -> list[Any]:
    """Move each wrist onto the hand model's wrist, which sits where the hand really starts.

    The pose model places its wrist a little inside the palm, so the forearm was drawn
    past the joint and every angle hinged on the wrong point. Position comes from the
    hand model; visibility stays with the pose model, which is what judges reliability.
    """
    refined = list(pose_landmarks)
    for side, hand in hands_by_side.items():
        index = indices.get(f"{side.upper()}_WRIST")
        if index is None or not 0 <= index < len(refined) or len(hand) <= HAND_WRIST:
            continue
        pose_wrist, hand_wrist = refined[index], hand[HAND_WRIST]
        if not all(
            math.isfinite(float(getattr(hand_wrist, axis))) for axis in ("x", "y")
        ):
            continue
        refined[index] = LandmarkPoint(
            x=float(hand_wrist.x),
            y=float(hand_wrist.y),
            z=float(getattr(hand_wrist, "z", 0.0) or 0.0),
            visibility=float(getattr(pose_wrist, "visibility", 1.0) or 1.0),
        )
    return refined


def hand_wrist_angles(
    hands: list[list[Any]],
    pose_landmarks: list[Any],
    indices: dict[str, int],
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    *,
    min_visibility: float = 0.55,
) -> dict[str, float]:
    """Signed wrist angle per side, 180 = straight, measured in the image plane.

    Both vectors come from image coordinates only. MediaPipe's normalized z is relative
    to a different origin for the pose model than for the hand model, so mixing them
    moved the reported angle by tens of degrees for no real motion.
    """
    return wrist_angles_from_sides(
        assign_hand_sides(
            hands,
            pose_landmarks,
            indices,
            aspect_ratio=aspect_ratio,
            min_visibility=min_visibility,
        ),
        pose_landmarks,
        indices,
        aspect_ratio,
        min_visibility=min_visibility,
    )


def wrist_angles_from_sides(
    sides: dict[str, list[Any]],
    pose_landmarks: list[Any],
    indices: dict[str, int],
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    *,
    min_visibility: float = 0.55,
) -> dict[str, float]:
    """Wrist angles for hands that were already matched to a side."""
    angles: dict[str, float] = {}
    for side, hand in sides.items():
        if len(hand) <= HAND_MIDDLE_MCP:
            continue
        elbow_index = indices.get(f"{side.upper()}_ELBOW")
        wrist_index = indices.get(f"{side.upper()}_WRIST")
        if elbow_index is None or wrist_index is None:
            continue
        if max(elbow_index, wrist_index) >= len(pose_landmarks):
            continue
        if not all(
            is_reliable(pose_landmarks[i], min_visibility)
            for i in (elbow_index, wrist_index)
        ):
            continue
        forearm = _image_vector(
            pose_landmarks[elbow_index], pose_landmarks[wrist_index], aspect_ratio
        )
        hand_direction = _image_vector(
            hand[HAND_WRIST], hand[HAND_MIDDLE_MCP], aspect_ratio
        )
        deviation = signed_wrist_deviation(forearm, hand_direction)
        if deviation is None:
            continue
        angles[f"{side}_wrist"] = STRAIGHT_WRIST_DEG - deviation * SIDE_ROTATION[side]
    return angles


def wrist_angles_3d(
    hand_world_by_side: dict[str, list[Any]],
    pose_landmarks: list[Any],
    pose_world_landmarks: list[Any] | None,
    indices: dict[str, int],
    *,
    min_visibility: float = 0.55,
) -> dict[str, float]:
    """Metric 3D wrist flexion around the hand's index-to-pinky hinge axis."""
    if pose_world_landmarks is None:
        return {}
    angles: dict[str, float] = {}
    for side, hand in hand_world_by_side.items():
        elbow_index = indices.get(f"{side.upper()}_ELBOW")
        wrist_index = indices.get(f"{side.upper()}_WRIST")
        if (
            elbow_index is None
            or wrist_index is None
            or max(elbow_index, wrist_index) >= len(pose_landmarks)
            or max(elbow_index, wrist_index) >= len(pose_world_landmarks)
            or len(hand) <= 17
            or not all(
                is_reliable(pose_landmarks[index], min_visibility)
                for index in (elbow_index, wrist_index)
            )
            or not all(_finite(hand[index]) for index in (0, 5, 9, 17))
        ):
            continue
        forearm = _vector3(
            pose_world_landmarks[elbow_index], pose_world_landmarks[wrist_index]
        )
        hand_direction = _vector3(hand[0], hand[9])
        hinge = _vector3(hand[5], hand[17])
        deviation = signed_angle_around_axis(forearm, hand_direction, hinge)
        if deviation is not None:
            angles[f"{side}_wrist"] = STRAIGHT_WRIST_DEG - deviation
    return angles


def signed_angle_around_axis(
    first: Vector, second: Vector, axis: Vector
) -> float | None:
    """Signed angle after projecting both vectors onto a plane normal to axis."""
    unit_axis = _unit(axis)
    if unit_axis is None:
        return None
    first_projected = _reject(first, unit_axis)
    second_projected = _reject(second, unit_axis)
    first_unit, second_unit = _unit(first_projected), _unit(second_projected)
    if first_unit is None or second_unit is None:
        return None
    sine = _dot(unit_axis, _cross(first_unit, second_unit))
    cosine = _dot(first_unit, second_unit)
    angle = math.degrees(math.atan2(sine, max(-1.0, min(1.0, cosine))))
    return max(-MAX_WRIST_DEVIATION_DEG, min(MAX_WRIST_DEVIATION_DEG, angle))


def signed_wrist_deviation(forearm: Vector, hand_direction: Vector) -> float | None:
    """Rotation from the forearm to the hand in the image plane, positive when bent up.

    The sign comes from the 2D cross product, which stays well conditioned whatever the
    forearm orientation. Taking it from a single vector component collapsed to zero for a
    vertical forearm, so both bend directions reported the same joint angle.
    """
    forearm_x, forearm_y = forearm[0], forearm[1]
    hand_x, hand_y = hand_direction[0], hand_direction[1]
    if not all(
        math.isfinite(value) for value in (forearm_x, forearm_y, hand_x, hand_y)
    ):
        return None
    if math.hypot(forearm_x, forearm_y) < MIN_PROJECTION:
        return None
    if math.hypot(hand_x, hand_y) < MIN_PROJECTION:
        # The hand points at the camera; its projected direction is landmark noise.
        return None
    cross = forearm_x * hand_y - forearm_y * hand_x
    dot = forearm_x * hand_x + forearm_y * hand_y
    deviation = math.degrees(math.atan2(-cross, dot))
    return max(-MAX_WRIST_DEVIATION_DEG, min(MAX_WRIST_DEVIATION_DEG, deviation))


class WristAngleHold:
    """Hold the last wrist angle briefly, and refuse jumps no wrist could perform.

    A foreshortened hand makes the measured angle flip by more than a hundred degrees
    between frames. Following that puts the robot into a long sweep, so the value is
    allowed to travel at most MAX_WRIST_RATE_DEG_S and catches up over a few frames.
    """

    def __init__(
        self, max_age_ms: int = 500, max_rate_deg_s: float = MAX_WRIST_RATE_DEG_S
    ) -> None:
        self._max_age_ms = max_age_ms
        self._max_rate_deg_s = max_rate_deg_s
        self._last: dict[str, tuple[float, int]] = {}
        self._timestamp_ms: int | None = None
        self.held_names: frozenset[str] = frozenset()

    def update(self, angles: dict[str, float], timestamp_ms: int) -> dict[str, float]:
        if self._timestamp_ms is not None and timestamp_ms <= self._timestamp_ms:
            self.reset()
        self._timestamp_ms = timestamp_ms
        for name, value in angles.items():
            if math.isfinite(value):
                self._last[name] = (
                    self._rate_limited(name, value, timestamp_ms),
                    timestamp_ms,
                )
        merged: dict[str, float] = {}
        for name, (value, seen_at) in list(self._last.items()):
            if timestamp_ms - seen_at <= self._max_age_ms:
                merged[name] = value
            else:
                del self._last[name]
        self.held_names = frozenset(merged) - frozenset(angles)
        return merged

    def _rate_limited(self, name: str, value: float, timestamp_ms: int) -> float:
        previous = self._last.get(name)
        if previous is None:
            return value
        elapsed_ms = timestamp_ms - previous[1]
        if elapsed_ms <= 0:
            return previous[0]
        allowed = self._max_rate_deg_s * elapsed_ms / 1000.0
        return max(previous[0] - allowed, min(previous[0] + allowed, value))

    def reset(self) -> None:
        self._last.clear()
        self._timestamp_ms = None
        self.held_names = frozenset()


class HandGestureFilter:
    """Confirm commands for 120 ms / at least two observations, independently per hand."""

    def __init__(self, settle_ms: int = 120, max_gap_ms: int = 250) -> None:
        self.settle_ms = settle_ms
        self.max_gap_ms = max_gap_ms
        self._pending: dict[str, tuple[str, int, int]] = {}
        self._stable: dict[str, str] = {}
        self._last_timestamp: int | None = None
        self._last_interval: int | None = None

    def update(self, gestures: tuple[str, ...], timestamp_ms: int) -> tuple[str, ...]:
        if self._last_timestamp is not None:
            gap = timestamp_ms - self._last_timestamp
            # A slow but steady camera is not a dropout; judge the gap against the
            # rate this camera actually delivers, not against a fixed 30 fps.
            allowed = max(self.max_gap_ms, 2.5 * (self._last_interval or gap))
            if gap <= 0 or gap > allowed:
                self.reset()
            else:
                self._last_interval = gap
        self._last_timestamp = timestamp_ms
        output = []
        for side in SIDES:
            observed = [
                g for g in gestures if g in (f"{side}_hand_open", f"{side}_fist")
            ]
            if len(observed) != 1:
                self._pending.pop(side, None)
                self._stable.pop(side, None)
                continue
            gesture = observed[0]
            previous = self._pending.get(side)
            if previous is None or previous[0] != gesture:
                self._pending[side] = (gesture, timestamp_ms, 1)
            else:
                self._pending[side] = (gesture, previous[1], previous[2] + 1)
                if (
                    timestamp_ms - previous[1] >= self.settle_ms
                    and previous[2] + 1 >= 2
                ):
                    self._stable[side] = gesture
            # Suppress an unconfirmed transition, rather than repeat an obsolete command.
            if self._stable.get(side) == gesture:
                output.append(gesture)
        return tuple(sorted(output))

    def reset(self) -> None:
        self._pending.clear()
        self._stable.clear()
        self._last_timestamp = None
        self._last_interval = None


def _finite(point: Any) -> bool:
    return all(
        math.isfinite(float(getattr(point, axis, 0.0))) for axis in ("x", "y", "z")
    )


def _image_vector(start: Any, end: Any, aspect_ratio: float) -> Vector:
    """Image-plane vector with x scaled so a pixel means the same in both directions."""
    return (
        (float(end.x) - float(start.x)) * aspect_ratio,
        float(end.y) - float(start.y),
        0.0,
    )


def _vector(start: Any, end: Any, aspect_ratio: float) -> Vector:
    return (
        (float(end.x) - float(start.x)) * aspect_ratio,
        float(end.y) - float(start.y),
        (float(getattr(end, "z", 0.0)) - float(getattr(start, "z", 0.0)))
        * aspect_ratio,
    )


def _vector3(start: Any, end: Any) -> Vector:
    return (
        float(end.x) - float(start.x),
        float(end.y) - float(start.y),
        float(end.z) - float(start.z),
    )


def _dot(first: Vector, second: Vector) -> float:
    return sum(a * b for a, b in zip(first, second))


def _cross(first: Vector, second: Vector) -> Vector:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _unit(vector: Vector) -> Vector | None:
    length = _norm(vector)
    if not math.isfinite(length) or length < 1e-8:
        return None
    return tuple(component / length for component in vector)


def _reject(vector: Vector, axis: Vector) -> Vector:
    parallel = _dot(vector, axis)
    return tuple(
        component - parallel * axis_component
        for component, axis_component in zip(vector, axis)
    )


def _norm(vector: Vector) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _distance(a: Any, b: Any) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))
