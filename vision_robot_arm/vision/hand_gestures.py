import math
from itertools import product
from typing import Any

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
        mcp, pip, dip, tip = hand[base:base+4]
        pip_angle = calculate_angle(mcp, pip, dip, aspect_ratio=aspect_ratio, use_depth=True)
        dip_angle = calculate_angle(pip, dip, tip, aspect_ratio=aspect_ratio, use_depth=True)
        if pip_angle is None or dip_angle is None:
            states.append(None)
            continue
        tip_distance = _norm(_vector(hand[0], tip, aspect_ratio))
        pip_distance = _norm(_vector(hand[0], pip, aspect_ratio))
        if pip_angle >= 155 and dip_angle >= 145 and tip_distance > pip_distance * EXTENDED_RATIO:
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
            distance = math.hypot((hand[0].x-wrist.x)*aspect_ratio, hand[0].y-wrist.y)
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
        if other_count != count or other_cost-cost >= ambiguity_margin:
            break
        other = dict(other_pairs)
        ambiguous.update(side for side, i in pairs if other.get(side) != i)
    return {side: hands[i] for side, i in pairs if side not in ambiguous}


def detect_hand_gestures(
    hands: list[list[Any]],
    pose_landmarks: list[Any],
    indices: dict[str, int],
    *,
    aspect_ratio: float = 1.0,
    min_visibility: float = 0.55,
) -> tuple[str, ...]:
    gestures: list[str] = []
    for side, hand in assign_hand_sides(hands, pose_landmarks, indices,
                                       aspect_ratio=aspect_ratio, min_visibility=min_visibility).items():
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
    angles: dict[str, float] = {}
    for side, hand in assign_hand_sides(hands, pose_landmarks, indices,
                                       aspect_ratio=aspect_ratio, min_visibility=min_visibility).items():
        if len(hand) <= HAND_MIDDLE_MCP:
            continue
        elbow_index = indices.get(f"{side.upper()}_ELBOW")
        wrist_index = indices.get(f"{side.upper()}_WRIST")
        if elbow_index is None or wrist_index is None:
            continue
        if max(elbow_index, wrist_index) >= len(pose_landmarks):
            continue
        if not all(is_reliable(pose_landmarks[i], min_visibility) for i in (elbow_index, wrist_index)):
            continue
        forearm = _image_vector(pose_landmarks[elbow_index], pose_landmarks[wrist_index], aspect_ratio)
        hand_direction = _image_vector(hand[HAND_WRIST], hand[HAND_MIDDLE_MCP], aspect_ratio)
        deviation = signed_wrist_deviation(forearm, hand_direction)
        if deviation is None:
            continue
        angles[f"{side}_wrist"] = STRAIGHT_WRIST_DEG - deviation
    return angles


def signed_wrist_deviation(forearm: Vector, hand_direction: Vector) -> float | None:
    """Rotation from the forearm to the hand in the image plane, positive when bent up.

    The sign comes from the 2D cross product, which stays well conditioned whatever the
    forearm orientation. Taking it from a single vector component collapsed to zero for a
    vertical forearm, so both bend directions reported the same joint angle.
    """
    forearm_x, forearm_y = forearm[0], forearm[1]
    hand_x, hand_y = hand_direction[0], hand_direction[1]
    if not all(math.isfinite(value) for value in (forearm_x, forearm_y, hand_x, hand_y)):
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
    """Keep the last wrist angle for a short time when the hand tracker drops a frame."""

    def __init__(self, max_age_ms: int = 500) -> None:
        self._max_age_ms = max_age_ms
        self._last: dict[str, tuple[float, int]] = {}
        self._timestamp_ms: int | None = None

    def update(self, angles: dict[str, float], timestamp_ms: int) -> dict[str, float]:
        if self._timestamp_ms is not None and timestamp_ms <= self._timestamp_ms:
            self.reset()
        self._timestamp_ms = timestamp_ms
        for name, value in angles.items():
            if math.isfinite(value):
                self._last[name] = (value, timestamp_ms)
        merged: dict[str, float] = {}
        for name, (value, seen_at) in list(self._last.items()):
            if timestamp_ms - seen_at <= self._max_age_ms:
                merged[name] = value
            else:
                del self._last[name]
        return merged

    def reset(self) -> None:
        self._last.clear()
        self._timestamp_ms = None


class HandGestureFilter:
    """Confirm commands for 120 ms / at least two observations, independently per hand."""

    def __init__(self, settle_ms: int = 120, max_gap_ms: int = 250) -> None:
        self.settle_ms = settle_ms
        self.max_gap_ms = max_gap_ms
        self._pending: dict[str, tuple[str, int, int]] = {}
        self._stable: dict[str, str] = {}
        self._last_timestamp: int | None = None

    def update(self, gestures: tuple[str, ...], timestamp_ms: int) -> tuple[str, ...]:
        if self._last_timestamp is not None and (
            timestamp_ms <= self._last_timestamp or timestamp_ms-self._last_timestamp > self.max_gap_ms
        ):
            self.reset()
        self._last_timestamp = timestamp_ms
        output = []
        for side in SIDES:
            observed = [g for g in gestures if g in (f"{side}_hand_open", f"{side}_fist")]
            if len(observed) != 1:
                self._pending.pop(side, None)
                self._stable.pop(side, None)
                continue
            gesture = observed[0]
            previous = self._pending.get(side)
            if previous is None or previous[0] != gesture:
                self._pending[side] = (gesture, timestamp_ms, 1)
            else:
                self._pending[side] = (gesture, previous[1], previous[2]+1)
                if timestamp_ms-previous[1] >= self.settle_ms and previous[2]+1 >= 2:
                    self._stable[side] = gesture
            # Suppress an unconfirmed transition, rather than repeat an obsolete command.
            if self._stable.get(side) == gesture:
                output.append(gesture)
        return tuple(sorted(output))

    def reset(self) -> None:
        self._pending.clear()
        self._stable.clear()
        self._last_timestamp = None


def _finite(point: Any) -> bool:
    return all(math.isfinite(float(getattr(point, axis, 0.0))) for axis in ("x", "y", "z"))


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
        (float(getattr(end, "z", 0.0)) - float(getattr(start, "z", 0.0))) * aspect_ratio,
    )


def _norm(vector: Vector) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _distance(a: Any, b: Any) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))
