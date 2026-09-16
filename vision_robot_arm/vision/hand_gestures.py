import math
from typing import Any

HAND_WRIST = 0
HAND_MIDDLE_MCP = 9
FINGER_PIPS = (6, 10, 14, 18)
FINGER_TIPS = (8, 12, 16, 20)
EXTENDED_RATIO = 1.1
OPEN_HAND_MIN_FINGERS = 3
MAX_WRIST_DISTANCE = 0.15

HAND_OPEN = "open"
HAND_FIST = "fist"

SIDES = ("left", "right")


def count_extended_fingers(hand: list[Any]) -> int:
    wrist = hand[HAND_WRIST]
    extended = 0
    for pip_index, tip_index in zip(FINGER_PIPS, FINGER_TIPS):
        pip_distance = _distance(hand[pip_index], wrist)
        tip_distance = _distance(hand[tip_index], wrist)
        if tip_distance > pip_distance * EXTENDED_RATIO:
            extended += 1
    return extended


def classify_hand(hand: list[Any]) -> str | None:
    if len(hand) <= max(FINGER_TIPS):
        return None
    extended = count_extended_fingers(hand)
    if extended >= OPEN_HAND_MIN_FINGERS:
        return HAND_OPEN
    if extended == 0:
        return HAND_FIST
    return None


def assign_hand_sides(
    hands: list[list[Any]],
    pose_landmarks: list[Any],
    indices: dict[str, int],
    max_distance: float = MAX_WRIST_DISTANCE,
) -> dict[str, list[Any]]:
    candidates = []
    for hand_number, hand in enumerate(hands):
        if not hand:
            continue
        for side in SIDES:
            wrist_index = indices.get(f"{side.upper()}_WRIST")
            if wrist_index is None or wrist_index >= len(pose_landmarks):
                continue
            distance = _distance(hand[HAND_WRIST], pose_landmarks[wrist_index])
            if distance <= max_distance:
                candidates.append((distance, side, hand_number))

    assigned: dict[str, list[Any]] = {}
    used_hands: set[int] = set()
    for _, side, hand_number in sorted(candidates, key=lambda item: item[0]):
        if side in assigned or hand_number in used_hands:
            continue
        assigned[side] = hands[hand_number]
        used_hands.add(hand_number)
    return assigned


def detect_hand_gestures(
    hands: list[list[Any]],
    pose_landmarks: list[Any],
    indices: dict[str, int],
) -> tuple[str, ...]:
    gestures: list[str] = []
    for side, hand in assign_hand_sides(hands, pose_landmarks, indices).items():
        shape = classify_hand(hand)
        if shape == HAND_OPEN:
            gestures.append(f"{side}_hand_open")
        elif shape == HAND_FIST:
            gestures.append(f"{side}_fist")
    return tuple(sorted(gestures))


Vector = tuple[float, float, float]

DEFAULT_ASPECT_RATIO = 16.0 / 9.0
MAX_WRIST_DEVIATION_DEG = 90.0
STRAIGHT_WRIST_DEG = 180.0


def hand_wrist_angles(
    hands: list[list[Any]],
    pose_landmarks: list[Any],
    indices: dict[str, int],
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> dict[str, float]:
    """Signed wrist angle per side: 180 = straight, below 180 = bent up, above = bent down."""
    angles: dict[str, float] = {}
    for side, hand in assign_hand_sides(hands, pose_landmarks, indices).items():
        if len(hand) <= HAND_MIDDLE_MCP:
            continue
        elbow_index = indices.get(f"{side.upper()}_ELBOW")
        wrist_index = indices.get(f"{side.upper()}_WRIST")
        if elbow_index is None or wrist_index is None:
            continue
        if max(elbow_index, wrist_index) >= len(pose_landmarks):
            continue
        forearm = _vector(pose_landmarks[elbow_index], pose_landmarks[wrist_index], aspect_ratio)
        hand_direction = _vector(hand[HAND_WRIST], hand[HAND_MIDDLE_MCP], aspect_ratio)
        deviation = signed_wrist_deviation(forearm, hand_direction)
        if deviation is None:
            continue
        angles[f"{side}_wrist"] = STRAIGHT_WRIST_DEG - deviation
    return angles


def signed_wrist_deviation(forearm: Vector, hand_direction: Vector) -> float | None:
    """Angle between forearm and hand in degrees, positive when the hand bends up on screen."""
    forearm_length = _norm(forearm)
    hand_length = _norm(hand_direction)
    if forearm_length < 1e-6 or hand_length < 1e-6:
        return None
    unit_forearm = tuple(component / forearm_length for component in forearm)
    unit_hand = tuple(component / hand_length for component in hand_direction)
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(unit_forearm, unit_hand))))
    magnitude = math.degrees(math.acos(cosine))
    projection = sum(a * b for a, b in zip(unit_hand, unit_forearm))
    perpendicular_y = unit_hand[1] - projection * unit_forearm[1]
    sign = -1.0 if perpendicular_y > 1e-6 else 1.0
    return max(-MAX_WRIST_DEVIATION_DEG, min(MAX_WRIST_DEVIATION_DEG, sign * magnitude))


class WristAngleHold:
    """Keep the last wrist angle for a short time when the hand tracker drops a frame."""

    def __init__(self, max_age_ms: int = 500) -> None:
        self._max_age_ms = max_age_ms
        self._last: dict[str, tuple[float, int]] = {}

    def update(self, angles: dict[str, float], timestamp_ms: int) -> dict[str, float]:
        for name, value in angles.items():
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
