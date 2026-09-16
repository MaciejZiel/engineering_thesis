import math
from typing import Any

HAND_WRIST = 0
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


def _distance(a: Any, b: Any) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))
