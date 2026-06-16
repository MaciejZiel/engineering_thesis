from typing import Any

from vision_robot_arm.landmarks import is_reliable


def detect_gestures(
    landmarks: list[Any],
    angles: dict[str, float | None],
    indices: dict[str, int],
    min_visibility: float,
) -> tuple[str, ...]:
    gestures: list[str] = []

    left_hand_up = _hand_above_shoulder(
        landmarks,
        indices,
        "LEFT",
        min_visibility,
    )
    right_hand_up = _hand_above_shoulder(
        landmarks,
        indices,
        "RIGHT",
        min_visibility,
    )

    if left_hand_up:
        gestures.append("left_hand_up")
    if right_hand_up:
        gestures.append("right_hand_up")
    if left_hand_up and right_hand_up:
        gestures.append("both_hands_up")

    if _arm_extended_side(landmarks, indices, "LEFT", min_visibility):
        gestures.append("left_arm_side")
    if _arm_extended_side(landmarks, indices, "RIGHT", min_visibility):
        gestures.append("right_arm_side")

    if _angle_below(angles, "left_elbow", 100.0):
        gestures.append("left_elbow_bent")
    if _angle_below(angles, "right_elbow", 100.0):
        gestures.append("right_elbow_bent")

    return tuple(gestures)


def _hand_above_shoulder(
    landmarks: list[Any],
    indices: dict[str, int],
    side: str,
    min_visibility: float,
) -> bool:
    wrist = landmarks[indices[f"{side}_WRIST"]]
    shoulder = landmarks[indices[f"{side}_SHOULDER"]]
    if not (
        is_reliable(wrist, min_visibility)
        and is_reliable(shoulder, min_visibility)
    ):
        return False
    return wrist.y < shoulder.y - 0.08


def _arm_extended_side(
    landmarks: list[Any],
    indices: dict[str, int],
    side: str,
    min_visibility: float,
) -> bool:
    wrist = landmarks[indices[f"{side}_WRIST"]]
    shoulder = landmarks[indices[f"{side}_SHOULDER"]]
    elbow = landmarks[indices[f"{side}_ELBOW"]]
    if not (
        is_reliable(wrist, min_visibility)
        and is_reliable(shoulder, min_visibility)
        and is_reliable(elbow, min_visibility)
    ):
        return False

    close_to_shoulder_height = abs(wrist.y - shoulder.y) < 0.18
    outside_shoulder = (
        wrist.x < shoulder.x - 0.18
        if side == "LEFT"
        else wrist.x > shoulder.x + 0.18
    )
    elbow_between = (
        wrist.x < elbow.x < shoulder.x
        if side == "LEFT"
        else shoulder.x < elbow.x < wrist.x
    )
    return close_to_shoulder_height and outside_shoulder and elbow_between


def _angle_below(
    angles: dict[str, float | None],
    name: str,
    threshold: float,
) -> bool:
    value = angles.get(name)
    return value is not None and value < threshold
