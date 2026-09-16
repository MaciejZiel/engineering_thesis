from typing import Any

from vision_robot_arm.core.hud import draw_label, draw_text_panel, panel_height, scaled
from vision_robot_arm.vision.landmarks import is_reliable
from vision_robot_arm.vision.metrics import ANGLE_DEFINITIONS


Point = tuple[int, int]
Color = tuple[int, int, int]

ARM_JOINT_LABELS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
JOINT_LABEL_COLOR: Color = (112, 169, 238)
HUD_MARGIN = 12
HINT_SCALE = 0.42
HINT_LINE_HEIGHT = 16
HINT_PADDING = 6
HINT_COLOR: Color = (190, 190, 190)
TRACKING_COLOR: Color = (112, 169, 238)


def pixel_point(landmark: Any, width: int, height: int) -> Point:
    return int(landmark.x * width), int(landmark.y * height)


def midpoint(a: Point, b: Point) -> Point:
    return (a[0] + b[0]) // 2, (a[1] + b[1]) // 2


def draw_segment(
    cv2: Any,
    frame: Any,
    start: Point,
    end: Point,
    color: Color,
    thickness: int = 3,
) -> None:
    cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, start, 4, color, -1, cv2.LINE_AA)
    cv2.circle(frame, end, 4, color, -1, cv2.LINE_AA)


def draw_named_segment(
    cv2: Any,
    frame: Any,
    landmarks: list[Any],
    indices: dict[str, int],
    first: str,
    second: str,
    min_visibility: float,
    color: Color,
    thickness: int = 3,
) -> None:
    height, width = frame.shape[:2]
    first_landmark = landmarks[indices[first]]
    second_landmark = landmarks[indices[second]]
    if not (
        is_reliable(first_landmark, min_visibility)
        and is_reliable(second_landmark, min_visibility)
    ):
        return

    draw_segment(
        cv2,
        frame,
        pixel_point(first_landmark, width, height),
        pixel_point(second_landmark, width, height),
        color,
        thickness,
    )


def reliable_point(
    landmarks: list[Any],
    indices: dict[str, int],
    name: str,
    min_visibility: float,
    width: int,
    height: int,
) -> Point | None:
    landmark = landmarks[indices[name]]
    if not is_reliable(landmark, min_visibility):
        return None
    return pixel_point(landmark, width, height)


def draw_joint_angle_labels(
    cv2: Any,
    frame: Any,
    landmarks: list[Any],
    indices: dict[str, int],
    angles: dict[str, float | None],
    min_visibility: float,
    joints: tuple[str, ...] = ARM_JOINT_LABELS,
) -> None:
    height, width = frame.shape[:2]
    for name in joints:
        value = angles.get(name)
        definition = ANGLE_DEFINITIONS.get(name)
        if value is None or definition is None:
            continue
        point = reliable_point(landmarks, indices, definition[1], min_visibility, width, height)
        if point is None:
            continue
        label = f"{joint_label(name)} {value:.0f}"
        offset = scaled(10, frame)
        draw_label(cv2, frame, label, (point[0] + offset, point[1] - offset), JOINT_LABEL_COLOR)


def joint_label(name: str) -> str:
    side, _, joint = name.partition("_")
    return f"{side[:1].upper()} {joint}"


def draw_stick_figure(
    cv2: Any,
    frame: Any,
    landmarks: list[Any],
    indices: dict[str, int],
    min_visibility: float,
) -> None:
    arm_color = (112, 169, 238)
    torso_color = (188, 183, 174)
    hip_color = torso_color
    leg_color = (155, 151, 144)
    head_color = (207, 203, 196)

    height, width = frame.shape[:2]

    draw_named_segment(
        cv2,
        frame,
        landmarks,
        indices,
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        min_visibility,
        torso_color,
    )

    for side in ("LEFT", "RIGHT"):
        draw_named_segment(
            cv2,
            frame,
            landmarks,
            indices,
            f"{side}_SHOULDER",
            f"{side}_ELBOW",
            min_visibility,
            arm_color,
        )
        draw_named_segment(
            cv2,
            frame,
            landmarks,
            indices,
            f"{side}_ELBOW",
            f"{side}_WRIST",
            min_visibility,
            arm_color,
        )

    left_shoulder = reliable_point(
        landmarks, indices, "LEFT_SHOULDER", min_visibility, width, height
    )
    right_shoulder = reliable_point(
        landmarks, indices, "RIGHT_SHOULDER", min_visibility, width, height
    )
    left_hip = reliable_point(
        landmarks, indices, "LEFT_HIP", min_visibility, width, height
    )
    right_hip = reliable_point(
        landmarks, indices, "RIGHT_HIP", min_visibility, width, height
    )

    shoulder_center = None
    hip_center = None
    if left_shoulder and right_shoulder:
        shoulder_center = midpoint(left_shoulder, right_shoulder)
    if left_hip and right_hip:
        hip_y = (left_hip[1] + right_hip[1]) // 2
        horizontal_left_hip = (left_hip[0], hip_y)
        horizontal_right_hip = (right_hip[0], hip_y)
        hip_center = midpoint(horizontal_left_hip, horizontal_right_hip)
        draw_segment(
            cv2,
            frame,
            horizontal_left_hip,
            horizontal_right_hip,
            hip_color,
            thickness=4,
        )

        for side, hip_point in (
            ("LEFT", horizontal_left_hip),
            ("RIGHT", horizontal_right_hip),
        ):
            knee = reliable_point(
                landmarks, indices, f"{side}_KNEE", min_visibility, width, height
            )
            ankle = reliable_point(
                landmarks, indices, f"{side}_ANKLE", min_visibility, width, height
            )
            if knee:
                draw_segment(cv2, frame, hip_point, knee, leg_color)
            if knee and ankle:
                draw_segment(cv2, frame, knee, ankle, leg_color)

    if shoulder_center and hip_center:
        draw_segment(cv2, frame, shoulder_center, hip_center, torso_color)

    nose = reliable_point(landmarks, indices, "NOSE", min_visibility, width, height)
    if nose:
        if shoulder_center and left_shoulder and right_shoulder:
            draw_segment(cv2, frame, nose, shoulder_center, head_color, thickness=2)
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            radius = max(8, min(34, int(shoulder_width * 0.18)))
        else:
            radius = 12
        cv2.circle(frame, nose, radius, head_color, 2, cv2.LINE_AA)


HAND_PALM_OUTLINE = (0, 5, 9, 13, 17, 0)
HAND_FINGER_CHAINS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20), (0, 1, 2, 3, 4))
HAND_MIDDLE_MCP = 9
HAND_COLOR: Color = (162, 200, 137)
HAND_LINK_COLOR: Color = (112, 169, 238)


def draw_hands(
    cv2: Any,
    frame: Any,
    hands_by_side: dict[str, list[Any]],
    landmarks: list[Any],
    indices: dict[str, int],
    min_visibility: float,
) -> None:
    """Draw the tracked hands and the wrist-to-palm link that defines the wrist angle."""
    height, width = frame.shape[:2]
    thin = max(1, round(2 * height / 1080))
    thick = max(2, round(4 * height / 1080))
    for side, hand in hands_by_side.items():
        if len(hand) < 21:
            continue
        points = [pixel_point(point, width, height) for point in hand]
        for chain in (HAND_PALM_OUTLINE, *HAND_FINGER_CHAINS):
            for first, second in zip(chain, chain[1:]):
                cv2.line(frame, points[first], points[second], HAND_COLOR, thin, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, thin + 1, HAND_COLOR, -1, cv2.LINE_AA)

        wrist_name = f"{side.upper()}_WRIST"
        if wrist_name not in indices:
            continue
        wrist = reliable_point(landmarks, indices, wrist_name, min_visibility, width, height)
        if wrist is None:
            continue
        palm = points[HAND_MIDDLE_MCP]
        cv2.line(frame, wrist, palm, HAND_LINK_COLOR, thick, cv2.LINE_AA)
        cv2.circle(frame, wrist, thick + 2, HAND_LINK_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, palm, thick + 1, HAND_LINK_COLOR, -1, cv2.LINE_AA)


def landmark_visibility_ratio(landmarks: list[Any], min_visibility: float) -> float:
    if not landmarks:
        return 0.0
    reliable = sum(1 for landmark in landmarks if is_reliable(landmark, min_visibility))
    return reliable / len(landmarks)


def draw_tracking_frame(
    cv2: Any,
    frame: Any,
    landmarks: list[Any],
    min_visibility: float,
) -> float:
    """Draw corner brackets around reliable pose points and return their visible ratio."""
    height, width = frame.shape[:2]
    reliable = [
        pixel_point(landmark, width, height)
        for landmark in landmarks
        if is_reliable(landmark, min_visibility)
    ]
    ratio = landmark_visibility_ratio(landmarks, min_visibility)
    if len(reliable) < 4:
        return ratio

    padding = scaled(24, frame)
    left = max(0, min(point[0] for point in reliable) - padding)
    top = max(0, min(point[1] for point in reliable) - padding)
    right = min(width - 1, max(point[0] for point in reliable) + padding)
    bottom = min(height - 1, max(point[1] for point in reliable) + padding)
    corner = max(scaled(18, frame), min(right - left, bottom - top) // 12)
    thickness = scaled(2, frame)

    segments = (
        ((left, top), (left + corner, top)),
        ((left, top), (left, top + corner)),
        ((right, top), (right - corner, top)),
        ((right, top), (right, top + corner)),
        ((left, bottom), (left + corner, bottom)),
        ((left, bottom), (left, bottom - corner)),
        ((right, bottom), (right - corner, bottom)),
        ((right, bottom), (right, bottom - corner)),
    )
    for start, end in segments:
        cv2.line(frame, start, end, TRACKING_COLOR, thickness, cv2.LINE_AA)
    draw_label(
        cv2,
        frame,
        f"TRACKED {ratio:.0%}",
        (left + scaled(6, frame), max(scaled(24, frame), top - scaled(8, frame))),
        TRACKING_COLOR,
        scale=0.45,
    )
    return ratio


def draw_overlay(
    cv2: Any,
    frame: Any,
    mode: str,
    person_detected: bool,
    calibrated: bool = False,
    recording: bool = False,
    robot_label: str = "off",
    gestures: tuple[str, ...] = (),
    status_lines: tuple[str, ...] = (),
) -> None:
    person = "person" if person_detected else "no person"
    flags = []
    if calibrated:
        flags.append("calibrated")
    if recording:
        flags.append("REC")
    lines = ["   ".join([person, f"robot: {robot_label}", *flags])]
    if gestures:
        lines.append("gestures: " + ", ".join(gestures[:4]))
    lines.extend(status_lines)
    margin = scaled(HUD_MARGIN, frame)
    draw_text_panel(cv2, frame, lines, (margin, margin))

    hint = f"output: {mode}   1 angles  2 points  3 both   c calibrate   r record   q quit"
    height = frame.shape[0]
    hint_top = height - margin - panel_height(frame, 1, HINT_LINE_HEIGHT, HINT_PADDING)
    draw_text_panel(
        cv2,
        frame,
        [hint],
        (margin, hint_top),
        scale=HINT_SCALE,
        line_height=HINT_LINE_HEIGHT,
        padding=HINT_PADDING,
        color=HINT_COLOR,
    )
