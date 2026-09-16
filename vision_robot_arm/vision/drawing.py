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
JOINT_LABEL_COLOR: Color = (0, 230, 255)
HUD_MARGIN = 12
HINT_SCALE = 0.42
HINT_LINE_HEIGHT = 16
HINT_PADDING = 6
HINT_COLOR: Color = (190, 190, 190)
TRACKING_COLOR: Color = (0, 145, 255)


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
    arm_color = (0, 145, 255)
    torso_color = (80, 255, 120)
    hip_color = (255, 210, 60)
    leg_color = (255, 120, 80)
    head_color = (210, 210, 255)

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
