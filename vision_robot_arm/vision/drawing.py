from typing import Any

from vision_robot_arm.vision.landmarks import is_reliable


Point = tuple[int, int]
Color = tuple[int, int, int]


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


def draw_stick_figure(
    cv2: Any,
    frame: Any,
    landmarks: list[Any],
    indices: dict[str, int],
    min_visibility: float,
) -> None:
    arm_color = (40, 220, 255)
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


def draw_overlay(
    cv2: Any,
    frame: Any,
    mode: str,
    person_detected: bool,
    calibrated: bool = False,
    recording: bool = False,
    robot_debug: bool = False,
    gestures: tuple[str, ...] = (),
) -> None:
    status = "detected" if person_detected else "not detected"
    calibration = "on" if calibrated else "off"
    recording_status = "on" if recording else "off"
    robot_status = "on" if robot_debug else "off"
    lines = [
        f"Mode: {mode} | 1 angles  2 points  3 both",
        (
            f"Person: {status} | calibration: {calibration} | "
            f"recording: {recording_status} | robot: {robot_status}"
        ),
        "c calibrate | r record | q/Esc quit",
    ]
    if gestures:
        lines.append("Gestures: " + ", ".join(gestures[:3]))
    for row, text in enumerate(lines):
        y = 28 + row * 26
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
