import time

from vision_robot_arm.config import ANGLE_MODE, BOTH_MODE, LANDMARK_MODE, AppConfig
from vision_robot_arm.drawing import draw_overlay, draw_stick_figure
from vision_robot_arm.landmarks import build_landmark_indices, build_landmark_names
from vision_robot_arm.output import emit_console_data
from vision_robot_arm.pose_tracker import PoseTracker
from vision_robot_arm.runtime import load_runtime_dependencies


def update_mode_from_key(key: int, current_mode: str) -> str:
    if key == ord("1"):
        return ANGLE_MODE
    if key == ord("2"):
        return LANDMARK_MODE
    if key == ord("3"):
        return BOTH_MODE
    return current_mode


def run_app(config: AppConfig) -> int:
    deps = load_runtime_dependencies()
    cv2 = deps.cv2
    indices = build_landmark_indices(deps.vision)
    names = build_landmark_names(deps.vision)

    capture = cv2.VideoCapture(config.camera)
    tracker = None

    try:
        if config.width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        if config.height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

        if not capture.isOpened():
            raise SystemExit(
                f"Could not open camera index {config.camera}. "
                "Try another index, for example: python main.py --camera 1"
            )

        tracker = PoseTracker(deps, config)
        mode = ANGLE_MODE
        next_print_at = 0.0
        started_at = time.monotonic()
        window_name = "Vision Robot Arm - Pose Tracker"

        print("Camera started.")
        print("Keys: 1 angles, 2 landmarks, 3 both, q/Esc quit.")

        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame could not be read.")
                return 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.monotonic() - started_at) * 1000)
            detection = tracker.detect(rgb_frame, timestamp_ms)

            if detection.landmarks:
                draw_stick_figure(
                    cv2,
                    frame,
                    detection.landmarks,
                    indices,
                    config.visibility_threshold,
                )

                now = time.monotonic()
                if now >= next_print_at:
                    emit_console_data(
                        mode,
                        detection.landmarks,
                        detection.world_landmarks,
                        indices,
                        names,
                        config.visibility_threshold,
                    )
                    next_print_at = now + config.print_interval

            draw_overlay(cv2, frame, mode, detection.has_pose)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            mode = update_mode_from_key(key, mode)
    finally:
        if tracker is not None:
            tracker.close()
        capture.release()
        cv2.destroyAllWindows()
