import time

from vision_robot_arm.core.config import ANGLE_MODE, BOTH_MODE, LANDMARK_MODE, AppConfig
from vision_robot_arm.core.pose_state import mirror_landmarks
from vision_robot_arm.core.runtime import load_runtime_dependencies
from vision_robot_arm.robot.controller import RobotController
from vision_robot_arm.robot.factory import create_robot_controller
from vision_robot_arm.robot.visualization import draw_simulation
from vision_robot_arm.vision.drawing import (
    draw_joint_angle_labels,
    draw_overlay,
    draw_stick_figure,
)
from vision_robot_arm.vision.hand_gestures import detect_hand_gestures
from vision_robot_arm.vision.hand_tracker import HandTracker
from vision_robot_arm.vision.landmarks import build_landmark_indices, build_landmark_names
from vision_robot_arm.vision.output import emit_console_data
from vision_robot_arm.vision.pose_tracker import PoseTracker
from vision_robot_arm.vision.recording import CsvPoseRecorder
from vision_robot_arm.vision.state_builder import PoseStateBuilder

SIMULATION_WINDOW = "Vision Robot Arm - Robot Simulation"
SIMULATION_SIZE = (960, 540)


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

    source = str(config.video_path) if config.video_path is not None else config.camera
    capture = cv2.VideoCapture(source)
    tracker: PoseTracker | None = None
    hand_tracker: HandTracker | None = None
    recorder: CsvPoseRecorder | None = None
    robot_controller: RobotController | None = None

    try:
        if config.video_path is None and config.width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        if config.video_path is None and config.height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

        if not capture.isOpened():
            if config.video_path is not None:
                raise SystemExit(f"Could not open video file: {config.video_path}")
            else:
                raise SystemExit(
                    f"Could not open camera index {config.camera}. "
                    "Try another index, for example: python main.py --camera 1"
                )

        tracker = PoseTracker(deps, config)
        if config.hands:
            hand_tracker = HandTracker(deps, config)
        recorder = CsvPoseRecorder(config.recording_dir)
        robot_controller = create_robot_controller(config.robot)
        state_builder = PoseStateBuilder(
            indices=indices,
            min_visibility=config.visibility_threshold,
            smoothing_alpha=config.smoothing_alpha,
        )
        mode = ANGLE_MODE
        next_print_at = 0.0
        started_at = time.monotonic()
        last_timestamp_ms = -1
        wait_delay_ms = _frame_wait_delay_ms(cv2, capture, config)
        mirrored = config.mirror and config.video_path is None
        window_name = "Vision Robot Arm - Pose Tracker"
        simulation_canvas = None
        if config.test_mode:
            simulation_canvas = deps.np.zeros(
                (SIMULATION_SIZE[1], SIMULATION_SIZE[0], 3), dtype=deps.np.uint8
            )

        print(_source_started_message(config))
        print("Keys: 1 angles, 2 landmarks, 3 both, c calibrate, r record, q/Esc quit.")
        if config.test_mode:
            print(f"Test mode: joint angle labels and robot simulation window ({config.robot.backend}).")

        while True:
            ok, frame = capture.read()
            if not ok:
                if config.video_path is not None:
                    if config.loop_video:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        state_builder.reset_tracking()
                        robot_controller.reset()
                        continue
                    print("Video ended.")
                    return 0
                print("Camera frame could not be read.")
                return 1

            frame = _fit_frame(cv2, frame, config)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = _frame_timestamp_ms(
                cv2,
                capture,
                config,
                started_at,
                last_timestamp_ms,
            )
            last_timestamp_ms = timestamp_ms
            detection = tracker.detect(rgb_frame, timestamp_ms)
            hand_gestures: tuple[str, ...] = ()
            if hand_tracker is not None and detection.landmarks:
                hands = hand_tracker.detect(rgb_frame, timestamp_ms)
                hand_gestures = detect_hand_gestures(hands, detection.landmarks, indices)
            if mirrored:
                frame = cv2.flip(frame, 1)

            current_state = None
            if detection.landmarks:
                current_state = state_builder.build(
                    timestamp_ms,
                    detection.landmarks,
                    detection.world_landmarks,
                    extra_gestures=hand_gestures,
                )
                display_landmarks = (
                    mirror_landmarks(current_state.landmarks)
                    if mirrored
                    else current_state.landmarks
                )
                draw_stick_figure(
                    cv2,
                    frame,
                    display_landmarks,
                    indices,
                    config.visibility_threshold,
                )
                if config.test_mode:
                    draw_joint_angle_labels(
                        cv2,
                        frame,
                        display_landmarks,
                        indices,
                        current_state.angles,
                        config.visibility_threshold,
                    )

                recorder.write_state(current_state, names)
                robot_controller.update(current_state)

                now = time.monotonic()
                if now >= next_print_at:
                    emit_console_data(
                        mode,
                        current_state,
                        names,
                    )
                    if config.test_mode:
                        for line in robot_controller.status_lines():
                            print(line)
                    next_print_at = now + config.print_interval
            else:
                state_builder.reset_tracking()
                robot_controller.reset()

            draw_overlay(
                cv2,
                frame,
                mode,
                detection.has_pose,
                calibrated=state_builder.calibrated,
                recording=recorder.is_recording,
                robot_label=config.robot.backend if config.robot.enabled else "off",
                gestures=current_state.gestures if current_state else (),
                status_lines=() if config.test_mode else tuple(robot_controller.status_lines()),
            )
            cv2.imshow(window_name, frame)
            if simulation_canvas is not None:
                draw_simulation(cv2, simulation_canvas, robot_controller.robot_state())
                cv2.imshow(SIMULATION_WINDOW, simulation_canvas)

            key = cv2.waitKey(wait_delay_ms) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("c") and current_state is not None:
                count = state_builder.capture_calibration(current_state)
                print(f"Calibration captured from {count} angles.")
            if key == ord("r"):
                is_recording, path = recorder.toggle(names)
                if is_recording:
                    print(f"Recording started: {path}")
                else:
                    print(f"Recording stopped: {path}")
            mode = update_mode_from_key(key, mode)
    finally:
        if recorder is not None:
            recorder.stop()
        if robot_controller is not None:
            robot_controller.close()
        if hand_tracker is not None:
            hand_tracker.close()
        if tracker is not None:
            tracker.close()
        capture.release()
        cv2.destroyAllWindows()


def _fit_frame(cv2: object, frame: object, config: AppConfig) -> object:
    if config.width <= 0 or config.height <= 0:
        return frame
    height, width = frame.shape[:2]
    if (width, height) == (config.width, config.height):
        return frame
    return cv2.resize(frame, (config.width, config.height), interpolation=cv2.INTER_LINEAR)


def _source_started_message(config: AppConfig) -> str:
    if config.video_path is not None:
        return f"Video started: {config.video_path}"
    return "Camera started."


def _frame_wait_delay_ms(cv2: object, capture: object, config: AppConfig) -> int:
    if config.video_path is None:
        return 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        return 1
    return max(1, int(1000 / fps))


def _frame_timestamp_ms(
    cv2: object,
    capture: object,
    config: AppConfig,
    started_at: float,
    last_timestamp_ms: int,
) -> int:
    if config.video_path is None:
        return int((time.monotonic() - started_at) * 1000)

    timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    return timestamp_ms
