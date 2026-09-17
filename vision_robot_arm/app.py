import time

from vision_robot_arm.core.config import ANGLE_MODE, BOTH_MODE, LANDMARK_MODE, AppConfig
from vision_robot_arm.core.display import enable_high_dpi_awareness
from vision_robot_arm.core.pose_state import LandmarkPoint, mirror_landmarks
from vision_robot_arm.core.runtime import load_runtime_dependencies
from vision_robot_arm.robot.controller import RobotController
from vision_robot_arm.robot.factory import create_robot_controller
from vision_robot_arm.robot.visualization import draw_simulation
from vision_robot_arm.vision.dashboard import (
    ACTION_CALIBRATE,
    ACTION_FULLSCREEN,
    ACTION_MODE,
    ACTION_QUIT,
    ACTION_RECORD,
    DashboardUi,
    cycle_output_mode,
)
from vision_robot_arm.vision.arm_pose import arm_elevation_angles
from vision_robot_arm.vision.drawing import (
    draw_hands,
    draw_joint_angle_labels,
    draw_stick_figure,
    landmark_visibility_ratio,
)
from vision_robot_arm.vision.hand_gestures import (
    HandGestureFilter,
    WristAngleHold,
    assign_hand_sides,
    gestures_from_sides,
    refine_pose_wrists,
    wrist_angles_from_sides,
)
from vision_robot_arm.vision.hand_tracker import HandTracker
from vision_robot_arm.vision.landmarks import build_landmark_indices, build_landmark_names
from vision_robot_arm.vision.output import emit_console_data
from vision_robot_arm.vision.pose_tracker import PoseTracker
from vision_robot_arm.vision.recording import CsvPoseRecorder
from vision_robot_arm.vision.smoothing import LandmarkSmoother
from vision_robot_arm.vision.state_builder import PoseStateBuilder

WINDOW_NAME = "Motion Twin - Dual UR7e Control"
# Some containers refuse to seek. Without a cap the loop would spin without ever
# repainting or reading a key, and only an outside kill would stop it.
MAX_REWIND_ATTEMPTS = 3


def update_mode_from_key(key: int, current_mode: str) -> str:
    if key == ord("1"):
        return ANGLE_MODE
    if key == ord("2"):
        return LANDMARK_MODE
    if key == ord("3"):
        return BOTH_MODE
    return current_mode


def run_app(config: AppConfig) -> int:
    enable_high_dpi_awareness()
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
        wrist_hold = WristAngleHold()
        gesture_filter = HandGestureFilter()
        hand_smoothers: dict[str, LandmarkSmoother] = {}
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
        timestamp_offset_ms = 0
        dashboard = DashboardUi(cv2, deps.np, WINDOW_NAME)
        dashboard.open()
        simulation_canvas = _simulation_canvas(deps.np, dashboard.simulation_target_size())
        last_frame_at = time.monotonic()
        display_fps = 0.0

        print(_source_started_message(config))
        print("Keys: 1/2/3 console, c calibrate, r record, f fullscreen, d details, Tab/Enter navigate, q/Esc quit.")
        if config.test_mode:
            print(f"Test mode: joint angle labels and embedded robot simulation ({config.robot.backend}).")

        rewind_attempts = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                if config.video_path is None:
                    print("Camera frame could not be read.")
                    return 1
                if not config.loop_video:
                    print("Video ended.")
                    return 0
                rewind_attempts += 1
                if rewind_attempts > MAX_REWIND_ATTEMPTS:
                    print(f"Video ended: {config.video_path} could not be rewound.")
                    return 1
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Keep video time running across the rewind; the filters downstream
                # derive their time constants from these stamps.
                timestamp_offset_ms = last_timestamp_ms + wait_delay_ms
                state_builder.reset_tracking()
                robot_controller.reset()
                wrist_hold.reset()
                gesture_filter.reset()
                hand_smoothers.clear()
                continue
            rewind_attempts = 0

            frame = _fit_frame(cv2, frame, config)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = _frame_timestamp_ms(
                cv2,
                capture,
                config,
                started_at,
                last_timestamp_ms,
                timestamp_offset_ms,
            )
            last_timestamp_ms = timestamp_ms
            detection = tracker.detect(rgb_frame, timestamp_ms)
            frame_aspect_ratio = frame.shape[1] / max(1, frame.shape[0])
            hand_gestures: tuple[str, ...] = ()
            extra_angles: dict[str, float] = {}
            hands_by_side: dict[str, list] = {}
            pose_landmarks = detection.landmarks
            if hand_tracker is not None and pose_landmarks:
                hands = hand_tracker.detect(rgb_frame, timestamp_ms)
                hands_by_side = _smooth_hands(
                    assign_hand_sides(
                        hands,
                        pose_landmarks,
                        indices,
                        aspect_ratio=frame_aspect_ratio,
                        min_visibility=config.visibility_threshold,
                    ),
                    hand_smoothers,
                    config.smoothing_alpha,
                    timestamp_ms,
                )
                # Everything downstream hinges on the wrist, so correct it first.
                pose_landmarks = refine_pose_wrists(pose_landmarks, hands_by_side, indices)
                hand_gestures = gesture_filter.update(
                    gestures_from_sides(hands_by_side, aspect_ratio=frame_aspect_ratio),
                    timestamp_ms,
                )
                extra_angles.update(
                    wrist_hold.update(
                        wrist_angles_from_sides(
                            hands_by_side,
                            pose_landmarks,
                            indices,
                            frame_aspect_ratio,
                            min_visibility=config.visibility_threshold,
                        ),
                        timestamp_ms,
                    )
                )
            if pose_landmarks:
                extra_angles.update(
                    arm_elevation_angles(
                        pose_landmarks,
                        indices,
                        config.visibility_threshold,
                        frame_aspect_ratio,
                    )
                )
            if mirrored:
                frame = cv2.flip(frame, 1)

            current_state = None
            tracking_quality = 0.0
            if pose_landmarks:
                current_state = state_builder.build(
                    timestamp_ms,
                    pose_landmarks,
                    detection.world_landmarks,
                    extra_gestures=hand_gestures,
                    extra_angles=extra_angles,
                    aspect_ratio=frame_aspect_ratio,
                    hand_tracking_enabled=hand_tracker is not None,
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
                display_hands = {
                    side: mirror_landmarks(hand) if mirrored else hand
                    for side, hand in hands_by_side.items()
                }
                draw_hands(
                    cv2,
                    frame,
                    display_hands,
                    display_landmarks,
                    indices,
                    config.visibility_threshold,
                )
                tracking_quality = landmark_visibility_ratio(
                    display_landmarks,
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
                wrist_hold.reset()
                gesture_filter.reset()
                hand_smoothers.clear()

            now = time.monotonic()
            frame_elapsed = now - last_frame_at
            if frame_elapsed > 0:
                instant_fps = 1.0 / frame_elapsed
                display_fps = instant_fps if display_fps == 0.0 else 0.9 * display_fps + 0.1 * instant_fps
            last_frame_at = now

            robot_state = robot_controller.robot_state()
            simulation_size = dashboard.simulation_target_size()
            if (simulation_canvas.shape[1], simulation_canvas.shape[0]) != simulation_size:
                simulation_canvas = _simulation_canvas(deps.np, simulation_size)
            draw_simulation(cv2, simulation_canvas, robot_state, compact=True)
            can_calibrate = current_state is not None and any(
                value is not None for value in current_state.angles.values()
            )
            dashboard_frame = dashboard.render(
                frame,
                simulation_canvas,
                mode=mode,
                person_detected=detection.has_pose,
                calibrated=state_builder.calibrated,
                recording=recorder.is_recording,
                robot_label=config.robot.backend if config.robot.enabled else "off",
                gestures=current_state.gestures if current_state else (),
                status_lines=tuple(robot_controller.status_lines()) + (
                    (recorder.last_error,) if recorder.last_error else ()
                ),
                tracking_quality=tracking_quality,
                fps=display_fps,
                source_label=_source_label(config),
                robot_state_available=robot_state is not None,
                can_calibrate=can_calibrate,
            )
            cv2.imshow(WINDOW_NAME, dashboard_frame)

            key = cv2.waitKey(wait_delay_ms) & 0xFF
            dashboard.sync_window_size()
            dashboard.handle_key(key)
            action = dashboard.consume_action()
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return 0
            if key in (ord("q"), 27) or action == ACTION_QUIT:
                return 0
            if key == ord("f") or action == ACTION_FULLSCREEN:
                dashboard.toggle_fullscreen()
            if (key == ord("c") or action == ACTION_CALIBRATE) and can_calibrate:
                count = state_builder.capture_calibration(current_state)
                print(f"Calibration captured from {count} angles.")
            if key == ord("r") or action == ACTION_RECORD:
                is_recording, path = recorder.toggle(names)
                if recorder.last_error:
                    print(recorder.last_error)
                elif is_recording:
                    print(f"Recording started: {path}")
                else:
                    print(f"Recording stopped: {path}")
            mode = cycle_output_mode(mode) if action == ACTION_MODE else update_mode_from_key(key, mode)
    finally:
        _release_all(
            ("robot", None if robot_controller is None else robot_controller.close),
            ("recording", None if recorder is None else recorder.stop),
            ("hand tracker", None if hand_tracker is None else hand_tracker.close),
            ("pose tracker", None if tracker is None else tracker.close),
            ("camera", capture.release),
        )
        cv2.destroyAllWindows()


def _smooth_hands(
    hands_by_side: dict[str, list],
    smoothers: dict[str, LandmarkSmoother],
    alpha: float,
    timestamp_ms: int | None = None,
) -> dict[str, list]:
    """The hand tracker output is raw, and it was drawn and measured exactly as it arrived."""
    for side in set(smoothers) - set(hands_by_side):
        del smoothers[side]
    smoothed = {}
    for side, hand in hands_by_side.items():
        smoother = smoothers.setdefault(side, LandmarkSmoother(alpha))
        smoothed[side] = smoother.update(
            [LandmarkPoint.from_landmark(point) for point in hand], timestamp_ms
        )
    return smoothed


def _release_all(*resources: tuple[str, object]) -> None:
    """Close everything, whatever fails: a full disk must not leave an arm streaming."""
    for name, close in resources:
        if close is None:
            continue
        try:
            close()
        except Exception as error:  # noqa: BLE001 - shutdown continues regardless
            print(f"Could not close the {name}: {error}")


def _simulation_canvas(np: object, size: tuple[int, int]) -> object:
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)


def _fit_frame(cv2: object, frame: object, config: AppConfig) -> object:
    if config.width <= 0 or config.height <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame
    scale = min(config.width / width, config.height / height, 1.0)
    if scale >= 1.0:
        return frame
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def _source_started_message(config: AppConfig) -> str:
    if config.video_path is not None:
        return f"Video started: {config.video_path}"
    return "Camera started."


def _source_label(config: AppConfig) -> str:
    if config.video_path is not None:
        return config.video_path.name[:24]
    return f"CAM {config.camera}"


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
    offset_ms: int = 0,
) -> int:
    """Strictly increasing stamps that keep real spacing, which MediaPipe and the filters need."""
    if config.video_path is None:
        timestamp_ms = int((time.monotonic() - started_at) * 1000)
    else:
        timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC)) + offset_ms
    # Two camera frames inside the same millisecond would make the landmarker raise
    # "Input timestamp must be monotonically increasing" and kill the loop.
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    return timestamp_ms
