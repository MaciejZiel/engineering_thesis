import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vision_robot_arm.core.config import ANGLE_MODE, BOTH_MODE, LANDMARK_MODE, AppConfig
from vision_robot_arm.core.display import enable_high_dpi_awareness
from vision_robot_arm.core.pose_state import LandmarkPoint, mirror_landmarks
from vision_robot_arm.core.runtime import load_runtime_dependencies
from vision_robot_arm.robot.controller import RobotController
from vision_robot_arm.robot.factory import create_robot_controller
from vision_robot_arm.robot.session import HardwareSession
from vision_robot_arm.robot.visualization import draw_simulation
from vision_robot_arm.vision.arm_pose import arm_elevation_angles
from vision_robot_arm.vision.dashboard import (
    ACTION_CALIBRATE,
    ACTION_CONTROL,
    ACTION_FULLSCREEN,
    ACTION_MODE,
    ACTION_QUIT,
    ACTION_RECORD,
    DashboardUi,
    cycle_output_mode,
)
from vision_robot_arm.vision.drawing import (
    draw_hands,
    draw_joint_angle_labels,
    draw_stick_figure,
    landmark_visibility_ratio,
)
from vision_robot_arm.vision.hand_gestures import (
    HandGestureFilter,
    WristAngleHold,
    anchor_hand_world_landmarks,
    assign_hand_indices,
    gestures_from_sides,
    refine_pose_wrists,
    wrist_angles_3d,
)
from vision_robot_arm.vision.hand_tracker import HandTracker
from vision_robot_arm.vision.landmarks import (
    build_landmark_indices,
    build_landmark_names,
)
from vision_robot_arm.vision.output import emit_console_data
from vision_robot_arm.vision.pose_tracker import PoseTracker
from vision_robot_arm.vision.recording import CsvPoseRecorder
from vision_robot_arm.vision.smoothing import LandmarkSmoother
from vision_robot_arm.vision.state_builder import PoseStateBuilder

WINDOW_NAME = "Motion Twin - Dual UR7e Control"
# Some containers refuse to seek. Without a cap the loop would spin without ever
# repainting or reading a key, and only an outside kill would stop it.
MAX_REWIND_ATTEMPTS = 3
CAMERA_FPS_TOLERANCE = 0.8
CAMERA_FALLBACK_SIZES = ((1920, 1080), (1280, 720), (960, 540), (640, 480))


@dataclass(frozen=True)
class CameraMode:
    width: int
    height: int
    fps: float
    codec: str

    def describe(self) -> str:
        return (
            f"{self.width}x{self.height} @ {self.fps:g} FPS ({self.codec or 'unknown'})"
        )


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
    inference_pool: ThreadPoolExecutor | None = None

    try:
        if not capture.isOpened():
            if config.video_path is not None:
                raise SystemExit(f"Could not open video file: {config.video_path}")
            else:
                raise SystemExit(
                    f"Could not open camera index {config.camera}. "
                    "Try another index, for example: python main.py --camera 1"
                )

        camera_mode = None
        if config.video_path is None:
            camera_mode = configure_camera(cv2, capture, config)
            print(f"Camera mode: {camera_mode.describe()}")

        tracker = PoseTracker(deps, config)
        wrist_hold = WristAngleHold()
        gesture_filter = HandGestureFilter()
        hand_smoothers: dict[str, LandmarkSmoother] = {}
        hand_world_smoothers: dict[str, LandmarkSmoother] = {}
        if config.hands:
            hand_tracker = HandTracker(deps, config)
            inference_pool = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="tracking"
            )
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
        simulation_canvas = _simulation_canvas(
            deps.np, dashboard.simulation_target_size()
        )
        last_frame_at = time.monotonic()
        display_fps = 0.0

        print(_source_started_message(config))
        print(
            "Keys: 1/2/3 console, c calibrate, r record, f fullscreen, d details, Tab/Enter navigate, q/Esc quit."
        )
        if config.test_mode:
            print(
                f"Test mode: joint angle labels and embedded robot simulation ({config.robot.backend})."
            )

        rewind_attempts = 0
        session_message: str | None = None
        while True:
            frame_started_at = time.monotonic()
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
                hand_world_smoothers.clear()
                continue
            rewind_attempts = 0

            frame = _fit_frame(cv2, frame, config)
            inference_frame = _resize_for_inference(cv2, frame, config)
            rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = _frame_timestamp_ms(
                cv2,
                capture,
                config,
                started_at,
                last_timestamp_ms,
                timestamp_offset_ms,
            )
            last_timestamp_ms = timestamp_ms
            if hand_tracker is not None and inference_pool is not None:
                pose_future = inference_pool.submit(
                    tracker.detect, rgb_frame, timestamp_ms
                )
                hand_future = inference_pool.submit(
                    hand_tracker.detect_frame, rgb_frame, timestamp_ms
                )
                detection = pose_future.result()
                hand_detection = hand_future.result()
            else:
                detection = tracker.detect(rgb_frame, timestamp_ms)
                hand_detection = None
            frame_aspect_ratio = frame.shape[1] / max(1, frame.shape[0])
            hand_gestures: tuple[str, ...] = ()
            extra_angles: dict[str, float] = {}
            extra_angle_sources: dict[str, str] = {}
            hands_by_side: dict[str, list] = {}
            hand_world_by_side: dict[str, list] = {}
            pose_landmarks = detection.landmarks
            if hand_tracker is not None and pose_landmarks:
                if hand_detection is None:
                    raise RuntimeError("Hand tracking result is missing.")
                hand_indices = assign_hand_indices(
                    hand_detection.landmarks,
                    pose_landmarks,
                    indices,
                    aspect_ratio=frame_aspect_ratio,
                    min_visibility=config.visibility_threshold,
                )
                image_hands = {
                    side: hand_detection.landmarks[index]
                    for side, index in hand_indices.items()
                    if index < len(hand_detection.landmarks)
                }
                local_world_hands = {
                    side: hand_detection.world_landmarks[index]
                    for side, index in hand_indices.items()
                    if index < len(hand_detection.world_landmarks)
                }
                hands_by_side = _smooth_hands(
                    image_hands,
                    hand_smoothers,
                    config.smoothing_alpha,
                    timestamp_ms,
                )
                hand_world_by_side = _smooth_hands(
                    anchor_hand_world_landmarks(
                        local_world_hands, detection.world_landmarks, indices
                    ),
                    hand_world_smoothers,
                    config.smoothing_alpha,
                    timestamp_ms,
                )
                # Everything downstream hinges on the wrist, so correct it first.
                pose_landmarks = refine_pose_wrists(
                    pose_landmarks, hands_by_side, indices
                )
                hand_gestures = gesture_filter.update(
                    gestures_from_sides(hands_by_side, aspect_ratio=frame_aspect_ratio),
                    timestamp_ms,
                )
                extra_angles.update(
                    wrist_hold.update(
                        wrist_angles_3d(
                            local_world_hands,
                            pose_landmarks,
                            detection.world_landmarks,
                            indices,
                            min_visibility=config.visibility_threshold,
                        ),
                        timestamp_ms,
                    )
                )
                extra_angle_sources.update(
                    {
                        name: (
                            "hand_world_3d_held"
                            if name in wrist_hold.held_names
                            else "hand_world_3d"
                        )
                        for name in extra_angles
                        if name.endswith("_wrist")
                    }
                )
            if pose_landmarks:
                elevation_angles = arm_elevation_angles(
                    pose_landmarks,
                    indices,
                    config.visibility_threshold,
                    frame_aspect_ratio,
                    world_landmarks=detection.world_landmarks,
                )
                extra_angles.update(elevation_angles)
                extra_angle_sources.update(
                    {name: "pose_world_3d" for name in elevation_angles}
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
                    hand_landmarks=hands_by_side,
                    hand_world_landmarks=hand_world_by_side,
                    world_only=True,
                    extra_angle_sources=extra_angle_sources,
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
                hand_world_smoothers.clear()

            now = time.monotonic()
            frame_elapsed = now - last_frame_at
            if frame_elapsed > 0:
                instant_fps = 1.0 / frame_elapsed
                display_fps = (
                    instant_fps
                    if display_fps == 0.0
                    else 0.9 * display_fps + 0.1 * instant_fps
                )
            last_frame_at = now

            robot_state = robot_controller.robot_state()
            simulation_size = dashboard.simulation_target_size()
            if (
                simulation_canvas.shape[1],
                simulation_canvas.shape[0],
            ) != simulation_size:
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
                status_lines=tuple(robot_controller.status_lines())
                + ((recorder.last_error,) if recorder.last_error else ()),
                tracking_quality=tracking_quality,
                fps=display_fps,
                source_label=_source_label(config),
                robot_state_available=robot_state is not None,
                can_calibrate=can_calibrate,
                control_label=(
                    robot_controller.action_label
                    if isinstance(robot_controller, HardwareSession)
                    else None
                ),
                alert=(
                    getattr(robot_controller, "error", None)
                    or recorder.last_error
                    or session_message
                ),
            )
            cv2.imshow(WINDOW_NAME, dashboard_frame)

            key = (
                cv2.waitKey(
                    _remaining_frame_delay_ms(
                        wait_delay_ms, time.monotonic() - frame_started_at
                    )
                )
                & 0xFF
            )
            dashboard.sync_window_size()
            dashboard.handle_key(key)
            action = dashboard.consume_action()
            if isinstance(robot_controller, HardwareSession):
                if key == ord("h") or action == ACTION_CONTROL:
                    robot_controller.advance()
                if key == ord("p"):
                    robot_controller.pause()
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return 0
            if key in (ord("q"), 27) or action == ACTION_QUIT:
                return 0
            if key == ord("f") or action == ACTION_FULLSCREEN:
                dashboard.toggle_fullscreen()
            if (key == ord("c") or action == ACTION_CALIBRATE) and can_calibrate:
                robot_controller.reset()
                sides = tuple(config.robot.hosts) or ("left", "right")
                required = tuple(
                    f"{side}_{source}"
                    for side in sides
                    for source in ("shoulder_elevation", "elbow", "wrist")
                )
                count = state_builder.capture_calibration(current_state, required)
                session_message = (
                    f"Calibration captured from {count} angles."
                    if count
                    else "Hold all arm joints steady and visible for 0.8 seconds, then retry calibration."
                )
                print(session_message)
            if key in (ord("k"), ord("l"), ord("x")):
                robot_controller.reset()
                profile = config.recording_dir / "calibration.json"
                try:
                    if key == ord("k"):
                        state_builder.save_calibration(profile)
                        session_message = f"Calibration saved: {profile}"
                    elif key == ord("l"):
                        state_builder.load_calibration(profile)
                        session_message = "Calibration loaded. Check the preview before enabling control."
                    else:
                        state_builder.reset_calibration()
                        session_message = (
                            "Calibration reset. Absolute mapping restored."
                        )
                except (OSError, ValueError) as error:
                    session_message = f"Calibration failed: {error}"
                print(session_message)
            if key == ord("r") or action == ACTION_RECORD:
                is_recording, path = recorder.toggle(names)
                if recorder.last_error:
                    print(recorder.last_error)
                elif is_recording:
                    print(f"Recording started: {path}")
                else:
                    print(f"Recording stopped: {path}")
            mode = (
                cycle_output_mode(mode)
                if action == ACTION_MODE
                else update_mode_from_key(key, mode)
            )
    finally:
        _release_all(
            ("robot", None if robot_controller is None else robot_controller.close),
            ("recording", None if recorder is None else recorder.stop),
            (
                "tracking workers",
                None
                if inference_pool is None
                else lambda: inference_pool.shutdown(wait=True, cancel_futures=True),
            ),
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


def configure_camera(cv2: object, capture: object, config: AppConfig) -> CameraMode:
    """Prefer capture rate over resolution and request compressed USB transport first."""
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    capture.set(cv2.CAP_PROP_FOURCC, mjpg)
    capture.set(cv2.CAP_PROP_FPS, config.camera_fps)

    candidates = []
    if config.width > 0 and config.height > 0:
        candidates.append((config.width, config.height))
        candidates.extend(
            size
            for size in CAMERA_FALLBACK_SIZES
            if size[0] <= config.width and size[1] <= config.height
        )
    else:
        candidates.append((0, 0))

    mode = _camera_mode(cv2, capture)
    for width, height in dict.fromkeys(candidates):
        if width > 0 and height > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Some drivers reset FPS when the dimensions change.
        capture.set(cv2.CAP_PROP_FPS, config.camera_fps)
        mode = _camera_mode(cv2, capture)
        if mode.fps <= 0 or mode.fps >= config.camera_fps * CAMERA_FPS_TOLERANCE:
            break
    return mode


def _camera_mode(cv2: object, capture: object) -> CameraMode:
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip(
        "\x00"
    )
    return CameraMode(
        width=max(0, round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        height=max(0, round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        fps=max(0.0, float(capture.get(cv2.CAP_PROP_FPS))),
        codec=codec,
    )


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


def _resize_for_inference(cv2: object, frame: object, config: AppConfig) -> object:
    """Keep the high-resolution preview while giving both models a smaller, equal frame."""
    height, width = frame.shape[:2]
    if width <= config.inference_width and height <= config.inference_height:
        return frame
    scale = min(config.inference_width / width, config.inference_height / height)
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
    if not math.isfinite(fps) or fps <= 0:
        return 1
    return max(1, int(1000 / fps))


def _remaining_frame_delay_ms(period_ms: int, processing_seconds: float) -> int:
    """Pump UI events without adding a second full frame period after inference."""
    return max(1, math.ceil(period_ms - max(0.0, processing_seconds) * 1000))


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
