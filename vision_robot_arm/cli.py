import argparse
from pathlib import Path

from vision_robot_arm.app import run_app
from vision_robot_arm.core.config import (
    DEFAULT_HAND_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RECORDING_DIR,
    AppConfig,
)
from vision_robot_arm.core.runtime import load_camera_dependency
from vision_robot_arm.robot.config import (
    BACKEND_CHOICES,
    BACKEND_DEBUG,
    BACKEND_NONE,
    BACKEND_SIM,
    COMMISSIONING_MAX_EXCURSION_DEG,
    COMMISSIONING_MAX_SPEED_DEG_S,
    DEFAULT_ELBOW_MAPPING,
    DEFAULT_SHOULDER_MAPPING,
    DEFAULT_WRIST_MAPPING,
    UR7E_MAX_JOINT_SPEED_DEG_S,
    OPERATION_CHOICES,
    OPERATION_MONITOR,
    UR_DASHBOARD_PORT,
    UR_RTDE_PORT,
    UR_SECONDARY_PORT,
    GRIPPER_DRIVER_CHOICES,
    JointLimit,
    JointMapping,
    RobotConfig,
)
from vision_robot_arm.robot.targets import JOINT_NAMES
from vision_robot_arm.vision.camera import (
    BACKEND_CHOICES as CAMERA_BACKEND_CHOICES,
    FORMAT_CHOICES as CAMERA_FORMAT_CHOICES,
    discover_cameras,
    format_camera_list,
)
from vision_robot_arm.vision.camera_diagnostics import DiagnosticSettings, diagnose_camera


def _parse_camera_target(val: str) -> int | str:
    if val.lower() == "auto":
        return "auto"
    try:
        index = int(val)
        if index < 0:
            raise ValueError("Negative camera index")
        return index
    except ValueError as error:
        raise argparse.ArgumentTypeError("Camera must be a non-negative index or 'auto'.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Webcam pose tracker for the vision robot arm prototype (two UR7e cobots)."
    )
    parser.add_argument(
        "--tracking-space", choices=("2d", "3d"), default=None,
        help="Camera measurement space. Test mode defaults to 2d; otherwise 3d.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available video cameras on the system and exit.",
    )
    parser.add_argument(
        "--diagnose-camera",
        type=_parse_camera_target,
        default=None,
        help="Measure delivery from a camera (index or 'auto'), without models or UI, then exit.",
    )
    parser.add_argument(
        "--diagnostic-seconds", type=float, default=5.0,
        help="Camera measurement window after warmup, 1..60 seconds. Default: 5.",
    )
    parser.add_argument(
        "--diagnostic-warmup", type=float, default=2.0,
        help="Camera warmup before measurement, 0..30 seconds. Default: 2.",
    )
    parser.add_argument(
        "--hand-detection-confidence",
        type=float,
        default=0.4,
        help="Independent hand detection confidence, 0..1. Default: 0.4.",
    )
    parser.add_argument(
        "--hand-presence-confidence",
        type=float,
        default=0.4,
        help="Independent hand presence confidence, 0..1. Default: 0.4.",
    )
    parser.add_argument(
        "--camera",
        type=_parse_camera_target,
        default=0,
        help="OpenCV camera index or 'auto'. Default: 0.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=CAMERA_BACKEND_CHOICES,
        default="auto",
        help="Video capture backend API. Default: auto.",
    )
    parser.add_argument(
        "--camera-format",
        choices=CAMERA_FORMAT_CHOICES,
        default="auto",
        help="Preferred pixel format for camera capture (e.g. mjpg for high FPS at 1080p). Default: auto.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional video file to process instead of a webcam.",
    )
    parser.add_argument(
        "--loop-video",
        action="store_true",
        help="Loop the video file when it reaches the end.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Frame width in pixels; frames are resized when the camera gives another size. 0 keeps the camera size. Default: 1920.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Frame height in pixels; frames are resized when the camera gives another size. 0 keeps the camera size. Default: 1080.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Preferred camera frame rate; resolution is reduced first to preserve it. Default: 30.",
    )
    parser.add_argument(
        "--inference-width",
        type=int,
        default=960,
        help="Maximum image width used by the tracking models. Default: 960.",
    )
    parser.add_argument(
        "--inference-height",
        type=int,
        default=540,
        help="Maximum image height used by the tracking models. Default: 540.",
    )
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="Do not mirror the camera image. Video files are never mirrored.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Seconds between console metric prints. Default: 0.5.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.55,
        help="Minimum MediaPipe visibility for drawing/angle calculation. Default: 0.55.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.35,
        help="Low-pass smoothing factor for landmarks and angles. Lower is smoother. Default: 0.35.",
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=DEFAULT_RECORDING_DIR,
        help=f"Directory for CSV recordings. Default: {DEFAULT_RECORDING_DIR}",
    )
    parser.add_argument(
        "--skeleton",
        type=Path,
        default=None,
        help=(
            "Load a measured skeleton profile so bone lengths stay fixed. "
            "Press b in the window to record one."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to a MediaPipe Pose Landmarker .task model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=DEFAULT_HAND_MODEL_PATH,
        help=f"Path to a MediaPipe Hand Landmarker .task model. Default: {DEFAULT_HAND_MODEL_PATH}",
    )
    parser.add_argument(
        "--no-hands",
        dest="hands",
        action="store_false",
        help="Disable hand tracking (open hand / fist gestures for the gripper).",
    )
    parser.add_argument(
        "--hand-tracking-interval",
        type=int,
        default=2,
        help="Run the heavier hand model every N frames and reuse its result between runs. Default: 2.",
    )
    parser.add_argument(
        "--num-poses",
        type=int,
        default=1,
        help="Maximum number of people to detect. Drawing/printing uses the first pose. Default: 1.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum pose detection confidence. Default: 0.5.",
    )
    parser.add_argument(
        "--min-pose-presence-confidence",
        type=float,
        default=0.5,
        help="Minimum pose presence confidence. Default: 0.5.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum pose tracking confidence. Default: 0.5.",
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help=(
            "Show joint angles next to the arm joints and open the robot simulation window. "
            "Uses the simulated robots when no --robot-backend is given."
        ),
    )

    robot = parser.add_argument_group("robot")
    robot.add_argument(
        "--robot-backend",
        choices=BACKEND_CHOICES,
        default=BACKEND_NONE,
        help=(
            "Where mapped robot commands go: none, debug (print), sim (two simulated UR7e arms), "
            "ur (real UR7e cobots over URScript/TCP) or serial (generic serial line protocol). "
            "Default: none."
        ),
    )
    robot.add_argument(
        "--robot-debug",
        action="store_true",
        help="Deprecated alias for --robot-backend debug.",
    )
    robot.add_argument(
        "--robot-print-interval",
        type=float,
        default=0.5,
        help="Seconds between robot debug command prints. Default: 0.5.",
    )
    robot.add_argument(
        "--robot-operation",
        choices=OPERATION_CHOICES,
        default=OPERATION_MONITOR,
        help="UR mode: read-only monitor (default), commissioning, or vision tracking.",
    )
    robot.add_argument(
        "--robot-commissioning-joint",
        choices=JOINT_NAMES,
        default="shoulder",
        help="Single joint enabled in commissioning mode. Default: shoulder.",
    )
    robot.add_argument(
        "--robot-tracking-excursion",
        type=float,
        default=40.0,
        help="Maximum tracking offset from the captured pose in degrees. Default: 40.",
    )
    robot.add_argument(
        "--robot-tracking-acceleration",
        type=float,
        default=7.0,
        help="Linear joint acceleration/deceleration limit in deg/s^2. Default: 7.",
    )
    robot.add_argument(
        "--robot-tracking-loss-grace",
        type=float,
        default=0.4,
        help="Keep the last valid target through brief camera dropouts. Default: 0.4 s.",
    )
    robot.add_argument(
        "--robot-telemetry-log",
        default=None,
        help="Optional JSONL file for targets, setpoints and RTDE feedback.",
    )
    robot.add_argument(
        "--robot-feedback-log",
        default=None,
        help="Optional separate JSONL file containing only values received from robot RTDE.",
    )
    robot.add_argument(
        "--robot-gripper-gesture-frames",
        type=int,
        default=3,
        help="Consecutive open/fist results required before a gripper command. Default: 3.",
    )
    robot.add_argument(
        "--robot-gripper-driver",
        choices=GRIPPER_DRIVER_CHOICES,
        default="digital",
        help="Gripper transport: tool digital output or Robotiq URCap socket.",
    )
    robot.add_argument("--robot-gripper-speed", type=int, default=80)
    robot.add_argument("--robot-gripper-force", type=int, default=50)
    robot.add_argument(
        "--robot-commissioning-speed",
        type=float,
        default=30.0,
        help=f"Commissioning speed in deg/s, at most {COMMISSIONING_MAX_SPEED_DEG_S:g}. Default: 30.",
    )
    robot.add_argument(
        "--robot-commissioning-excursion",
        type=float,
        default=80.0,
        help=f"Maximum offset from captured position in degrees, at most {COMMISSIONING_MAX_EXCURSION_DEG:g}. Default: 80.",
    )
    robot.add_argument(
        "--robot-commissioning-watchdog",
        type=float,
        default=0.15,
        help="Stop motion when jog refresh stops for this many seconds. Default: 0.15.",
    )
    robot.add_argument(
        "--robot-right-host",
        default=None,
        help="IP address of the UR7e driven by your right arm (--robot-backend ur).",
    )
    robot.add_argument(
        "--robot-left-host",
        default=None,
        help="IP address of the UR7e driven by your left arm (--robot-backend ur).",
    )
    robot.add_argument(
        "--robot-ur-port",
        type=int,
        default=UR_SECONDARY_PORT,
        help=f"URScript TCP port on the UR controller (30001 primary, 30002 secondary). Default: {UR_SECONDARY_PORT}.",
    )
    robot.add_argument(
        "--robot-rtde-port",
        type=int,
        default=UR_RTDE_PORT,
        help=f"RTDE feedback port on the UR controller. Default: {UR_RTDE_PORT}.",
    )
    robot.add_argument(
        "--robot-dashboard-port",
        type=int,
        default=UR_DASHBOARD_PORT,
        help=f"Dashboard server port used for the readiness check. Default: {UR_DASHBOARD_PORT}.",
    )
    robot.add_argument(
        "--no-robot-feedback",
        dest="robot_feedback",
        action="store_false",
        help="Do not read actual joint angles over RTDE; show commanded setpoints instead.",
    )
    robot.add_argument(
        "--no-robot-preflight",
        dest="robot_preflight",
        action="store_false",
        help="Skip the dashboard check for Remote Control, robot mode and safety status.",
    )
    robot.add_argument(
        "--robot-tool-output",
        type=int,
        default=0,
        help="Tool digital output driving the gripper. Default: 0.",
    )
    robot.add_argument(
        "--robot-servo-gain",
        type=int,
        default=300,
        help="servoj proportional gain, 100..2000. Default: 300.",
    )
    robot.add_argument(
        "--robot-servo-lookahead",
        type=float,
        default=0.1,
        help="servoj lookahead time in seconds, 0.03..0.2. Default: 0.1.",
    )
    robot.add_argument(
        "--robot-send-interval",
        type=float,
        default=0.05,
        help="Seconds between servoj / serial frames. Default: 0.05.",
    )
    robot.add_argument(
        "--robot-port",
        default=None,
        help="Serial port for --robot-backend serial, for example COM3.",
    )
    robot.add_argument(
        "--robot-baud",
        type=int,
        default=115200,
        help="Serial baud rate. Default: 115200.",
    )
    robot.add_argument(
        "--robot-max-speed",
        type=float,
        default=60.0,
        help=(
            "Maximum simulated joint speed in degrees per second. "
            f"UR7e hardware limit is {UR7E_MAX_JOINT_SPEED_DEG_S:.0f}. Default: 60."
        ),
    )
    robot.add_argument(
        "--robot-deadband",
        type=float,
        default=1.5,
        help="Ignore joint changes smaller than this many degrees. Default: 1.5.",
    )
    robot.add_argument(
        "--robot-shoulder-range",
        type=float,
        nargs=2,
        default=(
            DEFAULT_SHOULDER_MAPPING.limit.minimum,
            DEFAULT_SHOULDER_MAPPING.limit.maximum,
        ),
        metavar=("MIN", "MAX"),
        help="Allowed UR shoulder joint range in degrees. Default: -180 0.",
    )
    robot.add_argument(
        "--robot-elbow-range",
        type=float,
        nargs=2,
        default=(
            DEFAULT_ELBOW_MAPPING.limit.minimum,
            DEFAULT_ELBOW_MAPPING.limit.maximum,
        ),
        metavar=("MIN", "MAX"),
        help="Allowed UR elbow joint range in degrees (UR7e hardware: -160 160). Default: -160 160.",
    )
    robot.add_argument(
        "--robot-wrist-range",
        type=float,
        nargs=2,
        default=(
            DEFAULT_WRIST_MAPPING.limit.minimum,
            DEFAULT_WRIST_MAPPING.limit.maximum,
        ),
        metavar=("MIN", "MAX"),
        help="Allowed UR wrist 1 joint range in degrees. Default: -180 180.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> AppConfig:
    args = build_parser().parse_args(argv)

    backend = args.robot_backend
    if args.robot_debug and backend == BACKEND_NONE:
        backend = BACKEND_DEBUG
    if args.test_mode and backend == BACKEND_NONE:
        backend = BACKEND_SIM

    robot = RobotConfig(
        backend=backend,
        tracking_space=args.tracking_space or ("2d" if args.test_mode else "3d"),
        operation=args.robot_operation,
        print_interval=args.robot_print_interval,
        right_host=args.robot_right_host,
        left_host=args.robot_left_host,
        ur_port=args.robot_ur_port,
        rtde_port=args.robot_rtde_port,
        dashboard_port=args.robot_dashboard_port,
        feedback=args.robot_feedback,
        preflight=args.robot_preflight,
        tool_output=args.robot_tool_output,
        servo_gain=args.robot_servo_gain,
        servo_lookahead_s=args.robot_servo_lookahead,
        port=args.robot_port,
        baud_rate=args.robot_baud,
        send_interval=args.robot_send_interval,
        max_speed_deg_s=args.robot_max_speed,
        tracking_excursion_deg=args.robot_tracking_excursion,
        tracking_acceleration_deg_s2=args.robot_tracking_acceleration,
        tracking_loss_grace_s=args.robot_tracking_loss_grace,
        telemetry_log_path=args.robot_telemetry_log,
        feedback_log_path=args.robot_feedback_log,
        gripper_gesture_frames=args.robot_gripper_gesture_frames,
        gripper_driver=args.robot_gripper_driver,
        gripper_speed_percent=args.robot_gripper_speed,
        gripper_force_percent=args.robot_gripper_force,
        commissioning_joint=args.robot_commissioning_joint,
        commissioning_speed_deg_s=args.robot_commissioning_speed,
        commissioning_excursion_deg=args.robot_commissioning_excursion,
        commissioning_watchdog_s=args.robot_commissioning_watchdog,
        joint_deadband_deg=args.robot_deadband,
        shoulder=_with_limit(DEFAULT_SHOULDER_MAPPING, args.robot_shoulder_range),
        elbow=_with_limit(DEFAULT_ELBOW_MAPPING, args.robot_elbow_range),
        wrist=_with_limit(DEFAULT_WRIST_MAPPING, args.robot_wrist_range),
    )
    return AppConfig(
        camera=args.camera,
        camera_backend=args.camera_backend,
        camera_format=args.camera_format,
        video_path=args.video,
        loop_video=args.loop_video,
        robot=robot,
        test_mode=args.test_mode,
        width=args.width,
        height=args.height,
        camera_fps=args.fps,
        inference_width=args.inference_width,
        inference_height=args.inference_height,
        mirror=args.mirror,
        print_interval=args.print_interval,
        visibility_threshold=args.visibility_threshold,
        smoothing_alpha=args.smoothing_alpha,
        recording_dir=args.recording_dir,
        skeleton_path=args.skeleton,
        model_path=args.model,
        hand_model_path=args.hand_model,
        hands=args.hands,
        hand_tracking_interval=args.hand_tracking_interval,
        num_poses=args.num_poses,
        min_detection_confidence=args.min_detection_confidence,
        hand_detection_confidence=args.hand_detection_confidence,
        hand_presence_confidence=args.hand_presence_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )


def _with_limit(mapping: JointMapping, limits: tuple[float, float]) -> JointMapping:
    return JointMapping(
        source=mapping.source,
        offset_deg=mapping.offset_deg,
        sign=mapping.sign,
        limit=JointLimit(limits[0], limits[1]),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_cameras and args.diagnose_camera is not None:
        parser.error("--list-cameras and --diagnose-camera cannot be combined")
    if args.list_cameras:
        cameras = discover_cameras(load_camera_dependency())
        print(format_camera_list(cameras))
        return 0
    if args.diagnose_camera is not None:
        if args.video is not None:
            parser.error("--diagnose-camera cannot be combined with --video")
        settings = DiagnosticSettings(
            width=args.width, height=args.height, camera_fps=args.fps,
            camera_format=args.camera_format, camera_backend=args.camera_backend,
            duration_s=args.diagnostic_seconds, warmup_s=args.diagnostic_warmup,
        )
        try:
            settings.validate()
        except ValueError as error:
            parser.error(str(error))
        return diagnose_camera(load_camera_dependency(), args.diagnose_camera, settings=settings)
    config = parse_args()
    config.validate()
    return run_app(config)
