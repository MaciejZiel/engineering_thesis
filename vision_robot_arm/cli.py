import argparse
from pathlib import Path

from vision_robot_arm.app import run_app
from vision_robot_arm.core.config import (
    DEFAULT_HAND_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RECORDING_DIR,
    AppConfig,
)
from vision_robot_arm.robot.config import (
    BACKEND_CHOICES,
    BACKEND_DEBUG,
    BACKEND_NONE,
    BACKEND_SIM,
    JointLimit,
    RobotConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Webcam pose tracker for the vision robot arm prototype."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0.",
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
            "Show joint angles next to the arm joints and draw the robot arm panel on the "
            "camera image. Uses the simulated robot when no --robot-backend is given."
        ),
    )

    robot = parser.add_argument_group("robot")
    robot.add_argument(
        "--robot-backend",
        choices=BACKEND_CHOICES,
        default=BACKEND_NONE,
        help=(
            "Where mapped robot commands go: none, debug (print), sim (simulated arm) "
            "or serial (hardware over a serial port). Default: none."
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
        "--robot-send-interval",
        type=float,
        default=0.05,
        help="Minimum seconds between serial frames. Default: 0.05.",
    )
    robot.add_argument(
        "--robot-max-speed",
        type=float,
        default=90.0,
        help="Maximum simulated joint speed in degrees per second. Default: 90.",
    )
    robot.add_argument(
        "--robot-home",
        type=float,
        default=90.0,
        help="Starting joint angle of the simulated arm in degrees. Default: 90.",
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
        default=(0.0, 180.0),
        metavar=("MIN", "MAX"),
        help="Allowed shoulder joint range in degrees. Default: 0 180.",
    )
    robot.add_argument(
        "--robot-elbow-range",
        type=float,
        nargs=2,
        default=(0.0, 180.0),
        metavar=("MIN", "MAX"),
        help="Allowed elbow joint range in degrees. Default: 0 180.",
    )
    robot.add_argument(
        "--robot-wrist-range",
        type=float,
        nargs=2,
        default=(0.0, 180.0),
        metavar=("MIN", "MAX"),
        help="Allowed wrist joint range in degrees. Default: 0 180.",
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
        print_interval=args.robot_print_interval,
        port=args.robot_port,
        baud_rate=args.robot_baud,
        send_interval=args.robot_send_interval,
        max_speed_deg_s=args.robot_max_speed,
        home_deg=args.robot_home,
        joint_deadband_deg=args.robot_deadband,
        shoulder_limit=JointLimit(*args.robot_shoulder_range),
        elbow_limit=JointLimit(*args.robot_elbow_range),
        wrist_limit=JointLimit(*args.robot_wrist_range),
    )
    return AppConfig(
        camera=args.camera,
        video_path=args.video,
        loop_video=args.loop_video,
        robot=robot,
        test_mode=args.test_mode,
        width=args.width,
        height=args.height,
        mirror=args.mirror,
        print_interval=args.print_interval,
        visibility_threshold=args.visibility_threshold,
        smoothing_alpha=args.smoothing_alpha,
        recording_dir=args.recording_dir,
        model_path=args.model,
        hand_model_path=args.hand_model,
        hands=args.hands,
        num_poses=args.num_poses,
        min_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )


def main() -> int:
    config = parse_args()
    config.validate()
    return run_app(config)
