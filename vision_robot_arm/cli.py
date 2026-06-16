import argparse
from pathlib import Path

from vision_robot_arm.app import run_app
from vision_robot_arm.config import DEFAULT_MODEL_PATH, DEFAULT_RECORDING_DIR, AppConfig


def parse_args() -> AppConfig:
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
        "--robot-debug",
        action="store_true",
        help="Print mapped robot commands without controlling real hardware.",
    )
    parser.add_argument(
        "--robot-print-interval",
        type=float,
        default=0.5,
        help="Seconds between robot debug command prints. Default: 0.5.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional requested camera width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional requested camera height in pixels.",
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

    args = parser.parse_args()
    return AppConfig(
        camera=args.camera,
        video_path=args.video,
        loop_video=args.loop_video,
        robot_debug=args.robot_debug,
        robot_print_interval=args.robot_print_interval,
        width=args.width,
        height=args.height,
        print_interval=args.print_interval,
        visibility_threshold=args.visibility_threshold,
        smoothing_alpha=args.smoothing_alpha,
        recording_dir=args.recording_dir,
        model_path=args.model,
        num_poses=args.num_poses,
        min_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )


def main() -> int:
    config = parse_args()
    config.validate()
    return run_app(config)
