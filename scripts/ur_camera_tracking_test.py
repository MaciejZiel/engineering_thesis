"""Launch the repeatable one-arm live-camera UR tracking preset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_robot_arm.cli import main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-camera, one-UR tracking test with telemetry."
    )
    parser.add_argument("--host", default="10.20.3.20")
    parser.add_argument("--camera", default="auto")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    sys.argv = [
        sys.argv[0],
        "--camera", args.camera,
        "--width", "1280",
        "--height", "720",
        "--fps", "30",
        "--inference-width", "640",
        "--inference-height", "360",
        "--test-mode",
        "--robot-backend", "ur",
        "--robot-operation", "tracking",
        "--robot-right-host", args.host,
        "--robot-max-speed", "20",
        "--robot-tracking-excursion", "40",
        "--robot-tracking-acceleration", "7",
        "--robot-tracking-loss-grace", "0.4",
        "--robot-deadband", "2",
        "--robot-gripper-driver", "robotiq",
        "--robot-gripper-speed", "80",
        "--robot-gripper-force", "50",
        "--robot-gripper-gesture-frames", "3",
        "--robot-telemetry-log", "logs/ur_tracking.jsonl",
        "--robot-feedback-log", "logs/ur_robot_feedback.jsonl",
    ]
    return main()


if __name__ == "__main__":
    raise SystemExit(run())
