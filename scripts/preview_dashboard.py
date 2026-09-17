"""Render repeatable dashboard states without opening a camera or robot backend.

Run with .venv/bin/python scripts/preview_dashboard.py
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_robot_arm.robot.targets import ArmState, RobotState, full_joint_pose
from vision_robot_arm.robot.visualization_3d import draw_workspace_3d
from vision_robot_arm.vision.dashboard import DashboardUi
from vision_robot_arm.vision.ui_style import ACCENT, MUTED, Painter


def sample_camera() -> np.ndarray:
    """An explicitly labelled sample pose, not a captured camera image."""
    frame = np.full((720, 1280, 3), (43, 41, 38), np.uint8)
    cv2.line(frame, (0, 610), (1280, 610), (55, 53, 49), 1)
    neutral = (138, 133, 124)
    for start, end in (
        ((565, 230), (715, 230)),
        ((640, 230), (640, 430)),
        ((580, 430), (700, 430)),
        ((580, 430), (555, 540)),
        ((555, 540), (540, 640)),
        ((700, 430), (725, 540)),
        ((725, 540), (740, 640)),
        ((640, 170), (640, 230)),
    ):
        cv2.line(frame, start, end, neutral, 3, cv2.LINE_AA)
    cv2.circle(frame, (640, 142), 30, neutral, 2, cv2.LINE_AA)
    for points in (
        ((565, 230), (475, 320), (385, 250)),
        ((715, 230), (800, 305), (885, 210)),
    ):
        for start, end in pairwise(points):
            cv2.line(frame, start, end, ACCENT, 4, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 7, ACCENT, -1, cv2.LINE_AA)
    Painter(cv2, np, frame).text(
        "Sample pose · camera preview", 24, 24, size=15, color=MUTED
    )
    return frame


def render_previews(output: Path, sizes: list[tuple[int, int]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    camera = sample_camera()
    joints = full_joint_pose({"shoulder": -35, "elbow": 70, "wrist_1": -60})
    targets = full_joint_pose({"shoulder": -42, "elbow": 80, "wrist_1": -55})
    state = RobotState(
        {
            "left": ArmState(joints, targets, "open"),
            "right": ArmState(joints, targets, "close"),
        },
        False,
    )
    for width, height in sizes:
        ui = DashboardUi(cv2, np, "Offline preview")
        ui._canvas_size = (width, height)
        # Render the workspace at the size the panel really gets, as the app does;
        # rendering once and letting the dashboard downscale hides every detail flaw.
        panel_width, panel_height = ui.simulation_target_size()
        sim = np.zeros((panel_height, panel_width, 3), np.uint8)
        draw_workspace_3d(cv2, np, sim, state, None, {})
        for name in ("tracking", "waiting", "recording", "details"):
            ui._details = name == "details"
            detected = name != "waiting"
            frame = ui.render(
                camera if detected else np.full_like(camera, (43, 41, 38)),
                sim,
                mode="angles",
                person_detected=detected,
                calibrated=name == "recording",
                recording=name == "recording",
                robot_label="sim",
                gestures=("right_fist", "left_hand_open") if detected else (),
                status_lines=(
                    "sim R: S -35.0 -> -42.0 E 70.0 -> 80.0 W1 -60.0 -> -55.0 grip close",
                    "sim L: S -35.0 -> -42.0 E 70.0 -> 80.0 W1 -60.0 -> -55.0 grip open",
                    "sim lift_mode=off",
                ),
                tracking_quality=0.91 if detected else 0,
                fps=29.8,
                source_label="Camera 0",
            )
            path = output / f"{name}-{width}x{height}.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Could not save {path}")
            print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ui-review"))
    parser.add_argument(
        "--size", nargs=2, type=int, action="append", metavar=("WIDTH", "HEIGHT")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    render_previews(
        args.output, args.size or [(960, 540), (1440, 900), (1920, 1080), (3440, 1440)]
    )
    print(f"Rendered in {time.perf_counter() - started:.2f}s")
