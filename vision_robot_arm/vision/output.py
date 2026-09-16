import sys
import time
from typing import Any

from vision_robot_arm.core.config import ANGLE_MODE, BOTH_MODE, LANDMARK_MODE
from vision_robot_arm.vision.metrics import format_angles
from vision_robot_arm.core.pose_state import PoseState


def print_landmarks(
    landmarks: list[Any],
    world_landmarks: list[Any] | None,
    names: dict[int, str],
) -> None:
    for index, landmark in enumerate(landmarks):
        name = names.get(index, str(index))
        line = (
            f"  {name:20s} "
            f"x={landmark.x: .3f} y={landmark.y: .3f} "
            f"z={landmark.z: .3f} visibility={landmark.visibility: .2f}"
        )
        if world_landmarks and index < len(world_landmarks):
            world = world_landmarks[index]
            line += (
                f" | world_x={world.x: .3f} "
                f"world_y={world.y: .3f} world_z={world.z: .3f}"
            )
        print(line)


def emit_console_data(mode: str, state: PoseState, names: dict[int, str]) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"\n[{timestamp}] mode={mode}")
    if state.gestures:
        print("gestures " + ", ".join(state.gestures))

    if mode in (ANGLE_MODE, BOTH_MODE):
        print("angles   " + format_angles(state.angles))
        if state.calibrated:
            print("relative " + format_angles(state.relative_angles))

    if mode in (LANDMARK_MODE, BOTH_MODE):
        print_landmarks(state.landmarks, state.world_landmarks, names)

    sys.stdout.flush()
