import sys
import time
from typing import Any

from vision_robot_arm.config import ANGLE_MODE, BOTH_MODE, LANDMARK_MODE
from vision_robot_arm.metrics import calculate_angles, format_angles


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


def emit_console_data(
    mode: str,
    landmarks: list[Any],
    world_landmarks: list[Any] | None,
    indices: dict[str, int],
    names: dict[int, str],
    min_visibility: float,
) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"\n[{timestamp}] mode={mode}")

    if mode in (ANGLE_MODE, BOTH_MODE):
        angles = calculate_angles(landmarks, indices, min_visibility)
        print(format_angles(angles))

    if mode in (LANDMARK_MODE, BOTH_MODE):
        print_landmarks(landmarks, world_landmarks, names)

    sys.stdout.flush()
