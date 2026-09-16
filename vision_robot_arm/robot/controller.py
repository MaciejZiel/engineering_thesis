import time
from dataclasses import dataclass
from typing import Protocol

from vision_robot_arm.core.pose_state import PoseState


@dataclass(frozen=True)
class RobotCommand:
    target: str
    value: float | str
    source: str

    def format(self) -> str:
        if isinstance(self.value, float):
            return f"{self.target}={self.value:5.1f} ({self.source})"
        return f"{self.target}={self.value} ({self.source})"


class RobotController(Protocol):
    def update(self, state: PoseState) -> None:
        ...

    def close(self) -> None:
        ...


class NullRobotController:
    def update(self, state: PoseState) -> None:
        return

    def close(self) -> None:
        return


class DebugRobotController:
    def __init__(self, print_interval: float) -> None:
        self._print_interval = print_interval
        self._next_print_at = 0.0

    def update(self, state: PoseState) -> None:
        now = time.monotonic()
        if now < self._next_print_at:
            return

        commands = map_pose_to_robot_commands(state)
        if commands:
            formatted = " | ".join(command.format() for command in commands)
            print(f"robot {formatted}")
        self._next_print_at = now + self._print_interval

    def close(self) -> None:
        return


def create_robot_controller(
    enabled: bool,
    print_interval: float,
) -> RobotController:
    if enabled:
        return DebugRobotController(print_interval=print_interval)
    return NullRobotController()


def map_pose_to_robot_commands(state: PoseState) -> list[RobotCommand]:
    commands: list[RobotCommand] = []

    right_shoulder = state.angles.get("right_shoulder")
    if right_shoulder is not None:
        commands.append(
            RobotCommand(
                target="shoulder",
                value=_clamp(right_shoulder, 0.0, 180.0),
                source="right_shoulder",
            )
        )

    right_elbow = state.angles.get("right_elbow")
    if right_elbow is not None:
        commands.append(
            RobotCommand(
                target="elbow",
                value=_clamp(right_elbow, 0.0, 180.0),
                source="right_elbow",
            )
        )

    if "right_hand_up" in state.gestures:
        commands.append(RobotCommand("lift_mode", "on", "right_hand_up"))
    if "right_elbow_bent" in state.gestures:
        commands.append(RobotCommand("gripper", "close", "right_elbow_bent"))
    elif "right_arm_side" in state.gestures:
        commands.append(RobotCommand("gripper", "open", "right_arm_side"))

    return commands


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
