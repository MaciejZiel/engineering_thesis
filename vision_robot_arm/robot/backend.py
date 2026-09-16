import time
from typing import Callable, Protocol

from vision_robot_arm.robot.targets import JointTargets

Clock = Callable[[], float]


class RobotBackend(Protocol):
    def send(self, targets: JointTargets) -> None:
        ...

    def status_lines(self) -> list[str]:
        ...

    def close(self) -> None:
        ...


class DebugBackend:
    def __init__(self, print_interval: float, clock: Clock = time.monotonic) -> None:
        self._print_interval = print_interval
        self._clock = clock
        self._next_print_at = 0.0
        self._last_targets: JointTargets | None = None

    def send(self, targets: JointTargets) -> None:
        self._last_targets = targets
        now = self._clock()
        if now < self._next_print_at:
            return
        if targets.has_data:
            print(f"robot {targets.format()}")
        self._next_print_at = now + self._print_interval

    def status_lines(self) -> list[str]:
        if self._last_targets is None:
            return ["robot debug: waiting for pose"]
        return [f"robot debug: {self._last_targets.format()}"]

    def close(self) -> None:
        return
