import time
from typing import Callable, Protocol

from vision_robot_arm.robot.targets import GRIPPER_OPEN, ArmState, JointTargets

Clock = Callable[[], float]


class RobotBackend(Protocol):
    def send(self, targets: JointTargets) -> None:
        ...

    def arm_state(self) -> ArmState | None:
        ...

    def status_lines(self) -> list[str]:
        ...

    def close(self) -> None:
        ...


class TargetTracker:
    def __init__(self) -> None:
        self._joints: dict[str, float] = {}
        self._gripper = GRIPPER_OPEN
        self._lift_mode = False
        self.last_targets: JointTargets | None = None

    def update(self, targets: JointTargets) -> None:
        self.last_targets = targets
        self._joints.update(targets.joints)
        if targets.gripper is not None:
            self._gripper = targets.gripper
        self._lift_mode = targets.lift_mode

    def arm_state(self) -> ArmState | None:
        if self.last_targets is None:
            return None
        return ArmState(
            joints=dict(self._joints),
            targets=dict(self._joints),
            gripper=self._gripper,
            lift_mode=self._lift_mode,
        )


class DebugBackend:
    def __init__(self, print_interval: float, clock: Clock = time.monotonic) -> None:
        self._print_interval = print_interval
        self._clock = clock
        self._next_print_at = 0.0
        self._tracker = TargetTracker()

    def send(self, targets: JointTargets) -> None:
        self._tracker.update(targets)
        now = self._clock()
        if now < self._next_print_at:
            return
        if targets.has_data:
            print(f"robot {targets.format()}")
        self._next_print_at = now + self._print_interval

    def arm_state(self) -> ArmState | None:
        return self._tracker.arm_state()

    def status_lines(self) -> list[str]:
        last = self._tracker.last_targets
        if last is None:
            return ["robot debug: waiting for pose"]
        return [f"robot debug: {last.format()}"]

    def close(self) -> None:
        return
