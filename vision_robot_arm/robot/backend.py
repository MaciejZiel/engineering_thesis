import time
from typing import Callable, Protocol

from vision_robot_arm.robot.targets import (
    ARM_NAMES,
    GRIPPER_OPEN,
    ArmState,
    JointTargets,
    RobotState,
)

Clock = Callable[[], float]


class RobotBackend(Protocol):
    def send(self, targets: JointTargets) -> None:
        ...

    def robot_state(self) -> RobotState | None:
        ...

    def status_lines(self) -> list[str]:
        ...

    def close(self) -> None:
        ...


class TargetTracker:
    def __init__(self) -> None:
        self._joints: dict[str, dict[str, float]] = {arm: {} for arm in ARM_NAMES}
        self._grippers: dict[str, str] = {arm: GRIPPER_OPEN for arm in ARM_NAMES}
        self._lift_mode = False
        self.last_targets: JointTargets | None = None

    def update(self, targets: JointTargets) -> None:
        self.last_targets = targets
        for arm, arm_targets in targets.arms.items():
            self._joints.setdefault(arm, {}).update(arm_targets.joints)
            if arm_targets.gripper is not None:
                self._grippers[arm] = arm_targets.gripper
        self._lift_mode = targets.lift_mode

    def robot_state(self) -> RobotState | None:
        if self.last_targets is None:
            return None
        arms = {
            arm: ArmState(
                joints=dict(joints),
                targets=dict(joints),
                gripper=self._grippers.get(arm, GRIPPER_OPEN),
            )
            for arm, joints in self._joints.items()
        }
        return RobotState(arms=arms, lift_mode=self._lift_mode)


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

    def robot_state(self) -> RobotState | None:
        return self._tracker.robot_state()

    def status_lines(self) -> list[str]:
        last = self._tracker.last_targets
        if last is None:
            return ["robot debug: waiting for pose"]
        return [f"robot debug: {last.format()}"]

    def close(self) -> None:
        return
