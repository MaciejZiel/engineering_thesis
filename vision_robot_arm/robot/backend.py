import time
from typing import Callable, Protocol

from vision_robot_arm.robot.targets import (
    ARM_NAMES,
    GRIPPER_OPEN,
    ArmState,
    ArmTargets,
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
        self._commanded: set[str] = set()
        self._lift_mode = False
        self.last_targets: JointTargets | None = None

    def update(self, targets: JointTargets) -> None:
        self.last_targets = targets
        for arm, arm_targets in targets.arms.items():
            self._joints.setdefault(arm, {}).update(arm_targets.joints)
            if arm_targets.gripper is not None:
                self._grippers[arm] = arm_targets.gripper
                self._commanded.add(arm)
        self._lift_mode = targets.lift_mode

    def commanded_gripper(self, arm: str) -> str | None:
        """The latest gripper command for this arm, or None if none was ever given."""
        return self._grippers[arm] if arm in self._commanded else None

    def accumulated_targets(self) -> JointTargets | None:
        """Everything seen so far as one frame, for protocols that send state, not events."""
        if self.last_targets is None:
            return None
        return JointTargets(
            timestamp_ms=self.last_targets.timestamp_ms,
            arms={
                arm: ArmTargets(joints=dict(joints), gripper=self.commanded_gripper(arm))
                for arm, joints in self._joints.items()
            },
            lift_mode=self._lift_mode,
        )

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

    def commanded_gripper(self, arm: str) -> str | None:
        """The latest gripper command for this arm, or None if none was ever given."""
        return self._tracker.commanded_gripper(arm)

    def accumulated_targets(self) -> JointTargets | None:
        """Everything seen so far as one frame, for protocols that send state, not events."""
        return self._tracker.accumulated_targets()

    def robot_state(self) -> RobotState | None:
        return self._tracker.robot_state()

    def status_lines(self) -> list[str]:
        last = self._tracker.last_targets
        if last is None:
            return ["robot debug: waiting for pose"]
        return [f"robot debug: {last.format()}"]

    def close(self) -> None:
        return
