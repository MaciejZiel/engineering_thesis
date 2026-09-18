import math
import time

from vision_robot_arm.robot.backend import Clock
from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.targets import (
    ARM_NAMES,
    GRIPPER_OPEN,
    JOINT_NAMES,
    REPORTED_JOINTS,
    ArmState,
    JointTargets,
    RobotState,
)

# A tracking gap must not buy a huge integration step; the twin would then show
# motion that the speed limit forbids.
MAX_STEP_SECONDS = 0.2

JOINT_SHORT_NAMES = {
    "base": "B",
    "shoulder": "S",
    "elbow": "E",
    "wrist_1": "W1",
    "wrist_2": "W2",
    "wrist_3": "W3",
}


class SimulatedArm:
    def __init__(self, config: RobotConfig) -> None:
        self._config = config
        self.joints = {name: config.home_for(name) for name in JOINT_NAMES}
        self.targets = dict(self.joints)
        self.gripper = GRIPPER_OPEN
        self.tcp_target: tuple[float, float, float] | None = None

    def set_targets(
        self,
        joints: dict[str, float],
        gripper: str | None,
        tcp_target: tuple[float, float, float] | None = None,
    ) -> None:
        for joint, value in joints.items():
            if joint in self.targets:
                self.targets[joint] = self._config.limit_for(joint).clamp(value)
        if gripper is not None:
            self.gripper = gripper
        if tcp_target is not None:
            self.tcp_target = tcp_target

    def step(self, max_delta: float) -> None:
        for joint, target in self.targets.items():
            current = self.joints[joint]
            delta = target - current
            if abs(delta) <= max_delta:
                self.joints[joint] = target
            else:
                self.joints[joint] = current + math.copysign(max_delta, delta)

    @property
    def state(self) -> ArmState:
        return ArmState(
            joints=dict(self.joints),
            targets=dict(self.targets),
            gripper=self.gripper,
            tcp_target=self.tcp_target,
        )


class SimulationBackend:
    def __init__(self, config: RobotConfig, clock: Clock = time.monotonic) -> None:
        self._config = config
        self._clock = clock
        self._arms = {name: SimulatedArm(config) for name in ARM_NAMES}
        self._lift_mode = False
        self._last_time: float | None = None

    @property
    def state(self) -> RobotState:
        return RobotState(
            arms={name: arm.state for name, arm in self._arms.items()},
            lift_mode=self._lift_mode,
        )

    def send(self, targets: JointTargets) -> None:
        for name, arm_targets in targets.arms.items():
            arm = self._arms.get(name)
            if arm is not None:
                arm.set_targets(
                    arm_targets.joints,
                    arm_targets.gripper,
                    arm_targets.tcp_target,
                )
        self._lift_mode = targets.lift_mode

        now = self._clock()
        dt = 0.0 if self._last_time is None else min(now - self._last_time, MAX_STEP_SECONDS)
        self._last_time = now
        self.step(dt)

    def step(self, dt: float) -> RobotState:
        max_delta = self._config.max_speed_deg_s * max(dt, 0.0)
        for arm in self._arms.values():
            arm.step(max_delta)
        return self.state

    def robot_state(self) -> RobotState:
        return self.state

    def status_lines(self) -> list[str]:
        lines = []
        for name, arm in self._arms.items():
            joints = " ".join(
                f"{JOINT_SHORT_NAMES[joint]} {arm.joints[joint]:6.1f}->{arm.targets[joint]:6.1f}"
                for joint in REPORTED_JOINTS
            )
            lines.append(f"sim {name[0].upper()}: {joints} grip {arm.gripper}")
        lines.append(f"sim lift_mode={'on' if self._lift_mode else 'off'}")
        return lines

    def close(self) -> None:
        return
