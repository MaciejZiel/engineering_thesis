import math
import time

from vision_robot_arm.robot.backend import Clock
from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.targets import GRIPPER_OPEN, JOINT_NAMES, ArmState, JointTargets


class SimulationBackend:
    def __init__(self, config: RobotConfig, clock: Clock = time.monotonic) -> None:
        self._config = config
        self._clock = clock
        self._joints = {
            name: config.limit_for(name).clamp(config.home_deg) for name in JOINT_NAMES
        }
        self._targets = dict(self._joints)
        self._gripper = GRIPPER_OPEN
        self._lift_mode = False
        self._last_time: float | None = None

    @property
    def state(self) -> ArmState:
        return ArmState(
            joints=dict(self._joints),
            targets=dict(self._targets),
            gripper=self._gripper,
            lift_mode=self._lift_mode,
        )

    def send(self, targets: JointTargets) -> None:
        for joint, value in targets.joints.items():
            if joint in self._targets:
                self._targets[joint] = self._config.limit_for(joint).clamp(value)
        if targets.gripper is not None:
            self._gripper = targets.gripper
        self._lift_mode = targets.lift_mode

        now = self._clock()
        dt = 0.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        self.step(dt)

    def step(self, dt: float) -> ArmState:
        max_delta = self._config.max_speed_deg_s * max(dt, 0.0)
        for joint, target in self._targets.items():
            current = self._joints[joint]
            delta = target - current
            if abs(delta) <= max_delta:
                self._joints[joint] = target
            else:
                self._joints[joint] = current + math.copysign(max_delta, delta)
        return self.state

    def arm_state(self) -> ArmState:
        return self.state

    def status_lines(self) -> list[str]:
        lines = [
            f"sim {joint}={self._joints[joint]:5.1f} -> {self._targets[joint]:5.1f}"
            for joint in JOINT_NAMES
        ]
        lift = "on" if self._lift_mode else "off"
        lines.append(f"sim gripper={self._gripper} | lift_mode={lift}")
        return lines

    def close(self) -> None:
        return
