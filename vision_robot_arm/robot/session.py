"""Explicit, fail-closed hardware session; preview never authorizes motion."""

import math
from typing import Callable

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.config import (
    OPERATION_COMMISSIONING,
    OPERATION_MONITOR,
    OPERATION_TRACKING,
    RobotConfig,
)
from vision_robot_arm.robot.mapping import RobotMapper
from vision_robot_arm.robot.targets import MAPPED_JOINTS


class HardwareSession:
    def __init__(self, config: RobotConfig, factory: Callable):
        self._config = config
        self._factory = factory
        self._backend = None
        self._mapper = RobotMapper(config)
        self.phase = "disconnected"
        self.error: str | None = None
        self._usable = False

    def _fail(self, error: BaseException) -> None:
        self.error = str(error)
        self.phase = "fault"
        if self._backend is not None:
            self._backend.close()

    def advance(self) -> None:
        """Each step requires a separate operator action; faults cannot auto-resume."""
        try:
            if self.phase == "disconnected":
                self._backend = self._factory()
                self.phase = (
                    "monitoring"
                    if self._config.operation == OPERATION_MONITOR
                    else "connected"
                )
            elif self.phase == "connected":
                if self._config.operation == OPERATION_COMMISSIONING:
                    self._backend.arm_commissioning()
                    self.phase = "commissioning"
                else:
                    self._backend.arm_tracking()
                    self.phase = "ready"
            elif self.phase in ("ready", "paused") and self._usable:
                if self._backend.ready():
                    self._mapper.reset()
                    self.phase = "active"
            elif self.phase == "active":
                self.pause()
            elif self.phase == "commissioning":
                self.pause()
        except (Exception, SystemExit) as error:
            self._fail(error)

    def pause(self) -> None:
        if self.phase not in ("active", "homing", "commissioning"):
            return
        try:
            self._backend.pause()
            # Interrupted homing must be performed again, not counted as complete.
            self.phase = (
                "connected"
                if self.phase in ("homing", "commissioning")
                else "paused"
            )
            self._mapper.reset()
        except (Exception, SystemExit) as error:
            self._fail(error)

    def update(self, state: PoseState) -> None:
        if self.phase in ("monitoring", "commissioning"):
            return
        targets = self._mapper.map(state)
        self._usable = all(
            all(joint in targets.arm(side).joints
                and math.isfinite(targets.arm(side).joints[joint]) for joint in MAPPED_JOINTS)
            for side in self._config.hosts
        )
        try:
            if self.phase == "homing" and self._backend.ready():
                self.phase = "ready"
            if self.phase == "active":
                if not self._usable:
                    self.pause()
                else:
                    self._backend.send(targets)
        except (Exception, SystemExit) as error:
            self._fail(error)

    def reset(self) -> None:
        if self._config.operation != OPERATION_TRACKING:
            return
        self._usable = False
        self._mapper.reset()
        self.pause()

    def jog(self, direction: int) -> None:
        if self.phase != "commissioning":
            return
        try:
            self._backend.refresh_jog(direction)
        except (Exception, SystemExit) as error:
            self._fail(error)

    def robot_state(self):
        if self._backend is None or self.phase == "fault":
            return None
        try:
            if self.phase == "homing" and self._backend.ready():
                self.phase = "ready"
            return self._backend.robot_state()
        except (Exception, SystemExit) as error:
            self._fail(error)
            return None

    def status_lines(self) -> list[str]:
        lines = [f"Hardware: {self.phase}. {self.action_label}"]
        if self.error:
            lines.append(self.error)
        elif self._backend is not None:
            try:
                lines.extend(self._backend.status_lines())
            except (Exception, SystemExit) as error:
                self._fail(error)
                lines.append(str(error))
        return lines

    @property
    def action_label(self) -> str:
        return {
            "disconnected": "Connect robot",
            "monitoring": "Read-only monitoring",
            "commissioning": "Disarm commissioning",
            "connected": "Capture current pose (no motion)",
            "homing": "Homing — P to pause",
            "ready": "Enable control",
            "active": "Pause control",
            "paused": "Resume control",
            "fault": "Fault — restart required",
        }[self.phase]

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
