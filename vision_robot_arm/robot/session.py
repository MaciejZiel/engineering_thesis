"""Explicit, fail-closed hardware session; preview never authorizes motion."""

import math
import time
from typing import Callable

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.config import (
    OPERATION_COMMISSIONING,
    OPERATION_KEYFRAME,
    OPERATION_MONITOR,
    OPERATION_TRACKING,
    RobotConfig,
)
from vision_robot_arm.robot.mapping import RobotMapper
from vision_robot_arm.robot.targets import ArmTargets, JointTargets, MAPPED_JOINTS


class HardwareSession:
    def __init__(self, config: RobotConfig, factory: Callable, clock: Callable[[], float] = time.monotonic):
        self._config = config
        self._factory = factory
        self._backend = None
        self._mapper = RobotMapper(config)
        self.phase = "disconnected"
        self.error: str | None = None
        self._usable = False
        self._clock = clock
        self._last_targets = None
        self._tracking_lost_at: float | None = None
        self._latest_targets: JointTargets | None = None
        self._keyframe_start: JointTargets | None = None
        self._keyframe_robot_start: dict[str, dict[str, float]] = {}
        self._keyframe_final: JointTargets | None = None

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
                    self.phase = (
                        "keyframe_start"
                        if self._config.operation == OPERATION_KEYFRAME
                        else "ready"
                    )
            elif self.phase == "keyframe_start" and self._usable:
                self._capture_keyframe_start()
            elif self.phase == "keyframe_end" and self._usable:
                self._capture_keyframe_end()
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
        if self.phase not in ("active", "commissioning"):
            return
        try:
            self._backend.pause()
            self.phase = (
                "connected"
                if self.phase == "commissioning"
                else "paused"
            )
            self._mapper.reset()
            self._tracking_lost_at = None
            self._last_targets = None
        except (Exception, SystemExit) as error:
            self._fail(error)

    def update(self, state: PoseState) -> None:
        if self.phase in ("monitoring", "commissioning"):
            return
        targets = self._mapper.map(state)
        self._latest_targets = targets
        self._usable = all(
            all(joint in targets.arm(side).joints
                and math.isfinite(targets.arm(side).joints[joint]) for joint in MAPPED_JOINTS)
            for side in self._config.hosts
        )
        try:
            if self.phase == "active":
                if self._config.operation == OPERATION_KEYFRAME:
                    if self._keyframe_final is not None:
                        self._backend.send(self._keyframe_final)
                    return
                if not self._usable:
                    self.tracking_lost()
                else:
                    if self._tracking_lost_at is not None:
                        self._note_tracking("tracking_recovered", self._clock() - self._tracking_lost_at)
                    self._tracking_lost_at = None
                    self._last_targets = targets
                    self._backend.send(targets)
        except (Exception, SystemExit) as error:
            self._fail(error)

    def reset(self) -> None:
        if self._config.operation not in (OPERATION_TRACKING, OPERATION_KEYFRAME):
            return
        self._usable = False
        self._tracking_lost_at = None
        self._last_targets = None
        self._mapper.reset()
        self.pause()

    def tracking_lost(self) -> None:
        """Bridge short vision gaps, then fail closed if tracking does not recover."""
        if self.phase != "active":
            return
        if self._config.operation == OPERATION_KEYFRAME and self._keyframe_final is not None:
            try:
                self._backend.send(self._keyframe_final)
            except (Exception, SystemExit) as error:
                self._fail(error)
            return
        now = self._clock()
        if self._tracking_lost_at is None:
            self._tracking_lost_at = now
        elapsed = now - self._tracking_lost_at
        if (
            self._last_targets is not None
            and elapsed <= self._config.tracking_loss_grace_s
        ):
            self._note_tracking("tracking_gap_held", elapsed)
            try:
                self._backend.send(self._last_targets)
            except (Exception, SystemExit) as error:
                self._fail(error)
            return
        self._note_tracking("tracking_gap_stopped", elapsed)
        self.pause()

    def _note_tracking(self, event: str, elapsed_s: float) -> None:
        callback = getattr(self._backend, "note_tracking_event", None)
        if callback is not None:
            callback(event, elapsed_s)

    def _capture_keyframe_start(self) -> None:
        if self._latest_targets is None or not self._backend.ready():
            return
        robot_state = self._backend.robot_state()
        if robot_state is None:
            return
        starts: dict[str, dict[str, float]] = {}
        for side in self._config.hosts:
            arm = robot_state.arm(side)
            if arm is None or not all(joint in arm.joints for joint in MAPPED_JOINTS):
                return
            starts[side] = dict(arm.joints)
        self._keyframe_start = self._latest_targets
        self._keyframe_robot_start = starts
        self.phase = "keyframe_end"

    def _capture_keyframe_end(self) -> None:
        if self._latest_targets is None or self._keyframe_start is None:
            return
        arms: dict[str, ArmTargets] = {}
        for side in self._config.hosts:
            start = self._keyframe_start.arm(side)
            end = self._latest_targets.arm(side)
            origin = self._keyframe_robot_start[side]
            joints = {
                joint: origin[joint] + (
                    -1.0 if joint == "shoulder" else 1.0
                ) * (end.joints[joint] - start.joints[joint])
                for joint in MAPPED_JOINTS
            }
            arms[side] = ArmTargets(joints=joints, gripper=end.gripper)
        self._keyframe_final = JointTargets(
            timestamp_ms=self._latest_targets.timestamp_ms,
            arms=arms,
        )
        self._last_targets = self._keyframe_final
        self.phase = "active"

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
            "ready": "Enable control",
            "keyframe_start": "Capture start frame",
            "keyframe_end": "Capture end frame and move",
            "active": "Pause control",
            "paused": "Resume control",
            "fault": "Fault — restart required",
        }[self.phase]

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
