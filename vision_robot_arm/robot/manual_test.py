"""Standalone, camera-free session for cautious single-joint UR7e checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vision_robot_arm.robot.config import (
    BACKEND_UR,
    OPERATION_COMMISSIONING,
    OPERATION_MONITOR,
    RobotConfig,
)
from vision_robot_arm.robot.targets import JOINT_NAMES, RobotState
from vision_robot_arm.robot.ur_backend import URBackend
from vision_robot_arm.robot.ur_monitor import URMonitorBackend

BackendFactory = Callable[[RobotConfig], object]
@dataclass(frozen=True)
class ManualTestSettings:
    host: str
    side: str = "right"
    joint: str = "shoulder"
    speed_deg_s: float = 30.0
    excursion_deg: float = 80.0

    def config(self, operation: str) -> RobotConfig:
        host = self.host.strip()
        if not host:
            raise ValueError("Enter the robot IP address.")
        if self.side not in ("left", "right"):
            raise ValueError("Robot side must be left or right.")
        if self.joint not in JOINT_NAMES:
            raise ValueError(f"Unknown joint: {self.joint}")
        kwargs = {f"{self.side}_host": host}
        config = RobotConfig(
            backend=BACKEND_UR,
            operation=operation,
            commissioning_joint=self.joint,
            commissioning_speed_deg_s=self.speed_deg_s,
            commissioning_excursion_deg=self.excursion_deg,
            commissioning_watchdog_s=0.15,
            # The standalone laboratory jogger may run in the controller's
            # NORMAL state. Its one-joint, low-speed and bounded-motion guards
            # remain independent from the safety-mode check used elsewhere.
            commissioning_require_reduced=False,
            send_interval=0.05,
            **kwargs,
        )
        try:
            config.validate()
        except SystemExit as error:
            raise ValueError(str(error)) from error
        return config


class ManualArmTestSession:
    """Explicit state machine; no tracking state can enter this path."""

    def __init__(
        self,
        monitor_factory: BackendFactory = URMonitorBackend,
        commissioning_factory: BackendFactory | None = None,
    ) -> None:
        self._monitor_factory = monitor_factory
        self._commissioning_factory = commissioning_factory or (
            lambda config: URBackend(config, require_feedback=True)
        )
        self._backend = None
        self.settings: ManualTestSettings | None = None
        self.phase = "disconnected"
        self.error: str | None = None
        self._jog_direction = 0
        self._joint_speeds: dict[str, float] = {}

    def connect_monitor(self, settings: ManualTestSettings) -> None:
        self._require_phase("disconnected")
        config = settings.config(OPERATION_MONITOR)
        self._run_transition(
            lambda: self._monitor_factory(config), "monitoring", settings
        )

    def prepare_control(self) -> None:
        self._require_phase("monitoring")
        assert self.settings is not None
        config = self.settings.config(OPERATION_COMMISSIONING)
        previous = self._backend
        self._backend = None
        if previous is not None:
            previous.close()
        self._run_transition(
            lambda: self._commissioning_factory(config), "prepared", self.settings
        )

    def arm(self) -> None:
        self._require_phase("prepared")
        try:
            self._backend.arm_commissioning()
            self.phase = "armed"
        except (Exception, SystemExit) as error:
            self._fault(error)

    def begin_jog(self, direction: int) -> None:
        self._require_phase("armed")
        if direction not in (-1, 1):
            raise ValueError("Jog direction must be -1 or +1.")
        self._jog_direction = direction
        self._joint_speeds = {}

    def begin_multi_jog(self, speeds_deg_s: dict[str, float]) -> None:
        self._require_phase("armed")
        cleaned = {
            joint: float(speed)
            for joint, speed in speeds_deg_s.items()
            if joint in JOINT_NAMES and speed != 0.0
        }
        if not cleaned or len(cleaned) != len(speeds_deg_s):
            raise ValueError("Select at least one valid joint direction.")
        self._jog_direction = 0
        self._joint_speeds = cleaned

    def end_jog(self) -> None:
        if self._jog_direction == 0 and not self._joint_speeds:
            return
        self._jog_direction = 0
        self._joint_speeds = {}
        if self.phase != "armed" or self._backend is None:
            return
        try:
            # Send stopj immediately on release. The backend watchdog remains a
            # second line of defence if a release event is missed.
            self._backend.pause()
        except (Exception, SystemExit) as error:
            self._fault(error)

    def set_speed(self, speed_deg_s: float) -> None:
        if self.phase not in ("prepared", "armed") or self._backend is None:
            raise RuntimeError("Connect and prepare manual control before changing speed.")
        try:
            self._backend.set_commissioning_speed(speed_deg_s)
        except (Exception, SystemExit) as error:
            self._fault(error)

    def tick(self) -> RobotState | None:
        if self._backend is None or self.phase not in (
            "monitoring",
            "prepared",
            "armed",
        ):
            return None
        try:
            if self.phase == "armed" and self._jog_direction:
                self._backend.refresh_jog(self._jog_direction)
            elif self.phase == "armed" and self._joint_speeds:
                self._backend.refresh_joint_jogs(self._joint_speeds)
            return self._backend.robot_state()
        except (Exception, SystemExit) as error:
            self._fault(error)
            return None

    def status_lines(self) -> list[str]:
        if self.error:
            return [self.error]
        if self._backend is None:
            return ["No robot connection."]
        try:
            return list(self._backend.status_lines())
        except (Exception, SystemExit) as error:
            self._fault(error)
            return [str(error)]

    def stop_and_disconnect(self) -> None:
        self._jog_direction = 0
        self._joint_speeds = {}
        backend, self._backend = self._backend, None
        failure: BaseException | None = None
        if backend is not None:
            try:
                if self.phase in ("prepared", "armed"):
                    backend.pause()
            except BaseException as error:
                failure = error
            try:
                backend.close()
            except BaseException as error:
                failure = failure or error
        self.phase = "disconnected"
        self.settings = None
        self.error = None
        if failure is not None:
            raise RuntimeError(f"Robot shutdown reported an error: {failure}") from failure

    def close(self) -> None:
        try:
            self.stop_and_disconnect()
        except (Exception, SystemExit):
            # Closing every available transport remains more important than a
            # shutdown error that cannot be acted on after the window closes.
            self.phase = "disconnected"

    @property
    def can_jog(self) -> bool:
        return self.phase == "armed" and self.error is None

    @property
    def action_hint(self) -> str:
        return {
            "disconnected": "Enter one robot IP and connect in read-only mode.",
            "monitoring": "Confirm the live joint values, then prepare manual control.",
            "prepared": "The command channel is open. Capture the stationary current pose.",
            "armed": "Hold − or + to move the selected joint; release to stop.",
            "fault": "A fault is latched. Stop and disconnect before trying again.",
        }[self.phase]

    def _run_transition(self, create, phase: str, settings) -> None:
        try:
            self._backend = create()
            self.settings = settings
            self.phase = phase
            self.error = None
        except (Exception, SystemExit) as error:
            self._fault(error)

    def _fault(self, error: BaseException) -> None:
        self._jog_direction = 0
        self._joint_speeds = {}
        self.error = str(error)
        self.phase = "fault"
        if self._backend is not None:
            try:
                self._backend.close()
            except BaseException:
                pass
            self._backend = None

    def _require_phase(self, expected: str) -> None:
        if self.phase != expected:
            raise RuntimeError(
                f"This action requires {expected}; current phase is {self.phase}."
            )
