"""Read-only UR7e monitoring over Dashboard Server and RTDE."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.targets import (
    GRIPPER_OPEN,
    JOINT_NAMES,
    ArmState,
    RobotState,
)
from vision_robot_arm.robot.ur_dashboard import DashboardStatus, query_status
from vision_robot_arm.robot.ur_rtde import RtdeClient

FEEDBACK_MAX_AGE_S = 0.5
DASHBOARD_REFRESH_S = 1.0

Clock = Callable[[], float]
RtdeFactory = Callable[[str, int], RtdeClient]
StatusQuery = Callable[[str, int], DashboardStatus | None]


class MonitorConnectionError(RuntimeError):
    pass


class URMonitorArm:
    def __init__(
        self,
        name: str,
        host: str,
        config: RobotConfig,
        rtde_factory: RtdeFactory,
        status_query: StatusQuery,
        clock: Clock,
    ) -> None:
        self.name = name
        self.host = host
        self._clock = clock
        self._status_query = status_query
        self._dashboard_port = config.dashboard_port
        self._rtde = rtde_factory(host, config.rtde_port)
        self._sample: dict[str, Any] = {}
        self._sample_at: float | None = None
        self._dashboard: DashboardStatus | None = None
        self._dashboard_at: float | None = None
        if not self._rtde.connect():
            self._rtde.close()
            raise MonitorConnectionError(
                f"Could not start read-only RTDE monitoring for {name} at "
                f"{host}:{config.rtde_port}: {self._rtde.last_error}"
            )
        self.refresh_dashboard(force=True)

    def poll(self) -> None:
        sample = self._rtde.read()
        if sample is not None:
            self._sample = sample
            self._sample_at = self._clock()
        self.refresh_dashboard()

    def refresh_dashboard(self, force: bool = False) -> None:
        now = self._clock()
        if (
            not force
            and self._dashboard_at is not None
            and now - self._dashboard_at < DASHBOARD_REFRESH_S
        ):
            return
        self._dashboard = self._status_query(self.host, self._dashboard_port)
        self._dashboard_at = now

    @property
    def fresh(self) -> bool:
        return (
            self._sample_at is not None
            and self._clock() - self._sample_at <= FEEDBACK_MAX_AGE_S
        )

    def state(self) -> ArmState | None:
        joints = self._sample.get("actual_q") if self.fresh else None
        if (
            not joints
            or len(joints) != len(JOINT_NAMES)
            or not all(math.isfinite(value) for value in joints)
        ):
            return None
        degrees = {
            name: math.degrees(joints[index])
            for index, name in enumerate(JOINT_NAMES)
        }
        return ArmState(joints=degrees, targets=dict(degrees), gripper=GRIPPER_OPEN)

    def status_line(self) -> str:
        dashboard = (
            self._dashboard.describe()
            if self._dashboard is not None
            else "dashboard unavailable"
        )
        age = (
            f"{(self._clock() - self._sample_at) * 1000:.0f} ms"
            if self._sample_at is not None
            else "no sample"
        )
        speed = self._maximum_joint_speed_deg_s()
        tcp_speed = self._tcp_linear_speed_m_s()
        scaling = self._finite_scalar("speed_scaling")
        version = self._rtde.controller_version
        version_text = ".".join(map(str, version)) if version else "unknown"
        return (
            f"ur {self.name[0].upper()} {self.host} MONITOR-ONLY | {dashboard} | "
            f"RTDE {age}, PolyScope {version_text}, scale "
            f"{scaling * 100:.0f}%, joint {speed:.2f} deg/s, TCP {tcp_speed:.3f} m/s"
            if scaling is not None and speed is not None and tcp_speed is not None
            else f"ur {self.name[0].upper()} {self.host} MONITOR-ONLY | {dashboard} | RTDE {age}"
        )

    def _maximum_joint_speed_deg_s(self) -> float | None:
        values = self._sample.get("actual_qd") if self.fresh else None
        if not values or len(values) != len(JOINT_NAMES):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        return max(abs(math.degrees(value)) for value in values)

    def _tcp_linear_speed_m_s(self) -> float | None:
        values = self._sample.get("actual_TCP_speed") if self.fresh else None
        if not values or len(values) != 6 or not all(math.isfinite(value) for value in values):
            return None
        return math.sqrt(sum(value * value for value in values[:3]))

    def _finite_scalar(self, name: str) -> float | None:
        value = self._sample.get(name) if self.fresh else None
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

    def close(self) -> None:
        self._rtde.close()


class URMonitorBackend:
    """Backend-shaped read-only monitor; it has deliberately no send/home API."""

    def __init__(
        self,
        config: RobotConfig,
        rtde_factory: RtdeFactory = RtdeClient,
        status_query: StatusQuery = query_status,
        clock: Clock = time.monotonic,
    ) -> None:
        config.validate()
        self._arms: dict[str, URMonitorArm] = {}
        try:
            for name, host in config.hosts.items():
                self._arms[name] = URMonitorArm(
                    name, host, config, rtde_factory, status_query, clock
                )
        except BaseException:
            self.close()
            raise

    def robot_state(self) -> RobotState | None:
        arms = {}
        for name, arm in self._arms.items():
            arm.poll()
            state = arm.state()
            if state is not None:
                arms[name] = state
        return RobotState(arms=arms, lift_mode=False) if arms else None

    def status_lines(self) -> list[str]:
        for arm in self._arms.values():
            arm.poll()
        return [arm.status_line() for arm in self._arms.values()]

    def close(self) -> None:
        for arm in self._arms.values():
            arm.close()
