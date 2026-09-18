"""Drive real UR7e cobots over URScript, with the safeguards a real arm needs.

Streaming raw mapped angles at a moving person would make the controller chase steps
of tens of degrees. Every arm therefore owns a setpoint generator that ramps toward the
mapped pose at `--robot-max-speed`, starts from a feedback-confirmed current pose, and
decelerates on shutdown. Feedback comes from RTDE, so the on-screen twin shows the
robot, not the wish.
"""

import math
import socket
import time
from typing import Any, Callable

from vision_robot_arm.robot.backend import Clock, TargetTracker
from vision_robot_arm.robot.config import OPERATION_COMMISSIONING, RobotConfig
from vision_robot_arm.robot.simulation import SimulatedArm
from vision_robot_arm.robot.targets import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    JOINT_NAMES,
    MAPPED_JOINTS,
    REPORTED_JOINTS,
    ArmState,
    ArmTargets,
    JointTargets,
    RobotState,
    full_joint_pose,
)
from vision_robot_arm.robot.ur_dashboard import query_status
from vision_robot_arm.robot.ur_rtde import RtdeClient

CONNECT_TIMEOUT_S = 2.0
STOP_DECELERATION_DEG_S2 = 120.0
# A pose gap must not turn into one giant catch-up step when the person returns.
MAX_CATCHUP_INTERVALS = 3.0
FEEDBACK_MAX_AGE_S = 0.5
COMMISSIONING_STATIONARY_DEG_S = 0.5
JOINT_SHORT_NAMES = {"base": "B", "shoulder": "S", "elbow": "E", "wrist_1": "W1"}

Connector = Callable[[str, int], Any]
RtdeFactory = Callable[[str, int], RtdeClient | None]
StatusQuery = Callable[[str, int], Any]


class ControlFault(RuntimeError):
    """A latched control fault requires the session to be restarted."""


def encode_servoj(
    joints_deg: dict[str, float],
    duration_s: float,
    lookahead_s: float,
    gain: int,
) -> bytes:
    command = (
        f"servoj({_joint_vector(joints_deg)}, 0, 0, "
        f"{duration_s:.3f}, {lookahead_s:.3f}, {gain:d})\n"
    )
    return command.encode("ascii")


def encode_stopj(accel_deg_s2: float = STOP_DECELERATION_DEG_S2) -> bytes:
    return f"stopj({math.radians(accel_deg_s2):.3f})\n".encode("ascii")


def encode_gripper(closed: bool, tool_output: int = 0) -> bytes:
    value = "True" if closed else "False"
    return f"set_tool_digital_out({tool_output}, {value})\n".encode("ascii")


def default_connector(host: str, port: int) -> Any:
    return socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S)


def default_rtde_factory(host: str, port: int) -> RtdeClient | None:
    client = RtdeClient(host, port)
    if client.connect():
        return client
    print(
        f"robot: no RTDE feedback from {host}:{port} ({client.last_error}); "
        "running open-loop"
    )
    return None


class URArm:
    def __init__(
        self,
        name: str,
        host: str,
        config: RobotConfig,
        connector: Connector,
        rtde_factory: RtdeFactory | None,
        clock: Clock,
    ) -> None:
        self.name = name
        self.host = host
        self.port = config.ur_port
        self._config = config
        self._clock = clock
        try:
            self._socket = connector(host, config.ur_port)
        except OSError as error:
            raise SystemExit(
                f"Could not connect to the {name} UR7e at {host}:{config.ur_port}: {error}\n"
                "Check the IP address and put the robot in Remote Control mode."
            ) from error

        self._setpoints = SimulatedArm(config)
        self._gripper: str | None = None
        self._feedback: dict[str, Any] = {}
        self._feedback_at: float | None = None
        self._rtde: RtdeClient | None = None
        self._armed = False
        try:
            self._rtde = rtde_factory(host, config.rtde_port) if rtde_factory is not None else None
        except BaseException:
            self._release()
            raise

    def capture_current_as_setpoint(self) -> dict[str, float]:
        """Adopt fresh feedback as the control origin without issuing motion."""
        actual = self.feedback_joints
        if actual is None:
            raise ControlFault(f"{self.name}: fresh joint feedback is required.")
        self._setpoints.joints.update(actual)
        self._setpoints.targets.update(actual)
        self._armed = True
        return dict(actual)

    @property
    def feedback_speeds_deg_s(self) -> dict[str, float] | None:
        if self.feedback_joints is None:
            return None
        actual = self._feedback.get("actual_qd")
        if not actual or len(actual) != len(JOINT_NAMES) or not all(
            math.isfinite(value) for value in actual
        ):
            return None
        return {
            name: math.degrees(actual[index])
            for index, name in enumerate(JOINT_NAMES)
        }

    def commissioning_step(
        self,
        joint: str,
        direction: int,
        origin: dict[str, float],
        elapsed_s: float,
    ) -> None:
        if joint not in JOINT_NAMES or direction not in (-1, 1):
            raise ControlFault("Invalid commissioning jog request.")
        lower = max(
            self._config.limit_for(joint).minimum,
            origin[joint] - self._config.commissioning_excursion_deg,
        )
        upper = min(
            self._config.limit_for(joint).maximum,
            origin[joint] + self._config.commissioning_excursion_deg,
        )
        requested = self._setpoints.joints[joint] + (
            direction * self._config.commissioning_speed_deg_s * max(0.0, elapsed_s)
        )
        self._setpoints.joints[joint] = max(lower, min(upper, requested))
        self._setpoints.targets.update(self._setpoints.joints)
        self._send(
            encode_servoj(
                self._setpoints.joints,
                self._config.send_interval,
                self._config.servo_lookahead_s,
                self._config.servo_gain,
            )
        )

    def pause(self) -> None:
        self._send(encode_stopj())
        actual = self.feedback_joints
        if actual is not None:
            self._setpoints.joints.update(actual)
            self._setpoints.targets.update(actual)

    @property
    def feedback_joints(self) -> dict[str, float] | None:
        if self._feedback_at is None or self._clock() - self._feedback_at > FEEDBACK_MAX_AGE_S:
            return None
        actual = self._feedback.get("actual_q")
        if not actual or len(actual) != len(JOINT_NAMES) or not all(math.isfinite(v) for v in actual):
            return None
        return {name: math.degrees(actual[index]) for index, name in enumerate(JOINT_NAMES)}

    def update(self, targets: ArmTargets, elapsed_s: float) -> None:
        """Move the setpoint toward the mapped pose, then command that setpoint."""
        if not self._armed:
            return
        self._setpoints.set_targets(targets.joints, targets.gripper)
        self._setpoints.step(self._config.max_speed_deg_s * max(elapsed_s, 0.0))
        self._send(
            encode_servoj(
                self._setpoints.joints,
                self._config.send_interval,
                self._config.servo_lookahead_s,
                self._config.servo_gain,
            )
        )
        if targets.gripper is not None and targets.gripper != self._gripper:
            self._send(encode_gripper(targets.gripper == GRIPPER_CLOSE, self._config.tool_output))
            self._gripper = targets.gripper

    def poll_feedback(self) -> None:
        if self._rtde is None:
            return
        sample = self._rtde.read()
        if sample:
            self._feedback = sample
            self._feedback_at = self._clock()
        elif not self._rtde.connected:
            # Never present a stale pose as the live one.
            self._feedback = {}
            self._feedback_at = None

    def check_control_health(self) -> None:
        if self._rtde is None:
            return
        safety = self._feedback.get("safety_status")
        mode = self._feedback.get("robot_mode")
        if safety is not None and safety not in (1, 2):
            raise ControlFault(f"{self.name}: controller safety state {safety} blocks motion.")
        if mode is not None and mode != 7:
            raise ControlFault(f"{self.name}: controller is not running (mode {mode}).")
        if self._armed and self.feedback_joints is None:
            raise ControlFault(f"{self.name}: fresh position feedback is required to continue.")

    def state(self) -> ArmState:
        return ArmState(
            joints=self.feedback_joints or dict(self._setpoints.joints),
            targets=dict(self._setpoints.targets),
            gripper=self._gripper or GRIPPER_OPEN,
        )

    def status_line(self) -> str:
        health = self._health()
        joints = self.feedback_joints or self._setpoints.joints
        detail = " ".join(
            f"{JOINT_SHORT_NAMES[joint]} {joints[joint]:6.1f}"
            for joint in REPORTED_JOINTS
            if joint in joints
        )
        return f"ur {self.name[0].upper()} {self.host} {health} {detail} grip {self._gripper or 'n/a'}"

    def close(self) -> None:
        """Shutdown must always finish: a dead socket cannot stop the other arm."""
        try:
            self._socket.sendall(encode_stopj())
        except OSError:
            pass
        finally:
            self._release()

    def _release(self) -> None:
        if self._rtde is not None:
            self._rtde.close()
        try:
            self._socket.close()
        except OSError:
            pass

    def _health(self) -> str:
        if self._rtde is None:
            return "open-loop"
        if not self._rtde.connected:
            return "feedback-lost"
        if self.feedback_joints is None:
            return "feedback-stale"
        mode = ROBOT_MODES.get(self._feedback.get("robot_mode"), "?")
        safety = SAFETY_STATUSES.get(self._feedback.get("safety_status"), "?")
        return f"{mode}/{safety}"

    def _send(self, payload: bytes) -> None:
        try:
            self._socket.sendall(payload)
        except OSError as error:
            raise SystemExit(
                f"Lost the connection to the {self.name} UR7e at {self.host}: {error}\n"
                "The controller drops URScript clients when the robot leaves Remote Control mode."
            ) from error


ROBOT_MODES = {
    0: "DISCONNECTED",
    1: "CONFIRM_SAFETY",
    2: "BOOTING",
    3: "POWER_OFF",
    4: "POWER_ON",
    5: "IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
}
SAFETY_STATUSES = {
    1: "NORMAL",
    2: "REDUCED",
    3: "PROTECTIVE_STOP",
    4: "RECOVERY",
    5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION",
    9: "FAULT",
}


class URBackend:
    def __init__(
        self,
        config: RobotConfig,
        connector: Connector = default_connector,
        clock: Clock = time.monotonic,
        rtde_factory: RtdeFactory | None = default_rtde_factory,
        status_query: StatusQuery = query_status,
        require_feedback: bool = False,
    ) -> None:
        self._config = config
        self._clock = clock
        config.validate()
        self._fault: str | None = None
        self._require_feedback = require_feedback
        self._status_query = status_query
        self._commissioning = config.operation == OPERATION_COMMISSIONING
        if require_feedback and not config.feedback:
            raise ControlFault("Hardware control requires RTDE feedback.")
        if config.preflight:
            _preflight(config, status_query, require_reduced=self._commissioning)
        factory = rtde_factory if config.feedback else None
        self._arms: dict[str, URArm] = {}
        try:
            for name, host in config.hosts.items():
                self._arms[name] = URArm(name, host, config, connector, factory, clock)
                if require_feedback and self._arms[name]._rtde is None:
                    raise ControlFault(f"{name}: the interactive hardware session requires RTDE feedback.")
        except BaseException:
            self.close()
            raise
        self._tracker = TargetTracker()
        self._next_send_at = 0.0
        self._last_send_at: float | None = None
        self._commissioning_origins: dict[str, dict[str, float]] = {}
        self._jog_direction = 0
        self._jog_deadline = 0.0
        self._commissioning_last_step: float | None = None
    def arm_commissioning(self) -> None:
        if not self._commissioning:
            raise ControlFault("This backend is not in commissioning mode.")
        if self._fault is not None:
            raise ControlFault(self._fault)
        try:
            _preflight(self._config, self._status_query, require_reduced=True)
            origins = {}
            for name, arm in self._arms.items():
                arm.poll_feedback()
                self._check_commissioning_health(arm, require_stationary=True)
                origins[name] = arm.capture_current_as_setpoint()
            self._commissioning_origins = origins
            self._jog_direction = 0
            self._commissioning_last_step = self._clock()
        except BaseException as error:
            self._fault = str(error)
            self.close()
            raise

    def arm_tracking(self) -> None:
        """Capture a stationary RTDE pose; never move merely because control connected."""
        if self._commissioning:
            raise ControlFault("Commissioning cannot be armed for vision tracking.")
        if self._fault is not None:
            raise ControlFault(self._fault)
        try:
            if self._config.preflight:
                _preflight(self._config, self._status_query)
            arms = tuple(self._arms.values())
            for arm in arms:
                arm.poll_feedback()
                arm.check_control_health()
                speeds = arm.feedback_speeds_deg_s
                if arm.feedback_joints is None or speeds is None:
                    raise ControlFault(
                        f"{arm.name}: fresh position and velocity feedback is required."
                    )
                if max(abs(value) for value in speeds.values()) > COMMISSIONING_STATIONARY_DEG_S:
                    raise ControlFault(
                        f"{arm.name}: robot must be stationary before tracking is armed."
                    )
            # Validate every arm first, then atomically adopt both current poses.
            for arm in arms:
                arm.capture_current_as_setpoint()
        except BaseException as error:
            self._fault = str(error)
            self.close()
            raise

    def refresh_jog(self, direction: int) -> None:
        if not self._commissioning_origins:
            raise ControlFault("Capture the commissioning origin before jogging.")
        if direction not in (-1, 1):
            raise ControlFault("Jog direction must be -1 or +1.")
        self._jog_direction = direction
        self._jog_deadline = self._clock() + self._config.commissioning_watchdog_s

    def _commissioning_tick(self) -> None:
        if not self._commissioning or not self._commissioning_origins:
            return
        now = self._clock()
        if self._jog_direction and now > self._jog_deadline:
            self.pause()
            return
        if not self._jog_direction:
            return
        if (
            self._commissioning_last_step is not None
            and now - self._commissioning_last_step < self._config.send_interval
        ):
            return
        elapsed = (
            self._config.send_interval
            if self._commissioning_last_step is None
            else min(
                now - self._commissioning_last_step,
                self._config.send_interval * MAX_CATCHUP_INTERVALS,
            )
        )
        try:
            for arm in self._arms.values():
                arm.poll_feedback()
                self._check_commissioning_health(arm)
            for name, arm in self._arms.items():
                arm.commissioning_step(
                    self._config.commissioning_joint,
                    self._jog_direction,
                    self._commissioning_origins[name],
                    elapsed,
                )
            self._commissioning_last_step = now
        except BaseException as error:
            self._fault = str(error)
            self.close()
            raise

    def _check_commissioning_health(
        self, arm: URArm, require_stationary: bool = False
    ) -> None:
        arm.check_control_health()
        if arm._feedback.get("safety_status") != 2:
            raise ControlFault(
                f"{arm.name}: commissioning requires controller safety status REDUCED."
            )
        speeds = arm.feedback_speeds_deg_s
        if speeds is None:
            raise ControlFault(
                f"{arm.name}: fresh joint velocity feedback is required."
            )
        if require_stationary and max(
            abs(value) for value in speeds.values()
        ) > COMMISSIONING_STATIONARY_DEG_S:
            raise ControlFault(
                f"{arm.name}: robot must be stationary before commissioning."
            )

    def ready(self) -> bool:
        ready = True
        for arm in self._arms.values():
            arm.poll_feedback()
            arm.check_control_health()
            ready = arm._armed and ready
        return ready

    def pause(self) -> None:
        self._jog_direction = 0
        self._jog_deadline = 0.0
        self._tracker = TargetTracker()
        self._last_send_at = None
        self._next_send_at = 0.0
        try:
            for arm in self._arms.values():
                arm.pause()
        except BaseException:
            self.close()
            raise

    def send(self, targets: JointTargets) -> None:
        if self._commissioning:
            raise ControlFault("Vision targets are disabled in commissioning mode.")
        if self._fault is not None:
            raise ControlFault(self._fault)
        if not targets.has_data:
            return
        self._tracker.update(targets)
        now = self._clock()
        if now < self._next_send_at:
            return
        elapsed = self._catch_up_interval(now)
        self._next_send_at = now + self._config.send_interval
        self._last_send_at = now

        # Use everything seen since the last transmission: a gripper command that arrived
        # on a frame the rate limiter skipped must not be thrown away.
        accumulated = self._tracker.accumulated_targets()
        if accumulated is None:
            return
        try:
            # Check every arm before allowing either one to stream a new command.
            for arm in self._arms.values():
                arm.poll_feedback()
                arm.check_control_health()
            for name, arm in self._arms.items():
                arm.update(accumulated.arm(name), elapsed)
        except ControlFault as error:
            self._fault = str(error)
            self.close()
            raise

    def _catch_up_interval(self, now: float) -> float:
        """Time credited to the ramp. A long pose gap must not buy one huge step."""
        if self._last_send_at is None:
            return self._config.send_interval
        return min(now - self._last_send_at, self._config.send_interval * MAX_CATCHUP_INTERVALS)

    def robot_state(self) -> RobotState | None:
        if not self._arms:
            return None
        self._commissioning_tick()
        for arm in self._arms.values():
            arm.poll_feedback()
        return RobotState(
            arms={name: arm.state() for name, arm in self._arms.items()},
            lift_mode=bool(self._tracker.last_targets and self._tracker.last_targets.lift_mode),
        )

    def status_lines(self) -> list[str]:
        lines = [arm.status_line() for arm in self._arms.values()]
        if self._commissioning:
            state = "armed" if self._commissioning_origins else "not armed"
            lines.append(
                f"commissioning {state}: {self._config.commissioning_joint}, "
                f"{self._config.commissioning_speed_deg_s:g} deg/s, "
                f"+/-{self._config.commissioning_excursion_deg:g} deg"
            )
        return lines

    def close(self) -> None:
        for arm in self._arms.values():
            arm.close()


def _preflight(
    config: RobotConfig,
    status_query: StatusQuery,
    require_reduced: bool = False,
) -> None:
    for name, host in config.hosts.items():
        status = status_query(host, config.dashboard_port)
        if status is None:
            raise SystemExit(
                f"Cannot verify the {name} UR7e at {host}: dashboard unavailable. "
                "No motion is authorized. --no-robot-preflight explicitly disables this check."
            )
        problem = status.blocking_problem()
        if problem:
            raise SystemExit(
                f"The {name} UR7e at {host} is not ready to run URScript: {problem}.\n"
                f"Dashboard reports {status.describe()}. "
                "Start with --no-robot-preflight to skip this check."
            )
        if require_reduced and status.safety_status != "REDUCED":
            raise SystemExit(
                f"The {name} UR7e must report REDUCED safety status for commissioning. "
                f"Dashboard reports {status.describe()}."
            )


def _joint_vector(joints_deg: dict[str, float]) -> str:
    pose = full_joint_pose(joints_deg)
    return "[" + ", ".join(f"{math.radians(pose[name]):.4f}" for name in JOINT_NAMES) + "]"
