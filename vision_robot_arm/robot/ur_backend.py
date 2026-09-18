"""Drive real UR7e cobots over URScript, with the safeguards a real arm needs.

Streaming raw mapped angles at a moving person would make the controller chase steps
of tens of degrees. Every arm therefore owns a setpoint generator that ramps toward the
mapped pose at `--robot-max-speed`, starts from a feedback-confirmed current pose, and
decelerates on shutdown. Feedback comes from RTDE, so the on-screen twin shows the
robot, not the wish.
"""

import json
import math
import socket
import time
from pathlib import Path
from typing import Any, Callable

from vision_robot_arm.robot.backend import Clock, TargetTracker
from vision_robot_arm.robot.config import (
    MAX_FOLLOW_INTERVAL_S,
    COMMISSIONING_MAX_SPEED_DEG_S,
    GRIPPER_ROBOTIQ,
    OPERATION_COMMISSIONING,
    OPERATION_KEYFRAME,
    OPERATION_TRACKING,
    RobotConfig,
)
from vision_robot_arm.robot.robotiq_socket import GripperError, RobotiqSocketGripper
from vision_robot_arm.robot.simulation import SimulatedArm
from vision_robot_arm.robot.targets import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    JOINT_BASE,
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
JOINT_SHORT_NAMES = {
    "base": "B", "shoulder": "S", "elbow": "E",
    "wrist_1": "W1", "wrist_2": "W2", "wrist_3": "W3",
}

Connector = Callable[[str, int], Any]
RtdeFactory = Callable[[str, int], RtdeClient | None]
StatusQuery = Callable[[str, int], Any]
GripperFactory = Callable[[str], RobotiqSocketGripper]
# After a gripper socket error, wait this long before trying to reconnect. Each
# attempt can block for one reply timeout, so a dead daemon must cost the frame
# loop one short stall every ten seconds, not one per frame.
GRIPPER_RETRY_S = 10.0


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


def encode_movej(
    joints_deg: dict[str, float], accel_deg_s2: float, speed_deg_s: float
) -> bytes:
    """One complete joint move; the controller plans and executes the whole path."""
    return (
        f"movej({_joint_vector(joints_deg)}, "
        f"{math.radians(accel_deg_s2):.4f}, {math.radians(speed_deg_s):.4f})\n"
    ).encode("ascii")


def encode_speedj_vector(
    speeds_deg_s: dict[str, float], duration_s: float
) -> bytes:
    velocities = [0.0] * len(JOINT_NAMES)
    for joint, speed_deg_s in speeds_deg_s.items():
        velocities[JOINT_NAMES.index(joint)] = math.radians(speed_deg_s)
    vector = "[" + ", ".join(f"{value:.5f}" for value in velocities) + "]"
    # A gentle acceleration lets the controller blend repeated dead-man
    # updates instead of settling a new servoj position every UI frame.
    return f"speedj({vector}, {math.radians(10.0):.5f}, {duration_s:.3f})\n".encode("ascii")


def encode_speedj(joint: str, speed_deg_s: float, duration_s: float) -> bytes:
    return encode_speedj_vector({joint: speed_deg_s}, duration_s)


def encode_gripper(closed: bool, tool_output: int = 0) -> bytes:
    value = "True" if closed else "False"
    return f"set_tool_digital_out({tool_output}, {value})\n".encode("ascii")


def default_gripper_factory(host: str) -> RobotiqSocketGripper:
    return RobotiqSocketGripper(host)


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
        gripper_factory: GripperFactory | None = None,
    ) -> None:
        self.name = name
        self.host = host
        self.port = config.ur_port
        self._gripper_factory = gripper_factory
        self._robotiq: RobotiqSocketGripper | None = None
        self._gripper_error: str | None = None
        self._gripper_retry_at = 0.0
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
        self._tracking_velocities = {name: 0.0 for name in JOINT_NAMES}
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
        self._tracking_velocities = {name: 0.0 for name in JOINT_NAMES}
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

    def commissioning_step_multi(
        self,
        speeds_deg_s: dict[str, float],
        origin: dict[str, float],
        excursion_deg: float,
        watchdog_s: float,
    ) -> None:
        actual = self.feedback_joints
        if actual is None:
            raise ControlFault(f"{self.name}: fresh joint feedback is required.")
        allowed: dict[str, float] = {}
        for joint, speed in speeds_deg_s.items():
            if joint not in JOINT_NAMES or not math.isfinite(speed):
                raise ControlFault("Invalid commissioning jog request.")
            lower = max(
                self._config.limit_for(joint).minimum,
                origin[joint] - excursion_deg,
            )
            upper = min(
                self._config.limit_for(joint).maximum,
                origin[joint] + excursion_deg,
            )
            at_limit = actual[joint] >= upper if speed > 0 else actual[joint] <= lower
            if speed and not at_limit:
                allowed[joint] = speed
        if not allowed:
            self.pause()
            return
        self._setpoints.joints.update(actual)
        self._setpoints.targets.update(actual)
        self._send(encode_speedj_vector(allowed, watchdog_s))

    def move_joints(
        self, targets: dict[str, float], speed_deg_s: float, accel_deg_s2: float
    ) -> None:
        """Send exactly one movej. Nothing is streamed afterwards; feedback decides arrival.

        Joints not named in `targets` are commanded at their measured position, so
        the controller holds them while the named joint travels.
        """
        actual = self.feedback_joints
        if actual is None:
            raise ControlFault(f"{self.name}: fresh joint feedback is required.")
        self._setpoints.joints.update(actual)
        self._setpoints.targets.update(actual)
        self._setpoints.targets.update(targets)
        self._tracking_velocities = {name: 0.0 for name in JOINT_NAMES}
        self._send(encode_movej(self._setpoints.targets, accel_deg_s2, speed_deg_s))

    def pause(self) -> None:
        self._send(encode_stopj())
        self._tracking_velocities = {name: 0.0 for name in JOINT_NAMES}
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

    def update(self, targets: ArmTargets, elapsed_s: float, *, speed_limits=None) -> None:
        """Move the setpoint toward the mapped pose, then command that setpoint."""
        if not self._armed:
            return
        self._setpoints.set_targets(targets.joints, targets.gripper)
        if speed_limits is not None or self._config.operation in (OPERATION_TRACKING, OPERATION_KEYFRAME):
            self._step_tracking_setpoints(max(elapsed_s, 0.0), speed_limits)
        else:
            self._setpoints.step(self._config.max_speed_deg_s * max(elapsed_s, 0.0))
        self._send(
            encode_servoj(
                self._setpoints.joints,
                self._config.send_interval,
                self._config.servo_lookahead_s,
                self._config.servo_gain,
            )
        )
        self.send_gripper(targets.gripper)

    def connect_gripper(self) -> None:
        """Open the Robotiq socket if that driver is selected.

        The gripper is optional by design: one that does not answer is reported
        in the status line and retried every few seconds, and never keeps the
        arm from arming or moving.
        """
        if self._config.gripper_driver != GRIPPER_ROBOTIQ or self._gripper_factory is None:
            return
        if self._robotiq is not None or self._clock() < self._gripper_retry_at:
            return
        gripper = self._gripper_factory(self.host)
        try:
            gripper.connect()
        except GripperError as error:
            gripper.close()
            self._gripper_error = str(error)
            self._gripper_retry_at = self._clock() + GRIPPER_RETRY_S
            return
        self._robotiq = gripper
        self._gripper_error = None

    @property
    def gripper_state(self) -> str | None:
        return self._gripper

    @property
    def gripper_error(self) -> str | None:
        return self._gripper_error

    def send_gripper(self, gripper: str | None) -> None:
        """Command the gripper only when its requested state actually changes.

        The Robotiq driver talks to the URCap's own socket and never touches the
        motion program. The digital-output fallback is a URScript line and does
        replace whatever program the controller is running.
        """
        if gripper is None or gripper == self._gripper:
            return
        closed = gripper == GRIPPER_CLOSE
        if self._config.gripper_driver != GRIPPER_ROBOTIQ:
            self._send(encode_gripper(closed, self._config.tool_output))
            self._gripper = gripper
            return
        if self._robotiq is None:
            self.connect_gripper()
            if self._robotiq is None:
                return
        try:
            self._robotiq.command(
                closed,
                self._config.gripper_speed_percent,
                self._config.gripper_force_percent,
            )
        except GripperError as error:
            # A gripper hiccup must be visible on screen, not fatal for the arm.
            self._gripper_error = str(error)
            self._robotiq.close()
            self._robotiq = None
            self._gripper_retry_at = self._clock() + GRIPPER_RETRY_S
            return
        self._gripper_error = None
        self._gripper = gripper

    def _step_tracking_setpoints(self, elapsed_s: float, speed_limits=None) -> None:
        """Ramp velocity linearly and brake smoothly before each target."""
        if elapsed_s <= 0:
            return
        acceleration = self._config.tracking_acceleration_deg_s2
        for joint, target in self._setpoints.targets.items():
            max_speed = speed_limits.get(joint, 0.0) if speed_limits is not None else self._config.max_speed_deg_s
            current = self._setpoints.joints[joint]
            error = target - current
            if abs(error) < 1e-9:
                self._tracking_velocities[joint] = 0.0
                continue
            braking_speed = math.sqrt(2.0 * acceleration * abs(error))
            desired = math.copysign(min(max_speed, braking_speed), error)
            velocity = self._tracking_velocities[joint]
            max_change = acceleration * elapsed_s
            velocity += max(-max_change, min(max_change, desired - velocity))
            step = velocity * elapsed_s
            if step * error > 0 and abs(step) >= abs(error):
                self._setpoints.joints[joint] = target
                self._tracking_velocities[joint] = 0.0
            else:
                self._setpoints.joints[joint] = self._config.limit_for(joint).clamp(current + step)
                self._tracking_velocities[joint] = velocity

    def poll_feedback(self) -> dict[str, Any] | None:
        if self._rtde is None:
            return None
        sample = self._rtde.read()
        if sample:
            self._feedback = sample
            self._feedback_at = self._clock()
        elif not self._rtde.connected:
            # Never present a stale pose as the live one.
            self._feedback = {}
            self._feedback_at = None
        return sample

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
        grip = self._gripper or "n/a"
        if self._gripper_error:
            grip += f" (gripper error: {self._gripper_error})"
        return f"ur {self.name[0].upper()} {self.host} {health} {detail} grip {grip}"

    def close(self) -> None:
        """Shutdown must always finish: a dead socket cannot stop the other arm."""
        try:
            self._socket.sendall(encode_stopj())
        except OSError:
            pass
        finally:
            if self._robotiq is not None:
                self._robotiq.close()
                self._robotiq = None
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
        gripper_factory: GripperFactory = default_gripper_factory,
    ) -> None:
        self._config = config
        self._clock = clock
        config.validate()
        self._fault: str | None = None
        self._require_feedback = require_feedback
        self._status_query = status_query
        self._commissioning = config.operation == OPERATION_COMMISSIONING
        self._telemetry = None
        self._feedback_log = None
        self._next_telemetry_at = 0.0
        self._next_tracking_event_at = 0.0
        if require_feedback and not config.feedback:
            raise ControlFault("Hardware control requires RTDE feedback.")
        if config.preflight:
            _preflight(
                config,
                status_query,
                require_reduced=(
                    self._commissioning and config.commissioning_require_reduced
                ),
            )
        factory = rtde_factory if config.feedback else None
        self._arms: dict[str, URArm] = {}
        try:
            for name, host in config.hosts.items():
                self._arms[name] = URArm(
                    name, host, config, connector, factory, clock,
                    gripper_factory=gripper_factory,
                )
                if require_feedback and self._arms[name]._rtde is None:
                    raise ControlFault(f"{name}: the interactive hardware session requires RTDE feedback.")
        except BaseException:
            self.close()
            raise
        self._tracker = TargetTracker()
        self._next_send_at = 0.0
        self._last_send_at: float | None = None
        self._commissioning_origins: dict[str, dict[str, float]] = {}
        self._tracking_origins: dict[str, dict[str, float]] = {}
        self._jog_direction = 0
        self._jog_speeds: dict[str, float] = {}
        self._position_targets: dict[str, float] = {}
        self._position_speeds: dict[str, float] = {}
        self._position_status = "idle"
        # "stream": setpoints ramped and re-sent as servoj every tick.
        # "movej": one command already on the controller; the tick only watches.
        self._position_mode = "stream"
        self._jog_deadline = 0.0
        self._commissioning_last_step: float | None = None
        self._commissioning_speed_deg_s = config.commissioning_speed_deg_s
        self._follow_interval_s = config.follow_interval_s
        self._last_follow_at: float | None = None
        if config.telemetry_log_path:
            path = Path(config.telemetry_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._telemetry = path.open("a", encoding="utf-8", buffering=1)
            self._write_telemetry("session_started")
        if config.feedback_log_path:
            path = Path(config.feedback_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._feedback_log = path.open("a", encoding="utf-8", buffering=1)
            self._write_feedback("session_started")

    def arm_commissioning(self) -> None:
        if not self._commissioning:
            raise ControlFault("This backend is not in commissioning mode.")
        if self._fault is not None:
            raise ControlFault(self._fault)
        try:
            _preflight(
                self._config,
                self._status_query,
                require_reduced=self._config.commissioning_require_reduced,
            )
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
            # Best effort: open the gripper socket now so a fist later does not
            # wait for a connection. A silent gripper is shown, not fatal.
            for arm in arms:
                arm.connect_gripper()
            # Validate every arm first, then atomically adopt both current poses.
            self._tracking_origins = {
                arm.name: arm.capture_current_as_setpoint() for arm in arms
            }
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
        self._jog_speeds = {
            self._config.commissioning_joint: (
                direction * self._commissioning_speed_deg_s
            )
        }
        self._jog_deadline = self._clock() + self._config.commissioning_watchdog_s

    def refresh_joint_jogs(self, speeds_deg_s: dict[str, float]) -> None:
        if not self._commissioning_origins:
            raise ControlFault("Capture the commissioning origin before jogging.")
        cleaned = {
            joint: float(speed)
            for joint, speed in speeds_deg_s.items()
            if joint in JOINT_NAMES and math.isfinite(speed) and speed != 0.0
        }
        if not cleaned or len(cleaned) != len(speeds_deg_s):
            raise ControlFault("Select at least one valid joint velocity.")
        if any(abs(speed) > COMMISSIONING_MAX_SPEED_DEG_S for speed in cleaned.values()):
            raise ControlFault(
                f"Joint speed must not exceed {COMMISSIONING_MAX_SPEED_DEG_S:g} deg/s."
            )
        self._jog_direction = 0
        self._jog_speeds = cleaned
        self._jog_deadline = self._clock() + self._config.commissioning_watchdog_s

    def set_commissioning_speed(self, speed_deg_s: float) -> None:
        if (
            not math.isfinite(speed_deg_s)
            or not 0 < speed_deg_s <= COMMISSIONING_MAX_SPEED_DEG_S
        ):
            raise ControlFault(
                "Commissioning speed must be between 0 and "
                f"{COMMISSIONING_MAX_SPEED_DEG_S:g} deg/s."
            )
        self._commissioning_speed_deg_s = speed_deg_s

    def start_joint_positions(self, joints: dict[str, float], speeds: dict[str, float], *, relative=False) -> None:
        """Validate the complete request before a held manual position move."""
        if not self._commissioning or len(self._commissioning_origins) != 1:
            raise ValueError("Capture the stationary robot pose before entering targets.")
        if not joints or set(joints) != set(speeds) or not set(joints) <= set(JOINT_NAMES):
            raise ValueError("Enter at least one joint target and its speed.")
        if any(not math.isfinite(v) for v in joints.values()) or any(not math.isfinite(v) or not 0 < v <= COMMISSIONING_MAX_SPEED_DEG_S for v in speeds.values()):
            raise ValueError("Targets must be finite; speeds must be between 0 and 30 deg/s.")
        name, arm = next(iter(self._arms.items()))
        arm.poll_feedback()
        self._check_commissioning_health(arm, require_stationary=True)
        actual = arm.feedback_joints
        targets = {}
        for joint, value in joints.items():
            target = actual[joint] + value if relative else value
            self._validate_position_target(name, joint, target)
            targets[joint] = target
        arm.capture_current_as_setpoint()
        self._position_targets = targets
        self._position_speeds = dict(speeds)
        self._position_mode = "stream"
        self._jog_direction = 0
        self._jog_speeds = {}
        self._commissioning_last_step = self._clock()
        self._position_status = "running"
        self._jog_deadline = self._clock() + self._config.commissioning_watchdog_s

    def start_joint_move(self, joints: dict[str, float], speed_deg_s: float) -> None:
        """Validate, then send ONE movej. The controller owns the trajectory from here.

        Unlike start_joint_positions this streams nothing: a sequence step is a
        single bounded move to an already validated target, and the only later
        command is a stopj from stop, timeout, fault or a missed UI tick.
        """
        if not self._commissioning or len(self._commissioning_origins) != 1:
            raise ValueError("Capture the stationary robot pose before entering targets.")
        if not joints or not set(joints) <= set(JOINT_NAMES):
            raise ValueError("Enter at least one joint target.")
        if any(not math.isfinite(value) for value in joints.values()):
            raise ValueError("Targets must be finite.")
        if not math.isfinite(speed_deg_s) or not 0 < speed_deg_s <= COMMISSIONING_MAX_SPEED_DEG_S:
            raise ValueError(
                f"Speed must be between 0 and {COMMISSIONING_MAX_SPEED_DEG_S:g} deg/s."
            )
        name, arm = next(iter(self._arms.items()))
        arm.poll_feedback()
        self._check_commissioning_health(arm, require_stationary=True)
        for joint, target in joints.items():
            self._validate_position_target(name, joint, target)
        arm.capture_current_as_setpoint()
        arm.move_joints(
            dict(joints), speed_deg_s, self._config.tracking_acceleration_deg_s2
        )
        self._position_targets = dict(joints)
        self._position_speeds = {}
        self._position_mode = "movej"
        self._jog_direction = 0
        self._jog_speeds = {}
        self._commissioning_last_step = self._clock()
        self._position_status = "running"
        self._jog_deadline = self._clock() + self._config.commissioning_watchdog_s

    def _validate_position_target(self, name: str, joint: str, target: float) -> None:
        origin = self._commissioning_origins[name][joint]
        limit = self._config.limit_for(joint)
        low = max(limit.minimum, origin - self._config.commissioning_excursion_deg)
        high = min(limit.maximum, origin + self._config.commissioning_excursion_deg)
        if not math.isfinite(target) or not low <= target <= high:
            raise ValueError(f"{joint}: target {target:.1f} is outside {low:.1f} .. {high:.1f} deg.")

    def plan_joint_sequence(self, steps: tuple[tuple[str, float, float], ...]) -> tuple[dict[str, float], ...]:
        """Resolve all relative steps from one fresh pose without sending motion."""
        if not self._commissioning or len(self._commissioning_origins) != 1:
            raise ValueError("Capture the stationary robot pose before starting a sequence.")
        if not steps:
            raise ValueError("Add at least one sequence step.")
        name, arm = next(iter(self._arms.items()))
        arm.poll_feedback()
        self._check_commissioning_health(arm, require_stationary=True)
        position = dict(arm.feedback_joints)
        planned = []
        for number, (joint, delta, speed) in enumerate(steps, 1):
            if joint not in JOINT_NAMES or not math.isfinite(delta) or not math.isfinite(speed) or not 0 < speed <= COMMISSIONING_MAX_SPEED_DEG_S:
                raise ValueError(f"Step {number}: invalid joint, angle or speed.")
            position[joint] += delta
            try:
                self._validate_position_target(name, joint, position[joint])
            except ValueError as error:
                raise ValueError(f"Step {number}: {error}") from error
            planned.append({joint: position[joint]})
        return tuple(planned)

    def refresh_position_move(self) -> None:
        if self._position_targets:
            now = self._clock()
            if now > self._jog_deadline:
                self.pause()
                return
            self._jog_deadline = now + self._config.commissioning_watchdog_s

    @property
    def position_move_active(self) -> bool:
        return bool(self._position_targets)

    @property
    def position_move_status(self) -> str:
        return self._position_status

    def _commissioning_tick(self) -> None:
        if not self._commissioning or not self._commissioning_origins:
            return
        now = self._clock()
        if self._position_targets:
            if now > self._jog_deadline:
                self.pause()
                return
            elapsed = now - self._commissioning_last_step
            if elapsed < self._config.send_interval:
                return
            try:
                arm = next(iter(self._arms.values()))
                arm.poll_feedback()
                self._check_commissioning_health(arm)
                actual = arm.feedback_joints
                settled = all(abs(actual[j] - v) <= 0.3 for j, v in self._position_targets.items())
                if settled and max(abs(v) for v in arm.feedback_speeds_deg_s.values()) < 0.5:
                    self.pause()
                    self._position_status = "completed"
                    return
                if self._position_mode == "movej":
                    # The controller is executing the single movej. Re-sending
                    # anything here would turn one move back into a stream.
                    self._commissioning_last_step = now
                    return
                arm.update(ArmTargets(joints=self._position_targets), min(elapsed, self._config.send_interval * MAX_CATCHUP_INTERVALS), speed_limits=self._position_speeds)
                self._commissioning_last_step = now
            except BaseException as error:
                self._fault = str(error)
                self.close()
                raise
            return
        if self._jog_direction and now > self._jog_deadline:
            self.pause()
            return
        if self._jog_speeds and now > self._jog_deadline:
            self.pause()
            return
        if not self._jog_speeds:
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
                arm.commissioning_step_multi(
                    self._jog_speeds,
                    self._commissioning_origins[name],
                    self._config.commissioning_excursion_deg,
                    self._config.commissioning_watchdog_s,
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
        allowed_safety_states = (
            (2,) if self._config.commissioning_require_reduced else (1, 2)
        )
        if arm._feedback.get("safety_status") not in allowed_safety_states:
            raise ControlFault(
                f"{arm.name}: controller safety status blocks commissioning."
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
        self._position_status = "stopped"
        self._position_targets = {}
        self._position_speeds = {}
        self._jog_speeds = {}
        self._jog_deadline = 0.0
        self._tracker = TargetTracker()
        self._last_send_at = None
        self._next_send_at = 0.0
        self._last_follow_at = None
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
        if self._follow_interval_s > 0:
            self._follow_step(now)
            return
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
                sample = arm.poll_feedback()
                if sample:
                    self._log_robot_feedback(arm, sample, now)
                arm.check_control_health()
            for name, arm in self._arms.items():
                raw = accumulated.arm(name)
                bounded = self._bounded_tracking_targets(name, raw)
                arm.update(bounded, elapsed)
                self._log_tracking_sample(name, raw, bounded, arm, now)
        except ControlFault as error:
            self._fault = str(error)
            self.close()
            raise

    def set_follow_interval(self, seconds: float) -> None:
        """Switch between streaming and sampled following while running."""
        if not math.isfinite(seconds) or not 0 <= seconds <= MAX_FOLLOW_INTERVAL_S:
            raise ControlFault(
                f"Follow interval must be between 0 and {MAX_FOLLOW_INTERVAL_S:g} seconds."
            )
        if seconds == self._follow_interval_s:
            return
        self._follow_interval_s = seconds
        self._last_follow_at = None
        # Re-seat every setpoint on the measured pose, so the other mode starts
        # from where the arm really is rather than where the last one left it.
        if self._arms and self._tracking_origins:
            self.pause()

    @property
    def follow_interval_s(self) -> float:
        return self._follow_interval_s

    def _follow_step(self, now: float) -> None:
        """Sample the pose every follow interval and send one movej per arm.

        Between samples nothing is streamed; the controller executes the move it
        was given. A sample is skipped while an arm is still travelling, because
        a movej that replaces a running one stops the arm before restarting it.
        After three intervals the wait ends regardless, so a noisy velocity
        reading cannot freeze following for good. This is the variant that ran
        on the physical arm without a fault; a streamed variant was tried and
        withdrawn after the controller reported it could not follow the path.
        """
        accumulated = self._tracker.accumulated_targets()
        if accumulated is None:
            return
        try:
            for arm in self._arms.values():
                sample = arm.poll_feedback()
                if sample:
                    self._log_robot_feedback(arm, sample, now)
                arm.check_control_health()
            # Gestures act the moment they are seen; they never wait for a
            # sample, and over its own socket the gripper never disturbs motion.
            for name, arm in self._arms.items():
                before = arm.gripper_state
                arm.send_gripper(accumulated.arm(name).gripper)
                if arm.gripper_state != before:
                    self._write_telemetry("gripper_command", arm=name, gripper=arm.gripper_state)
            if self._last_follow_at is not None:
                waited = now - self._last_follow_at
                if waited < self._follow_interval_s:
                    return
                still_moving = any(
                    speeds is not None
                    and max(abs(value) for value in speeds.values()) > COMMISSIONING_STATIONARY_DEG_S
                    for speeds in (arm.feedback_speeds_deg_s for arm in self._arms.values())
                )
                if still_moving and waited < 3.0 * self._follow_interval_s:
                    return
            for name, arm in self._arms.items():
                raw = accumulated.arm(name)
                bounded = self._bounded_tracking_targets(name, raw)
                if bounded.joints:
                    arm.move_joints(
                        bounded.joints,
                        self._config.max_speed_deg_s,
                        self._config.tracking_acceleration_deg_s2,
                    )
                self._log_tracking_sample(name, raw, bounded, arm, now)
            self._last_follow_at = now
        except ControlFault as error:
            self._fault = str(error)
            self.close()
            raise

    def _catch_up_interval(self, now: float) -> float:
        """Time credited to the ramp. A long pose gap must not buy one huge step."""
        if self._last_send_at is None:
            return self._config.send_interval
        return min(now - self._last_send_at, self._config.send_interval * MAX_CATCHUP_INTERVALS)

    def _bounded_tracking_targets(self, name: str, targets: ArmTargets) -> ArmTargets:
        if self._config.operation not in (OPERATION_TRACKING, OPERATION_KEYFRAME):
            return targets
        origin = self._tracking_origins.get(name)
        if origin is None:
            raise ControlFault(f"{name}: capture the tracking origin before sending targets.")
        bounded = {}
        for joint, value in targets.joints.items():
            if joint not in origin:
                continue
            # The base sweeps the whole arm across the table; it never gets the
            # room the pitch joints are allowed.
            excursion = (
                self._config.base_excursion_deg
                if joint == JOINT_BASE
                else self._config.tracking_excursion_deg
            )
            bounded[joint] = max(origin[joint] - excursion, min(origin[joint] + excursion, value))
        return ArmTargets(
            joints=bounded,
            gripper=targets.gripper,
            tcp_target=targets.tcp_target,
        )

    def _log_tracking_sample(
        self,
        name: str,
        raw: ArmTargets,
        bounded: ArmTargets,
        arm: URArm,
        now: float,
    ) -> None:
        if self._telemetry is None or now < self._next_telemetry_at:
            return
        self._next_telemetry_at = now + 0.2
        actual = arm.feedback_joints
        speeds = arm.feedback_speeds_deg_s
        self._write_telemetry(
            "tracking_sample",
            arm=name,
            raw_target_deg=raw.joints,
            bounded_target_deg=bounded.joints,
            sent_setpoint_deg=dict(arm._setpoints.joints),
            actual_joint_deg=actual,
            actual_speed_deg_s=speeds,
            robot_mode=ROBOT_MODES.get(arm._feedback.get("robot_mode"), "UNKNOWN"),
            safety_status=SAFETY_STATUSES.get(
                arm._feedback.get("safety_status"), "UNKNOWN"
            ),
            settings={
                "max_speed_deg_s": self._config.max_speed_deg_s,
                "acceleration_deg_s2": self._config.tracking_acceleration_deg_s2,
                "excursion_deg": self._config.tracking_excursion_deg,
                "deadband_deg": self._config.joint_deadband_deg,
            },
        )

    def _write_telemetry(self, event: str, **fields: Any) -> None:
        if self._telemetry is None:
            return
        record = {"time_unix_s": time.time(), "event": event, **fields}
        self._telemetry.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _log_robot_feedback(
        self, arm: URArm, sample: dict[str, Any], monotonic_s: float
    ) -> None:
        actual_q = sample.get("actual_q")
        actual_qd = sample.get("actual_qd")
        self._write_feedback(
            "rtde_sample",
            arm=arm.name,
            host=arm.host,
            monotonic_s=round(monotonic_s, 6),
            actual_joint_rad=actual_q,
            actual_joint_deg=(
                {name: math.degrees(actual_q[index]) for index, name in enumerate(JOINT_NAMES)}
                if actual_q and len(actual_q) == len(JOINT_NAMES)
                else None
            ),
            actual_speed_rad_s=actual_qd,
            actual_speed_deg_s=(
                {name: math.degrees(actual_qd[index]) for index, name in enumerate(JOINT_NAMES)}
                if actual_qd and len(actual_qd) == len(JOINT_NAMES)
                else None
            ),
            actual_tcp_pose=sample.get("actual_TCP_pose"),
            actual_tcp_speed=sample.get("actual_TCP_speed"),
            robot_mode=sample.get("robot_mode"),
            robot_mode_name=ROBOT_MODES.get(sample.get("robot_mode"), "UNKNOWN"),
            safety_status=sample.get("safety_status"),
            safety_status_name=SAFETY_STATUSES.get(sample.get("safety_status"), "UNKNOWN"),
            speed_scaling=sample.get("speed_scaling"),
            target_speed_fraction=sample.get("target_speed_fraction"),
            runtime_state=sample.get("runtime_state"),
        )

    def _write_feedback(self, event: str, **fields: Any) -> None:
        if self._feedback_log is None:
            return
        record = {"time_unix_s": time.time(), "event": event, **fields}
        self._feedback_log.write(json.dumps(record, separators=(",", ":")) + "\n")

    def note_tracking_event(self, event: str, elapsed_s: float) -> None:
        """Record camera gaps without flooding the JSONL file every video frame."""
        now = self._clock()
        if event == "tracking_gap_held" and now < self._next_tracking_event_at:
            return
        self._next_tracking_event_at = now + 0.1
        self._write_telemetry(event, gap_elapsed_s=round(elapsed_s, 4))

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
        if not self._commissioning:
            lines.append(
                f"follow: one movej every {self._follow_interval_s:.2f} s  (, / . adjust)"
                if self._follow_interval_s > 0
                else f"follow: servoj stream every {self._config.send_interval * 1000:.0f} ms  (, / . adjust)"
            )
        if self._commissioning:
            state = "armed" if self._commissioning_origins else "not armed"
            lines.append(
                f"commissioning {state}: {self._config.commissioning_joint}, "
                f"{self._commissioning_speed_deg_s:g} deg/s, "
                f"+/-{self._config.commissioning_excursion_deg:g} deg"
            )
        return lines

    def close(self) -> None:
        for arm in self._arms.values():
            arm.close()
        if self._telemetry is not None:
            self._write_telemetry("session_closed")
            self._telemetry.close()
            self._telemetry = None
        if self._feedback_log is not None:
            self._write_feedback("session_closed")
            self._feedback_log.close()
            self._feedback_log = None


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
