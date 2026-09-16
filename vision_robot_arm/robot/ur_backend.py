"""Drive real UR7e cobots over URScript, with the safeguards a real arm needs.

Streaming raw mapped angles at a moving person would make the controller chase steps
of tens of degrees. Every arm therefore owns a setpoint generator that ramps toward the
mapped pose at `--robot-max-speed`, starts from a known home pose, and decelerates on
shutdown. Feedback comes from RTDE, so the on-screen twin shows the robot, not the wish.
"""

import math
import socket
import time
from typing import Any, Callable

from vision_robot_arm.robot.backend import Clock, TargetTracker
from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.simulation import SimulatedArm
from vision_robot_arm.robot.targets import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    JOINT_NAMES,
    MAPPED_JOINTS,
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
# With feedback the homing window ends when the arm has arrived, not when a timer says so.
HOME_TOLERANCE_DEG = 3.0
HOMING_TIMEOUT_FACTOR = 5.0
JOINT_SHORT_NAMES = {"shoulder": "S", "elbow": "E", "wrist_1": "W1"}

Connector = Callable[[str, int], Any]
RtdeFactory = Callable[[str, int], RtdeClient | None]
StatusQuery = Callable[[str, int], Any]


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


def encode_movej(joints_deg: dict[str, float], speed_deg_s: float, accel_deg_s2: float) -> bytes:
    command = (
        f"movej({_joint_vector(joints_deg)}, "
        f"a={math.radians(accel_deg_s2):.3f}, v={math.radians(speed_deg_s):.3f})\n"
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
        self._rtde: RtdeClient | None = None
        self._homed = False
        self._ready_at = clock() + config.start_seconds
        self._homing_deadline = self._ready_at + config.start_seconds * HOMING_TIMEOUT_FACTOR
        try:
            self._rtde = rtde_factory(host, config.rtde_port) if rtde_factory is not None else None
            self._send(
                encode_movej(
                    self._setpoints.joints, config.start_speed_deg_s, config.start_accel_deg_s2
                )
            )
        except BaseException:
            self._release()
            raise

    @property
    def homing(self) -> bool:
        """True until the arm reaches the home pose; latched, because a servo always lags."""
        if self._homed:
            return False
        now = self._clock()
        if now < self._ready_at:
            return True
        actual = self.feedback_joints
        if now < self._homing_deadline and actual is not None and any(
            abs(actual[joint] - target) > HOME_TOLERANCE_DEG
            for joint, target in self._setpoints.joints.items()
        ):
            return True
        self._homed = True
        return False

    @property
    def feedback_joints(self) -> dict[str, float] | None:
        actual = self._feedback.get("actual_q")
        if not actual or len(actual) < len(JOINT_NAMES):
            return None
        return {name: math.degrees(actual[index]) for index, name in enumerate(JOINT_NAMES)}

    def update(self, targets: ArmTargets, elapsed_s: float) -> None:
        """Move the setpoint toward the mapped pose, then command that setpoint."""
        self._setpoints.set_targets(targets.joints, targets.gripper)
        if self.homing:
            # The arm is still driving to the home pose; leave the setpoint there so
            # the first servoj continues from where the robot actually is.
            return
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
        elif not self._rtde.connected:
            # Never present a stale pose as the live one.
            self._feedback = {}

    def state(self) -> ArmState:
        return ArmState(
            joints=self.feedback_joints or dict(self._setpoints.joints),
            targets=dict(self._setpoints.targets),
            gripper=self._gripper or GRIPPER_OPEN,
        )

    def status_line(self) -> str:
        health = self._health()
        if self.homing:
            detail = f"homing for {self._ready_at - self._clock():.1f}s"
        else:
            joints = self.feedback_joints or self._setpoints.joints
            detail = " ".join(
                f"{JOINT_SHORT_NAMES[joint]} {joints[joint]:6.1f}"
                for joint in MAPPED_JOINTS
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
    ) -> None:
        self._config = config
        self._clock = clock
        if config.preflight:
            _preflight(config, status_query)
        factory = rtde_factory if config.feedback else None
        self._arms: dict[str, URArm] = {}
        try:
            for name, host in config.hosts.items():
                self._arms[name] = URArm(name, host, config, connector, factory, clock)
        except SystemExit:
            self.close()
            raise
        self._tracker = TargetTracker()
        self._next_send_at = 0.0
        self._last_send_at: float | None = None

    def send(self, targets: JointTargets) -> None:
        if not targets.has_data:
            return
        self._tracker.update(targets)
        now = self._clock()
        if now < self._next_send_at:
            return
        elapsed = self._catch_up_interval(now)
        self._next_send_at = now + self._config.send_interval
        self._last_send_at = now

        for name, arm in self._arms.items():
            arm.poll_feedback()
            arm.update(targets.arm(name), elapsed)

    def _catch_up_interval(self, now: float) -> float:
        """Time credited to the ramp. A long pose gap must not buy one huge step."""
        if self._last_send_at is None:
            return self._config.send_interval
        return min(now - self._last_send_at, self._config.send_interval * MAX_CATCHUP_INTERVALS)

    def robot_state(self) -> RobotState | None:
        if not self._arms:
            return None
        for arm in self._arms.values():
            arm.poll_feedback()
        return RobotState(
            arms={name: arm.state() for name, arm in self._arms.items()},
            lift_mode=bool(self._tracker.last_targets and self._tracker.last_targets.lift_mode),
        )

    def status_lines(self) -> list[str]:
        return [arm.status_line() for arm in self._arms.values()]

    def close(self) -> None:
        for arm in self._arms.values():
            arm.close()


def _preflight(config: RobotConfig, status_query: StatusQuery) -> None:
    for name, host in config.hosts.items():
        status = status_query(host, config.dashboard_port)
        if status is None:
            continue
        problem = status.blocking_problem()
        if problem:
            raise SystemExit(
                f"The {name} UR7e at {host} is not ready to run URScript: {problem}.\n"
                f"Dashboard reports {status.describe()}. "
                "Start with --no-robot-preflight to skip this check."
            )


def _joint_vector(joints_deg: dict[str, float]) -> str:
    pose = full_joint_pose(joints_deg)
    return "[" + ", ".join(f"{math.radians(pose[name]):.4f}" for name in JOINT_NAMES) + "]"
