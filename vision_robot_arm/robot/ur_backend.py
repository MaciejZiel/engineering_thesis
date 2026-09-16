import math
import socket
import time
from typing import Any, Callable

from vision_robot_arm.robot.backend import Clock, TargetTracker
from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.targets import (
    GRIPPER_CLOSE,
    JOINT_NAMES,
    MAPPED_JOINTS,
    JointTargets,
    RobotState,
    full_joint_pose,
)

CONNECT_TIMEOUT_S = 2.0
GRIPPER_TOOL_OUTPUT = 0

Connector = Callable[[str, int], Any]


def encode_servoj(
    joints_deg: dict[str, float],
    duration_s: float,
    lookahead_s: float,
    gain: int,
) -> bytes:
    pose = full_joint_pose(joints_deg)
    radians = ", ".join(f"{math.radians(pose[name]):.4f}" for name in JOINT_NAMES)
    command = f"servoj([{radians}], 0, 0, {duration_s:.3f}, {lookahead_s:.3f}, {gain:d})\n"
    return command.encode("ascii")


def encode_gripper(closed: bool, tool_output: int = GRIPPER_TOOL_OUTPUT) -> bytes:
    value = "True" if closed else "False"
    return f"set_tool_digital_out({tool_output}, {value})\n".encode("ascii")


def default_connector(host: str, port: int) -> Any:
    return socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S)


class URArm:
    def __init__(self, name: str, host: str, port: int, connector: Connector) -> None:
        self.name = name
        self.host = host
        self.port = port
        try:
            self._socket = connector(host, port)
        except OSError as error:
            raise SystemExit(
                f"Could not connect to {name} UR7e at {host}:{port}: {error}\n"
                "Check the IP address and switch the robot to Remote Control mode."
            ) from error
        self.last_joints: dict[str, float] = {}
        self.last_gripper: str | None = None

    def send_joints(self, joints_deg: dict[str, float], duration_s: float, lookahead_s: float, gain: int) -> None:
        self._socket.sendall(encode_servoj(joints_deg, duration_s, lookahead_s, gain))
        self.last_joints = dict(joints_deg)

    def send_gripper(self, gripper: str) -> None:
        if gripper == self.last_gripper:
            return
        self._socket.sendall(encode_gripper(gripper == GRIPPER_CLOSE))
        self.last_gripper = gripper

    def close(self) -> None:
        self._socket.close()


class URBackend:
    def __init__(
        self,
        config: RobotConfig,
        connector: Connector = default_connector,
        clock: Clock = time.monotonic,
    ) -> None:
        self._config = config
        self._clock = clock
        self._arms = {
            name: URArm(name, host, config.ur_port, connector) for name, host in config.hosts.items()
        }
        self._tracker = TargetTracker()
        self._next_send_at = 0.0

    def send(self, targets: JointTargets) -> None:
        if not targets.has_data:
            return
        self._tracker.update(targets)
        now = self._clock()
        if now < self._next_send_at:
            return
        self._next_send_at = now + self._config.send_interval

        state = self._tracker.robot_state()
        if state is None:
            return
        for name, arm in self._arms.items():
            arm_state = state.arm(name)
            if arm_state is None or not arm_state.joints:
                continue
            arm.send_joints(
                arm_state.joints,
                self._config.send_interval,
                self._config.servo_lookahead_s,
                self._config.servo_gain,
            )
            arm.send_gripper(arm_state.gripper)

    def robot_state(self) -> RobotState | None:
        return self._tracker.robot_state()

    def status_lines(self) -> list[str]:
        lines = []
        for name, arm in self._arms.items():
            if arm.last_joints:
                joints = " ".join(
                    f"{joint} {arm.last_joints[joint]:6.1f}"
                    for joint in MAPPED_JOINTS
                    if joint in arm.last_joints
                )
            else:
                joints = "waiting for pose"
            gripper = arm.last_gripper or "n/a"
            lines.append(f"ur {name[0].upper()} {arm.host}:{arm.port} {joints} grip {gripper}")
        return lines

    def close(self) -> None:
        for arm in self._arms.values():
            arm.close()
