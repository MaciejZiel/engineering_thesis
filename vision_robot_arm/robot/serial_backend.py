import importlib
import time
from typing import Any, Callable

from vision_robot_arm.robot.backend import Clock, TargetTracker
from vision_robot_arm.robot.targets import (
    ARM_NAMES,
    GRIPPER_CLOSE,
    JOINT_ELBOW,
    JOINT_SHOULDER,
    JOINT_WRIST,
    JointTargets,
    RobotState,
)

ARM_CODES = {arm: arm[0].upper() for arm in ARM_NAMES}
JOINT_CODES = {
    JOINT_SHOULDER: "S",
    JOINT_ELBOW: "E",
    JOINT_WRIST: "W",
}
GRIPPER_CODE = "G"
LIFT_MODE_CODE = "L"
FRAME_TERMINATOR = "\n"

Importer = Callable[[str], Any]


def encode_targets(targets: JointTargets) -> bytes:
    fields = []
    for arm, arm_code in ARM_CODES.items():
        arm_targets = targets.arm(arm)
        for joint, value in arm_targets.joints.items():
            if joint in JOINT_CODES:
                fields.append(f"{arm_code}{JOINT_CODES[joint]}:{value:.1f}")
        if arm_targets.gripper is not None:
            closed = 1 if arm_targets.gripper == GRIPPER_CLOSE else 0
            fields.append(f"{arm_code}{GRIPPER_CODE}:{closed}")
    fields.append(f"{LIFT_MODE_CODE}:{1 if targets.lift_mode else 0}")
    return (";".join(fields) + FRAME_TERMINATOR).encode("ascii")


def load_serial_module(importer: Importer = importlib.import_module) -> Any:
    try:
        return importer("serial")
    except ImportError as error:
        raise SystemExit(
            "pyserial is required for --robot-backend serial.\n"
            "Install it with: python -m pip install pyserial"
        ) from error


class SerialBackend:
    def __init__(
        self,
        port: str,
        baud_rate: int,
        send_interval: float,
        serial_module: Any | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        module = serial_module if serial_module is not None else load_serial_module()
        try:
            self._connection = module.Serial(port, baud_rate, timeout=0)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Could not open serial port {port}: {error}") from error

        self._port = port
        self._baud_rate = baud_rate
        self._send_interval = send_interval
        self._clock = clock
        self._next_send_at = 0.0
        self._last_frame: bytes | None = None
        self._tracker = TargetTracker()

    def send(self, targets: JointTargets) -> None:
        if not targets.has_data and self._tracker.last_targets is None:
            return
        self._tracker.update(targets)
        now = self._clock()
        if now < self._next_send_at:
            return

        # Send the accumulated state, not this one frame: a command that lands between
        # two transmissions must not be lost, and a cleared flag must still be sent.
        accumulated = self._tracker.accumulated_targets()
        if accumulated is None:
            return
        frame = encode_targets(accumulated)
        self._connection.write(frame)
        self._last_frame = frame
        self._next_send_at = now + self._send_interval

    def robot_state(self) -> RobotState | None:
        return self._tracker.robot_state()

    def status_lines(self) -> list[str]:
        last = self._last_frame.decode("ascii").strip() if self._last_frame else "idle"
        return [f"serial {self._port} @ {self._baud_rate}: {last}"]

    def close(self) -> None:
        self._connection.close()
