from dataclasses import dataclass
import math

from vision_robot_arm.robot.targets import (
    JOINT_ELBOW,
    JOINT_NAMES,
    JOINT_SHOULDER,
    JOINT_WRIST_1,
    UR_HOME_DEG,
)

BACKEND_NONE = "none"
BACKEND_DEBUG = "debug"
BACKEND_SIM = "sim"
BACKEND_UR = "ur"
BACKEND_SERIAL = "serial"
BACKEND_CHOICES = (BACKEND_NONE, BACKEND_DEBUG, BACKEND_SIM, BACKEND_UR, BACKEND_SERIAL)

OPERATION_MONITOR = "monitor"
OPERATION_COMMISSIONING = "commissioning"
OPERATION_TRACKING = "tracking"
OPERATION_CHOICES = (
    OPERATION_MONITOR,
    OPERATION_COMMISSIONING,
    OPERATION_TRACKING,
)

UR_SECONDARY_PORT = 30002
UR_RTDE_PORT = 30004
UR_DASHBOARD_PORT = 29999
UR7E_MAX_JOINT_SPEED_DEG_S = 180.0
COMMISSIONING_MAX_SPEED_DEG_S = 30.0
COMMISSIONING_MAX_EXCURSION_DEG = 80.0


@dataclass(frozen=True)
class JointLimit:
    minimum: float = -360.0
    maximum: float = 360.0

    def clamp(self, value: float) -> float:
        return max(self.minimum, min(self.maximum, value))


UR7E_JOINT_RANGE = JointLimit(-360.0, 360.0)
UR7E_ELBOW_RANGE = JointLimit(-160.0, 160.0)


@dataclass(frozen=True)
class JointMapping:
    source: str
    offset_deg: float
    sign: float
    limit: JointLimit

    def to_robot(self, body_angle_deg: float) -> float:
        return self.limit.clamp(self.offset_deg + self.sign * body_angle_deg)

    def to_body(self, robot_angle_deg: float) -> float:
        return (robot_angle_deg - self.offset_deg) / self.sign


DEFAULT_SHOULDER_MAPPING = JointMapping(
    source="shoulder_elevation", offset_deg=-180.0, sign=1.0, limit=JointLimit(-180.0, 0.0)
)
DEFAULT_ELBOW_MAPPING = JointMapping(
    source="elbow", offset_deg=180.0, sign=-1.0, limit=UR7E_ELBOW_RANGE
)
DEFAULT_WRIST_MAPPING = JointMapping(
    source="wrist", offset_deg=180.0, sign=-1.0, limit=JointLimit(-180.0, 180.0)
)


@dataclass(frozen=True)
class RobotConfig:
    backend: str = BACKEND_NONE
    operation: str = OPERATION_MONITOR
    print_interval: float = 0.5
    right_host: str | None = None
    left_host: str | None = None
    ur_port: int = UR_SECONDARY_PORT
    rtde_port: int = UR_RTDE_PORT
    dashboard_port: int = UR_DASHBOARD_PORT
    feedback: bool = True
    preflight: bool = True
    servo_gain: int = 300
    servo_lookahead_s: float = 0.1
    tool_output: int = 0
    port: str | None = None
    baud_rate: int = 115200
    send_interval: float = 0.05
    max_speed_deg_s: float = 60.0
    tracking_excursion_deg: float = 40.0
    tracking_acceleration_deg_s2: float = 7.0
    telemetry_log_path: str | None = None
    commissioning_joint: str = JOINT_SHOULDER
    commissioning_speed_deg_s: float = 30.0
    commissioning_excursion_deg: float = 80.0
    commissioning_watchdog_s: float = 0.15
    commissioning_require_reduced: bool = True
    joint_deadband_deg: float = 1.5
    shoulder: JointMapping = DEFAULT_SHOULDER_MAPPING
    elbow: JointMapping = DEFAULT_ELBOW_MAPPING
    wrist: JointMapping = DEFAULT_WRIST_MAPPING

    @property
    def enabled(self) -> bool:
        return self.backend != BACKEND_NONE

    @property
    def hosts(self) -> dict[str, str]:
        hosts = {}
        if self.right_host:
            hosts["right"] = self.right_host
        if self.left_host:
            hosts["left"] = self.left_host
        return hosts

    def mapping_for(self, joint: str) -> JointMapping | None:
        if joint == JOINT_SHOULDER:
            return self.shoulder
        if joint == JOINT_ELBOW:
            return self.elbow
        if joint == JOINT_WRIST_1:
            return self.wrist
        return None

    def limit_for(self, joint: str) -> JointLimit:
        mapping = self.mapping_for(joint)
        if mapping is not None:
            return mapping.limit
        if joint == JOINT_ELBOW:
            return UR7E_ELBOW_RANGE
        return UR7E_JOINT_RANGE

    def home_for(self, joint: str) -> float:
        return self.limit_for(joint).clamp(UR_HOME_DEG.get(joint, 0.0))

    def validate(self) -> None:
        for flag, value in (
            ("--robot-print-interval", self.print_interval),
            ("--robot-send-interval", self.send_interval),
            ("--robot-max-speed", self.max_speed_deg_s),
            ("--robot-tracking-excursion", self.tracking_excursion_deg),
            ("--robot-tracking-acceleration", self.tracking_acceleration_deg_s2),
            ("--robot-deadband", self.joint_deadband_deg),
            ("--robot-servo-lookahead", self.servo_lookahead_s),
            ("--robot-commissioning-speed", self.commissioning_speed_deg_s),
            ("--robot-commissioning-excursion", self.commissioning_excursion_deg),
            ("--robot-commissioning-watchdog", self.commissioning_watchdog_s),
        ):
            _validate_finite_number(flag, value)
        for flag, value in (
            ("--robot-baud", self.baud_rate),
            ("--robot-ur-port", self.ur_port),
            ("--robot-rtde-port", self.rtde_port),
            ("--robot-dashboard-port", self.dashboard_port),
            ("--robot-tool-output", self.tool_output),
            ("--robot-servo-gain", self.servo_gain),
        ):
            if type(value) is not int:
                raise SystemExit(f"{flag} must be an integer")
        if self.backend not in BACKEND_CHOICES:
            choices = ", ".join(BACKEND_CHOICES)
            raise SystemExit(f"--robot-backend must be one of: {choices}")
        if self.operation not in OPERATION_CHOICES:
            choices = ", ".join(OPERATION_CHOICES)
            raise SystemExit(f"--robot-operation must be one of: {choices}")
        if self.commissioning_joint not in JOINT_NAMES:
            raise SystemExit(
                "--robot-commissioning-joint must be one of: "
                + ", ".join(JOINT_NAMES)
            )
        if not 0 < self.commissioning_speed_deg_s <= COMMISSIONING_MAX_SPEED_DEG_S:
            raise SystemExit(
                f"--robot-commissioning-speed must be between 0 and {COMMISSIONING_MAX_SPEED_DEG_S:g} deg/s"
            )
        if not 0 < self.commissioning_excursion_deg <= COMMISSIONING_MAX_EXCURSION_DEG:
            raise SystemExit(
                f"--robot-commissioning-excursion must be between 0 and {COMMISSIONING_MAX_EXCURSION_DEG:g} degrees"
            )
        if not 0.05 <= self.commissioning_watchdog_s <= 0.5:
            raise SystemExit(
                "--robot-commissioning-watchdog must be between 0.05 and 0.5 seconds"
            )
        if self.print_interval <= 0:
            raise SystemExit("--robot-print-interval must be greater than 0")
        if self.send_interval <= 0:
            raise SystemExit("--robot-send-interval must be greater than 0")
        if not 0 < self.max_speed_deg_s <= UR7E_MAX_JOINT_SPEED_DEG_S:
            raise SystemExit(
                f"--robot-max-speed must be between 0 and {UR7E_MAX_JOINT_SPEED_DEG_S:.0f} deg/s (UR7e limit)"
            )
        if not 0 < self.tracking_excursion_deg <= COMMISSIONING_MAX_EXCURSION_DEG:
            raise SystemExit(
                "--robot-tracking-excursion must be between 0 and "
                f"{COMMISSIONING_MAX_EXCURSION_DEG:g} degrees"
            )
        if not 0 < self.tracking_acceleration_deg_s2 <= UR7E_MAX_JOINT_SPEED_DEG_S:
            raise SystemExit(
                "--robot-tracking-acceleration must be between 0 and "
                f"{UR7E_MAX_JOINT_SPEED_DEG_S:g} deg/s^2"
            )
        if self.baud_rate <= 0:
            raise SystemExit("--robot-baud must be greater than 0")
        if self.joint_deadband_deg < 0:
            raise SystemExit("--robot-deadband must be 0 or greater")
        for flag, port in (
            ("--robot-ur-port", self.ur_port),
            ("--robot-rtde-port", self.rtde_port),
            ("--robot-dashboard-port", self.dashboard_port),
        ):
            if not 0 < port < 65536:
                raise SystemExit(f"{flag} must be between 1 and 65535")
        if not 0 <= self.tool_output <= 1:
            raise SystemExit("--robot-tool-output must be 0 or 1")
        if self.servo_gain < 100 or self.servo_gain > 2000:
            raise SystemExit("--robot-servo-gain must be between 100 and 2000")
        if self.servo_lookahead_s < 0.03 or self.servo_lookahead_s > 0.2:
            raise SystemExit("--robot-servo-lookahead must be between 0.03 and 0.2 seconds")
        if self.backend == BACKEND_UR and not self.hosts:
            raise SystemExit(
                "--robot-right-host and/or --robot-left-host is required with --robot-backend ur"
            )
        if self.backend == BACKEND_UR and self.operation == OPERATION_COMMISSIONING:
            if len(self.hosts) != 1:
                raise SystemExit("Commissioning requires exactly one UR7e host")
            if not self.feedback or not self.preflight:
                raise SystemExit(
                    "Commissioning requires RTDE feedback and dashboard preflight"
                )
        if self.backend == BACKEND_UR and self.operation == OPERATION_TRACKING:
            if not self.feedback or not self.preflight:
                raise SystemExit(
                    "Vision tracking requires RTDE feedback and dashboard preflight"
                )
        if self.backend == BACKEND_SERIAL and not self.port:
            raise SystemExit("--robot-port is required with --robot-backend serial")
        hardware = {
            "shoulder": UR7E_JOINT_RANGE,
            "elbow": UR7E_ELBOW_RANGE,
            "wrist": UR7E_JOINT_RANGE,
        }
        for name, mapping in (("shoulder", self.shoulder), ("elbow", self.elbow), ("wrist", self.wrist)):
            for field, value in (
                ("minimum", mapping.limit.minimum), ("maximum", mapping.limit.maximum),
                ("offset", mapping.offset_deg), ("sign", mapping.sign),
            ):
                _validate_finite_number(f"{name} joint mapping {field}", value)
            if not isinstance(mapping.source, str) or not mapping.source.strip():
                raise SystemExit(f"{name} joint mapping source must not be empty")
            if mapping.limit.minimum >= mapping.limit.maximum:
                raise SystemExit(f"{name} joint limit minimum must be below its maximum")
            if mapping.sign == 0:
                raise SystemExit(f"{name} joint mapping sign must not be 0")
            allowed = hardware[name]
            if mapping.limit.minimum < allowed.minimum or mapping.limit.maximum > allowed.maximum:
                raise SystemExit(
                    f"--robot-{name}-range must stay inside the UR7e limit "
                    f"{allowed.minimum:.0f} {allowed.maximum:.0f}"
                )


def _validate_finite_number(name: str, value: float) -> None:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise SystemExit(f"{name} must be a finite number")
