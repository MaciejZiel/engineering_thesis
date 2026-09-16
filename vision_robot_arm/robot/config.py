from dataclasses import dataclass

from vision_robot_arm.robot.targets import (
    JOINT_ELBOW,
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

UR_SECONDARY_PORT = 30002
UR7E_MAX_JOINT_SPEED_DEG_S = 180.0


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
    source="shoulder", offset_deg=-180.0, sign=1.0, limit=JointLimit(-180.0, 0.0)
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
    print_interval: float = 0.5
    right_host: str | None = None
    left_host: str | None = None
    ur_port: int = UR_SECONDARY_PORT
    servo_gain: int = 300
    servo_lookahead_s: float = 0.1
    port: str | None = None
    baud_rate: int = 115200
    send_interval: float = 0.05
    max_speed_deg_s: float = 60.0
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
        if self.backend not in BACKEND_CHOICES:
            choices = ", ".join(BACKEND_CHOICES)
            raise SystemExit(f"--robot-backend must be one of: {choices}")
        if self.print_interval <= 0:
            raise SystemExit("--robot-print-interval must be greater than 0")
        if self.send_interval <= 0:
            raise SystemExit("--robot-send-interval must be greater than 0")
        if not 0 < self.max_speed_deg_s <= UR7E_MAX_JOINT_SPEED_DEG_S:
            raise SystemExit(
                f"--robot-max-speed must be between 0 and {UR7E_MAX_JOINT_SPEED_DEG_S:.0f} deg/s (UR7e limit)"
            )
        if self.baud_rate <= 0:
            raise SystemExit("--robot-baud must be greater than 0")
        if self.joint_deadband_deg < 0:
            raise SystemExit("--robot-deadband must be 0 or greater")
        if not 0 < self.ur_port < 65536:
            raise SystemExit("--robot-ur-port must be between 1 and 65535")
        if self.servo_gain < 100 or self.servo_gain > 2000:
            raise SystemExit("--robot-servo-gain must be between 100 and 2000")
        if self.servo_lookahead_s < 0.03 or self.servo_lookahead_s > 0.2:
            raise SystemExit("--robot-servo-lookahead must be between 0.03 and 0.2 seconds")
        if self.backend == BACKEND_UR and not self.hosts:
            raise SystemExit(
                "--robot-right-host and/or --robot-left-host is required with --robot-backend ur"
            )
        if self.backend == BACKEND_SERIAL and not self.port:
            raise SystemExit("--robot-port is required with --robot-backend serial")
        for name, mapping in (("shoulder", self.shoulder), ("elbow", self.elbow), ("wrist", self.wrist)):
            if mapping.limit.minimum >= mapping.limit.maximum:
                raise SystemExit(f"{name} joint limit minimum must be below its maximum")
            if mapping.sign == 0:
                raise SystemExit(f"{name} joint mapping sign must not be 0")
