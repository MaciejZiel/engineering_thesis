from dataclasses import dataclass

BACKEND_NONE = "none"
BACKEND_DEBUG = "debug"
BACKEND_SIM = "sim"
BACKEND_SERIAL = "serial"
BACKEND_CHOICES = (BACKEND_NONE, BACKEND_DEBUG, BACKEND_SIM, BACKEND_SERIAL)


@dataclass(frozen=True)
class JointLimit:
    minimum: float = 0.0
    maximum: float = 180.0

    def clamp(self, value: float) -> float:
        return max(self.minimum, min(self.maximum, value))


@dataclass(frozen=True)
class RobotConfig:
    backend: str = BACKEND_NONE
    print_interval: float = 0.5
    port: str | None = None
    baud_rate: int = 115200
    send_interval: float = 0.05
    max_speed_deg_s: float = 90.0
    home_deg: float = 90.0
    joint_deadband_deg: float = 1.5
    shoulder_limit: JointLimit = JointLimit()
    elbow_limit: JointLimit = JointLimit()
    wrist_limit: JointLimit = JointLimit()

    @property
    def enabled(self) -> bool:
        return self.backend != BACKEND_NONE

    def limit_for(self, joint: str) -> JointLimit:
        if joint == "shoulder":
            return self.shoulder_limit
        if joint == "elbow":
            return self.elbow_limit
        if joint == "wrist":
            return self.wrist_limit
        return JointLimit()

    def validate(self) -> None:
        if self.backend not in BACKEND_CHOICES:
            choices = ", ".join(BACKEND_CHOICES)
            raise SystemExit(f"--robot-backend must be one of: {choices}")
        if self.print_interval <= 0:
            raise SystemExit("--robot-print-interval must be greater than 0")
        if self.send_interval <= 0:
            raise SystemExit("--robot-send-interval must be greater than 0")
        if self.max_speed_deg_s <= 0:
            raise SystemExit("--robot-max-speed must be greater than 0")
        if self.baud_rate <= 0:
            raise SystemExit("--robot-baud must be greater than 0")
        if self.joint_deadband_deg < 0:
            raise SystemExit("--robot-deadband must be 0 or greater")
        if self.backend == BACKEND_SERIAL and not self.port:
            raise SystemExit("--robot-port is required with --robot-backend serial")
        limits = (
            ("shoulder", self.shoulder_limit),
            ("elbow", self.elbow_limit),
            ("wrist", self.wrist_limit),
        )
        for name, limit in limits:
            if limit.minimum >= limit.maximum:
                raise SystemExit(f"{name} joint limit minimum must be below its maximum")
