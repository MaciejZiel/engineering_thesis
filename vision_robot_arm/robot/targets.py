import math
from dataclasses import dataclass, field

ARM_RIGHT = "right"
ARM_LEFT = "left"
ARM_NAMES = (ARM_RIGHT, ARM_LEFT)

JOINT_SHOULDER = "shoulder"
JOINT_ELBOW = "elbow"
JOINT_WRIST = "wrist"
JOINT_NAMES = (JOINT_SHOULDER, JOINT_ELBOW, JOINT_WRIST)

GRIPPER_OPEN = "open"
GRIPPER_CLOSE = "close"


@dataclass(frozen=True)
class ArmTargets:
    joints: dict[str, float] = field(default_factory=dict)
    gripper: str | None = None

    @property
    def has_data(self) -> bool:
        return bool(self.joints) or self.gripper is not None

    def format(self) -> str:
        parts = [f"{name}={value:5.1f}" for name, value in self.joints.items()]
        if self.gripper is not None:
            parts.append(f"gripper={self.gripper}")
        return " ".join(parts) if parts else "no data"


@dataclass(frozen=True)
class JointTargets:
    timestamp_ms: int
    arms: dict[str, ArmTargets] = field(default_factory=dict)
    lift_mode: bool = False

    @property
    def has_data(self) -> bool:
        return any(arm.has_data for arm in self.arms.values()) or self.lift_mode

    def arm(self, name: str) -> ArmTargets:
        return self.arms.get(name, ArmTargets())

    def format(self) -> str:
        parts = [f"{name[0].upper()}: {arm.format()}" for name, arm in self.arms.items() if arm.has_data]
        parts.append(f"lift_mode={'on' if self.lift_mode else 'off'}")
        return " | ".join(parts)


@dataclass(frozen=True)
class ArmState:
    joints: dict[str, float]
    targets: dict[str, float]
    gripper: str

    @property
    def settled(self) -> bool:
        return all(
            math.isclose(self.joints[name], self.targets[name], abs_tol=1e-6)
            for name in self.joints
        )


@dataclass(frozen=True)
class RobotState:
    arms: dict[str, ArmState]
    lift_mode: bool

    def arm(self, name: str) -> ArmState | None:
        return self.arms.get(name)
