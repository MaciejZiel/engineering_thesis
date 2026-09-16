from dataclasses import dataclass, field

JOINT_SHOULDER = "shoulder"
JOINT_ELBOW = "elbow"
JOINT_NAMES = (JOINT_SHOULDER, JOINT_ELBOW)

GRIPPER_OPEN = "open"
GRIPPER_CLOSE = "close"


@dataclass(frozen=True)
class JointTargets:
    timestamp_ms: int
    joints: dict[str, float] = field(default_factory=dict)
    gripper: str | None = None
    lift_mode: bool = False

    @property
    def has_data(self) -> bool:
        return bool(self.joints) or self.gripper is not None or self.lift_mode

    def format(self) -> str:
        parts = [f"{name}={value:5.1f}" for name, value in self.joints.items()]
        if self.gripper is not None:
            parts.append(f"gripper={self.gripper}")
        parts.append(f"lift_mode={'on' if self.lift_mode else 'off'}")
        return " | ".join(parts)
