import math
from dataclasses import dataclass, field

ROBOT_MODEL = "UR7e"

ARM_RIGHT = "right"
ARM_LEFT = "left"
ARM_NAMES = (ARM_RIGHT, ARM_LEFT)

JOINT_BASE = "base"
JOINT_SHOULDER = "shoulder"
JOINT_ELBOW = "elbow"
JOINT_WRIST_1 = "wrist_1"
JOINT_WRIST_2 = "wrist_2"
JOINT_WRIST_3 = "wrist_3"
JOINT_WRIST = JOINT_WRIST_1
JOINT_NAMES = (
    JOINT_BASE,
    JOINT_SHOULDER,
    JOINT_ELBOW,
    JOINT_WRIST_1,
    JOINT_WRIST_2,
    JOINT_WRIST_3,
)
# The three pitch joints share parallel horizontal axes (UR DH: alpha 0 for
# joints 2 and 3). They are driven in every tracking space and a session needs
# all of them before it is usable.
MAPPED_JOINTS = (JOINT_SHOULDER, JOINT_ELBOW, JOINT_WRIST_1)
# The three rotation joints - base about the vertical, wrist 2 yaw, wrist 3
# roll - are driven from 3D body angles only. In the 2D tracking space they hold
# the pose captured when control was enabled.
ROTATION_JOINTS = (JOINT_BASE, JOINT_WRIST_2, JOINT_WRIST_3)
REPORTED_JOINTS = JOINT_NAMES
HELD_JOINTS = ROTATION_JOINTS

UR_HOME_DEG = {
    JOINT_BASE: 0.0,
    JOINT_SHOULDER: -90.0,
    JOINT_ELBOW: 0.0,
    JOINT_WRIST_1: -90.0,
    JOINT_WRIST_2: 0.0,
    JOINT_WRIST_3: 0.0,
}

GRIPPER_OPEN = "open"
GRIPPER_CLOSE = "close"


@dataclass(frozen=True)
class ArmTargets:
    joints: dict[str, float] = field(default_factory=dict)
    gripper: str | None = None
    tcp_target: tuple[float, float, float] | None = None

    @property
    def has_data(self) -> bool:
        return bool(self.joints) or self.gripper is not None

    def format(self) -> str:
        parts = [f"{name}={value:6.1f}" for name, value in self.joints.items()]
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
    tcp_target: tuple[float, float, float] | None = None

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


def full_joint_pose(joints: dict[str, float]) -> dict[str, float]:
    pose = dict(UR_HOME_DEG)
    for name, value in joints.items():
        if name in pose:
            pose[name] = value
    return pose
