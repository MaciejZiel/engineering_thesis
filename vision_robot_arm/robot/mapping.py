import math

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.targets import (
    ARM_NAMES,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    MAPPED_JOINTS,
    ArmTargets,
    JointTargets,
)

LIFT_MODE_GESTURE = "right_hand_up"
FIST_GESTURE = "{arm}_fist"
OPEN_HAND_GESTURE = "{arm}_hand_open"


class RobotMapper:
    def __init__(self, config: RobotConfig) -> None:
        self._config = config
        self._last_joints: dict[str, float] = {}

    def map(self, state: PoseState) -> JointTargets:
        arms = {arm: self._map_arm(arm, state) for arm in ARM_NAMES}
        return JointTargets(
            timestamp_ms=state.timestamp_ms,
            arms=arms,
            lift_mode=LIFT_MODE_GESTURE in state.gestures,
        )

    def reset(self) -> None:
        self._last_joints.clear()

    def _map_arm(self, arm: str, state: PoseState) -> ArmTargets:
        joints: dict[str, float] = {}
        for joint in MAPPED_JOINTS:
            mapping = self._config.mapping_for(joint)
            if mapping is None:
                continue
            body_angle = state.angles.get(f"{arm}_{mapping.source}")
            if body_angle is None or not math.isfinite(body_angle):
                continue
            joints[joint] = self._apply_deadband(f"{arm}_{joint}", mapping.to_robot(body_angle))
        return ArmTargets(joints=joints, gripper=_gripper_from_gestures(arm, state.gestures))

    def _apply_deadband(self, key: str, value: float) -> float:
        previous = self._last_joints.get(key)
        if previous is not None and abs(value - previous) < self._config.joint_deadband_deg:
            return previous
        self._last_joints[key] = value
        return value


def _gripper_from_gestures(arm: str, gestures: tuple[str, ...]) -> str | None:
    if FIST_GESTURE.format(arm=arm) in gestures:
        return GRIPPER_CLOSE
    if OPEN_HAND_GESTURE.format(arm=arm) in gestures:
        return GRIPPER_OPEN
    return None
