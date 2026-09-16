from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.targets import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    JOINT_ELBOW,
    JOINT_SHOULDER,
    JointTargets,
)

JOINT_SOURCES = {
    JOINT_SHOULDER: "right_shoulder",
    JOINT_ELBOW: "right_elbow",
}

LIFT_MODE_GESTURE = "right_hand_up"
GRIPPER_CLOSE_GESTURE = "right_elbow_bent"
GRIPPER_OPEN_GESTURE = "right_arm_side"


class RobotMapper:
    def __init__(self, config: RobotConfig) -> None:
        self._config = config
        self._last_joints: dict[str, float] = {}

    def map(self, state: PoseState) -> JointTargets:
        joints: dict[str, float] = {}
        for joint, source in JOINT_SOURCES.items():
            angle = state.angles.get(source)
            if angle is None:
                continue
            joints[joint] = self._apply_deadband(
                joint,
                self._config.limit_for(joint).clamp(angle),
            )

        return JointTargets(
            timestamp_ms=state.timestamp_ms,
            joints=joints,
            gripper=_gripper_from_gestures(state.gestures),
            lift_mode=LIFT_MODE_GESTURE in state.gestures,
        )

    def reset(self) -> None:
        self._last_joints.clear()

    def _apply_deadband(self, joint: str, value: float) -> float:
        previous = self._last_joints.get(joint)
        if previous is not None and abs(value - previous) < self._config.joint_deadband_deg:
            return previous
        self._last_joints[joint] = value
        return value


def _gripper_from_gestures(gestures: tuple[str, ...]) -> str | None:
    if GRIPPER_CLOSE_GESTURE in gestures:
        return GRIPPER_CLOSE
    if GRIPPER_OPEN_GESTURE in gestures:
        return GRIPPER_OPEN
    return None
