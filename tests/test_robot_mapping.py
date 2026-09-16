import unittest

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.controller import map_pose_to_robot_commands


class RobotMappingTests(unittest.TestCase):
    def test_maps_pose_state_to_debug_robot_commands(self) -> None:
        state = PoseState(
            timestamp_ms=1,
            landmarks=[],
            world_landmarks=None,
            raw_angles={},
            angles={"right_shoulder": 45.0, "right_elbow": 210.0},
            relative_angles={},
            gestures=("right_elbow_bent",),
            calibrated=False,
        )

        commands = map_pose_to_robot_commands(state)
        formatted = [command.format() for command in commands]

        self.assertIn("shoulder= 45.0 (right_shoulder)", formatted)
        self.assertIn("elbow=180.0 (right_elbow)", formatted)
        self.assertIn("gripper=close (right_elbow_bent)", formatted)


if __name__ == "__main__":
    unittest.main()
