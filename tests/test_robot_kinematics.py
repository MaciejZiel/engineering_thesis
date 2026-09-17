import math
import unittest

from vision_robot_arm.robot.kinematics import solve_position_ik, ur7e_joint_points
from vision_robot_arm.robot.targets import UR_HOME_DEG


class PositionIkTests(unittest.TestCase):
    def test_reaches_xyz_target_without_changing_wrist_orientation(self) -> None:
        target = (0.23, -0.34, 0.62)

        solution = solve_position_ik(target, UR_HOME_DEG)

        self.assertIsNotNone(solution)
        assert solution is not None
        self.assertLess(math.dist(ur7e_joint_points(solution)[-1], target), 0.008)
        for joint in ("wrist_1", "wrist_2", "wrist_3"):
            self.assertEqual(solution[joint], UR_HOME_DEG[joint])

    def test_previous_solution_keeps_nearby_targets_on_continuous_branch(self) -> None:
        first = solve_position_ik((0.23, -0.34, 0.62), UR_HOME_DEG)
        assert first is not None

        second = solve_position_ik((0.25, -0.34, 0.62), first)

        self.assertIsNotNone(second)
        assert second is not None
        self.assertLess(abs(second["base"] - first["base"]), 10.0)
        self.assertLess(abs(second["elbow"] - first["elbow"]), 10.0)

    def test_rejects_unreachable_and_non_finite_targets(self) -> None:
        self.assertIsNone(solve_position_ik((5.0, 5.0, 5.0), UR_HOME_DEG))
        self.assertIsNone(solve_position_ik((math.nan, 0.0, 0.0), UR_HOME_DEG))


if __name__ == "__main__":
    unittest.main()
