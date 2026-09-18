import math
import unittest

from vision_robot_arm.robot.kinematics import (
    DEFAULT_IK_LIMITS,
    clears_floor,
    lowest_link_point,
    solve_position_ik,
    ur7e_joint_points,
)
from vision_robot_arm.robot.config import RobotConfig
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


class SafetyPlaneTests(unittest.TestCase):
    """A table-mounted arm cannot swing below its own table."""

    def test_a_pose_reaching_under_the_table_is_caught(self) -> None:
        folded = {"base": 0.0, "shoulder": 0.0, "elbow": 90.0}

        self.assertLess(lowest_link_point(folded, (0.0, 0.0, 0.0)), -0.2)
        self.assertFalse(clears_floor(folded, (0.0, 0.0, 0.0)))

    def test_the_base_offset_moves_the_plane_with_the_robot(self) -> None:
        folded = {"base": 0.0, "shoulder": 0.0, "elbow": 90.0}

        self.assertTrue(clears_floor(folded, (0.0, 0.0, 0.4)))

    def test_a_normal_working_pose_clears_the_table(self) -> None:
        self.assertTrue(clears_floor(UR_HOME_DEG, (0.0, 0.0, 0.0)))

    def test_the_solver_never_returns_a_pose_through_the_table(self) -> None:
        base = (0.25, 0.0, 0.0)
        seed = dict(UR_HOME_DEG)
        checked = 0
        for xi in range(7):
            for yi in range(5):
                for zi in range(7):
                    target = (
                        base[0] - 0.55 + 1.10 * xi / 6,
                        -0.72 + 0.90 * yi / 4,
                        0.18 + 0.90 * zi / 6,
                    )
                    solution = solve_position_ik(target, seed, base)
                    if solution is None:
                        continue
                    seed = solution
                    checked += 1
                    self.assertTrue(clears_floor(solution, base))
        self.assertGreater(checked, 100, "the sweep solved too little to mean anything")


class SoftwareLimitTests(unittest.TestCase):
    def test_the_solver_keeps_inside_its_declared_limits(self) -> None:
        base = (0.25, 0.0, 0.0)
        seed = dict(UR_HOME_DEG)
        for step in range(40):
            target = (0.25 + 0.4 * math.cos(step), -0.5, 0.3 + 0.3 * math.sin(step))
            solution = solve_position_ik(target, seed, base)
            if solution is None:
                continue
            seed = solution
            for joint, (minimum, maximum) in DEFAULT_IK_LIMITS.items():
                with self.subTest(joint=joint):
                    self.assertGreaterEqual(solution[joint], minimum - 1e-6)
                    self.assertLessEqual(solution[joint], maximum + 1e-6)

    def test_the_twin_clamps_to_the_same_band_as_the_solver(self) -> None:
        """A simulated pose the controller would never be asked for is a lie."""
        config = RobotConfig(backend="sim")

        for joint, (minimum, maximum) in DEFAULT_IK_LIMITS.items():
            with self.subTest(joint=joint):
                limit = config.limit_for(joint)
                self.assertGreaterEqual(limit.minimum, minimum)
                self.assertLessEqual(limit.maximum, maximum)
