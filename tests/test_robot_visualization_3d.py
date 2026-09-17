import math
import unittest

import cv2
import numpy as np

from vision_robot_arm.core.pose_state import LandmarkPoint, PoseState
from vision_robot_arm.robot.targets import UR_HOME_DEG
from vision_robot_arm.robot.kinematics import ur7e_joint_points
from vision_robot_arm.robot.visualization_3d import (
    BACKGROUND,
    BASE_SEPARATION_M,
    draw_workspace_3d,
)


class Ur7eKinematicsTests(unittest.TestCase):
    def test_standard_dh_chain_has_six_joints_and_metric_link_lengths(self) -> None:
        points = ur7e_joint_points(UR_HOME_DEG, (0, 0, 0))
        self.assertEqual(len(points), 7)
        self.assertAlmostEqual(math.dist(points[1], points[2]), 0.425, places=4)
        self.assertAlmostEqual(math.dist(points[2], points[3]), 0.3922, places=4)

    def test_two_robot_bases_are_half_a_metre_apart(self) -> None:
        left = (-BASE_SEPARATION_M / 2, 0, 0)
        right = (BASE_SEPARATION_M / 2, 0, 0)
        self.assertAlmostEqual(math.dist(left, right), 0.5)


class WorkspaceRendererTests(unittest.TestCase):
    def test_offline_workspace_still_draws_both_home_robots_and_xyz_grid(self) -> None:
        canvas = np.zeros((320, 480, 3), dtype=np.uint8)
        draw_workspace_3d(cv2, np, canvas, None, None, {})
        background = np.asarray(BACKGROUND, dtype=np.uint8)
        self.assertGreater(np.count_nonzero(np.any(canvas != background, axis=2)), 500)

    def test_tracked_depth_changes_the_rendered_arm_geometry(self) -> None:
        indices = {"LEFT_SHOULDER": 0, "LEFT_ELBOW": 1, "LEFT_WRIST": 2}

        def state(elbow_depth: float) -> PoseState:
            points = [
                LandmarkPoint(-0.2, -0.4, 0),
                LandmarkPoint(-0.2, -0.2, elbow_depth),
                LandmarkPoint(-0.2, 0, elbow_depth),
            ]
            return PoseState(1, points, points, {}, {}, {}, (), False)

        flat = np.zeros((320, 480, 3), dtype=np.uint8)
        deep = np.zeros_like(flat)
        draw_workspace_3d(cv2, np, flat, None, state(0), indices)
        draw_workspace_3d(cv2, np, deep, None, state(0.35), indices)
        self.assertFalse(np.array_equal(flat, deep))

    def test_non_finite_tracking_point_is_skipped_without_crashing(self) -> None:
        points = [LandmarkPoint(float("nan"), 0, 0)] * 3
        state = PoseState(1, points, points, {}, {}, {}, (), False)
        canvas = np.zeros((200, 300, 3), dtype=np.uint8)
        draw_workspace_3d(
            cv2,
            np,
            canvas,
            None,
            state,
            {"LEFT_SHOULDER": 0, "LEFT_ELBOW": 1, "LEFT_WRIST": 2},
        )

    def test_body_forward_axis_changes_rendered_arm_geometry(self) -> None:
        indices = {"LEFT_SHOULDER": 0, "LEFT_ELBOW": 1, "LEFT_WRIST": 2}

        def state(forward: float) -> PoseState:
            camera = [LandmarkPoint(0, 0, 0)] * 3
            body = [
                LandmarkPoint(-0.2, 0, 0),
                LandmarkPoint(-0.2, forward, -0.2),
                LandmarkPoint(-0.2, forward, -0.4),
            ]
            return PoseState(
                1, camera, camera, {}, {}, {}, (), False, body_landmarks=body
            )

        flat = np.zeros((320, 480, 3), dtype=np.uint8)
        forward = np.zeros_like(flat)
        draw_workspace_3d(cv2, np, flat, None, state(0), indices)
        draw_workspace_3d(cv2, np, forward, None, state(0.45), indices)
        self.assertFalse(np.array_equal(flat, forward))
