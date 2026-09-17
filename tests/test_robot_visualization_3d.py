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
    CONTENT_BOX,
    DROP_LINE,
    LABEL,
    LABEL_MIN_WIDTH,
    BASE,
    LEFT_ARM,
    RIGHT_ARM,
    Camera3D,
    _projector,
    draw_workspace_3d,
)
from vision_robot_arm.robot.targets import ArmState, RobotState, full_joint_pose


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


def render(
    width: int, height: int, robot_state=None, pose_state=None, indices=None
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    draw_workspace_3d(cv2, np, canvas, robot_state, pose_state, indices or {})
    return canvas


def matches(canvas: np.ndarray, color, tolerance: int = 10) -> np.ndarray:
    """Anti-aliased strokes rarely reach their exact colour, so match a neighbourhood."""
    distance = np.abs(canvas.astype(int) - np.asarray(color, dtype=int)).max(axis=2)
    return distance <= tolerance


def arm_state(shoulder: float, elbow: float) -> RobotState:
    joints = full_joint_pose({"shoulder": shoulder, "elbow": elbow})
    return RobotState(
        {
            "left": ArmState(joints, joints, "open"),
            "right": ArmState(joints, joints, "open"),
        },
        False,
    )


def tracked_pose(elbow_height: float = -0.22) -> PoseState:
    points = [LandmarkPoint(0.0, 0.0, 0.0) for _ in range(17)]
    points[11] = LandmarkPoint(-0.19, 0.0, 0.0)
    points[12] = LandmarkPoint(0.19, 0.0, 0.0)
    points[13] = LandmarkPoint(-0.30, 0.10, elbow_height)
    points[14] = LandmarkPoint(0.31, 0.08, elbow_height)
    points[15] = LandmarkPoint(-0.22, 0.34, -0.14)
    points[16] = LandmarkPoint(0.26, 0.31, -0.10)
    return PoseState(1, points, points, {}, {}, {}, (), False, body_landmarks=points)


BODY_INDICES = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
}


class ReadabilityTests(unittest.TestCase):
    """The preview used to render a thumb-sized scene in the corner of a wide panel."""

    SHAPES = ((432, 267), (1310, 597), (280, 280), (900, 300))

    def test_the_view_fills_the_panel_whatever_shape_it_is_given(self) -> None:
        for width, height in self.SHAPES:
            with self.subTest(shape=(width, height)):
                project = _projector(np, width, height, Camera3D())
                corners = [
                    project((x, y, z))
                    for x in CONTENT_BOX[0]
                    for y in CONTENT_BOX[1]
                    for z in CONTENT_BOX[2]
                ]
                xs = [corner[0] for corner in corners]
                ys = [corner[1] for corner in corners]
                self.assertGreaterEqual(min(xs), -1)
                self.assertLessEqual(max(xs), width + 1)
                self.assertGreaterEqual(min(ys), -1)
                self.assertLessEqual(max(ys), height + 1)
                self.assertTrue(
                    max(xs) - min(xs) > width * 0.85
                    or max(ys) - min(ys) > height * 0.85,
                    "the framed volume touches neither panel edge",
                )

    def test_the_view_holds_still_whatever_the_arms_and_the_body_do(self) -> None:
        """A camera fitted to the live content breathes on every frame. This one is fixed."""
        scenes = (
            (None, None),
            (arm_state(-70, 55), tracked_pose(-0.55)),
            (arm_state(-70, 55), tracked_pose(0.35)),
            (arm_state(-120, -40), tracked_pose(-0.10)),
            (arm_state(-20, 140), tracked_pose(0.20)),
        )
        centres = []
        for robot_state, pose in scenes:
            mask = matches(render(432, 267, robot_state, pose, BODY_INDICES), BASE)
            rows, columns = np.nonzero(mask)
            self.assertGreater(len(rows), 300, "the bases were not drawn")
            centres.append((columns.mean(), rows.mean()))

        for axis in (0, 1):
            spread = max(c[axis] for c in centres) - min(c[axis] for c in centres)
            self.assertLess(spread, 2.5, "the bases drifted between frames")

    def test_the_small_preview_leaves_the_words_to_the_dashboard(self) -> None:
        pose = tracked_pose()
        narrow = render(432, 267, arm_state(-60, 40), pose, BODY_INDICES)
        wide = render(LABEL_MIN_WIDTH, 420, arm_state(-60, 40), pose, BODY_INDICES)

        def top_band(canvas: np.ndarray) -> np.ndarray:
            return canvas[: round(canvas.shape[0] * 0.25)]

        self.assertEqual(np.count_nonzero(matches(top_band(narrow), LABEL)), 0)
        self.assertGreater(np.count_nonzero(matches(top_band(wide), LABEL)), 0)

    def test_a_raised_tool_is_tied_to_the_floor_by_a_drop_line(self) -> None:
        """Perspective alone cannot say how high a hand is."""
        canvas = render(432, 267, arm_state(-90, 0))

        self.assertGreater(np.count_nonzero(matches(canvas, DROP_LINE)), 20)

    def test_each_base_ring_carries_the_colour_of_its_own_arm(self) -> None:
        canvas = render(432, 267)
        project = _projector(np, 432, 267, Camera3D())

        for x, color in (
            (-BASE_SEPARATION_M / 2, LEFT_ARM),
            (BASE_SEPARATION_M / 2, RIGHT_ARM),
        ):
            with self.subTest(base=x):
                px, py = project((x, 0.0, 0.0))[:2]
                patch = canvas[py - 16 : py + 16, px - 16 : px + 16]
                self.assertTrue(np.any(matches(patch, color)))
