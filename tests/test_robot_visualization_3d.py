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
    HUMAN_ARM,
    HUMAN_HEAD,
    HUMAN_TORSO,
    LEFT_ARM,
    RIGHT_ARM,
    Camera3D,
    _arm_outline,
    _projector,
    draw_body_3d,
    draw_workspace_3d,
    draw_workspace_views,
    FRONT_CAMERA,
    TOP_CAMERA,
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
        draw_body_3d(cv2, np, flat, state(0), indices)
        draw_body_3d(cv2, np, deep, state(0.35), indices)
        self.assertFalse(np.array_equal(flat, deep))

    def test_non_finite_tracking_point_is_skipped_without_crashing(self) -> None:
        points = [LandmarkPoint(float("nan"), 0, 0)] * 3
        state = PoseState(1, points, points, {}, {}, {}, (), False)
        canvas = np.zeros((200, 300, 3), dtype=np.uint8)
        indices = {"LEFT_SHOULDER": 0, "LEFT_ELBOW": 1, "LEFT_WRIST": 2}
        draw_workspace_3d(cv2, np, canvas, None, state, indices)
        draw_body_3d(cv2, np, canvas, state, indices)

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
        draw_body_3d(cv2, np, flat, state(0), indices)
        draw_body_3d(cv2, np, forward, state(0.45), indices)
        self.assertFalse(np.array_equal(flat, forward))


def render(
    width: int,
    height: int,
    robot_state=None,
    pose_state=None,
    indices=None,
    mirrored: bool = False,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    draw_workspace_3d(
        cv2, np, canvas, robot_state, pose_state, indices or {}, mirrored=mirrored
    )
    return canvas


def render_body(
    width: int, height: int, pose_state=None, indices=None, mirrored: bool = False
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    draw_body_3d(cv2, np, canvas, pose_state, indices or {}, mirrored=mirrored)
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
    points = [LandmarkPoint(0.0, 0.0, 0.0) for _ in range(33)]
    points[11] = LandmarkPoint(-0.19, 0.0, 0.0)
    points[12] = LandmarkPoint(0.19, 0.0, 0.0)
    points[13] = LandmarkPoint(-0.30, 0.10, elbow_height)
    points[14] = LandmarkPoint(0.31, 0.08, elbow_height)
    points[15] = LandmarkPoint(-0.22, 0.34, -0.14)
    points[16] = LandmarkPoint(0.26, 0.31, -0.10)
    points[0] = LandmarkPoint(0.0, 0.06, 0.26)
    points[23] = LandmarkPoint(-0.13, 0.0, -0.52)
    points[24] = LandmarkPoint(0.13, 0.0, -0.52)
    return PoseState(1, points, points, {}, {}, {}, (), False, body_landmarks=points)


BODY_INDICES = {
    "NOSE": 0,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
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
        narrow = render_body(432, 267, pose, BODY_INDICES)
        wide = render_body(LABEL_MIN_WIDTH, 420, pose, BODY_INDICES)

        def top_band(canvas: np.ndarray) -> np.ndarray:
            return canvas[: round(canvas.shape[0] * 0.25)]

        self.assertEqual(np.count_nonzero(matches(top_band(narrow), LABEL)), 0)
        self.assertGreater(np.count_nonzero(matches(top_band(wide), LABEL)), 0)

    def test_the_tracked_arms_hang_off_a_body_instead_of_floating(self) -> None:
        """Two bare sticks in mid-air do not read as a person."""
        with_body = render_body(432, 267, tracked_pose(), BODY_INDICES)

        invisible = tracked_pose()
        hidden = PoseState(
            1,
            invisible.landmarks,
            invisible.world_landmarks,
            {},
            {},
            {},
            (),
            False,
            body_landmarks=[
                LandmarkPoint(point.x, point.y, point.z, visibility=0.0)
                for point in invisible.body_landmarks
            ],
        )
        without_body = render_body(432, 267, hidden, BODY_INDICES)

        for color in (HUMAN_TORSO, HUMAN_HEAD):
            with self.subTest(part=color):
                drawn = np.count_nonzero(matches(with_body, color))
                # Anti-aliased arm edges stray near these colours, so compare the two.
                self.assertGreater(drawn, 40)
                self.assertGreater(drawn, 4 * np.count_nonzero(matches(without_body, color)))

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


class SplitViewTests(unittest.TestCase):
    """The operator used to be drawn over the robots, which tangled both."""

    def test_the_workspace_view_no_longer_draws_the_operator(self) -> None:
        empty = render(432, 267, arm_state(-70, 55))
        with_person = render(432, 267, arm_state(-70, 55), tracked_pose(), BODY_INDICES)

        self.assertTrue(np.array_equal(empty, with_person))

    def test_the_body_view_draws_the_operator_and_no_robots(self) -> None:
        canvas = render_body(432, 267, tracked_pose(), BODY_INDICES)

        self.assertGreater(np.count_nonzero(matches(canvas, HUMAN_ARM)), 100)
        for color in (LEFT_ARM, RIGHT_ARM):
            with self.subTest(arm=color):
                self.assertEqual(np.count_nonzero(matches(canvas, color)), 0)

    def test_mirroring_swaps_which_side_of_the_screen_each_robot_is_on(self) -> None:
        """The camera image is mirrored, so the hand you see yourself raise has to
        belong to the robot on that same side of the screen."""
        bases = (
            (-BASE_SEPARATION_M / 2, 0.0, 0.0),
            (BASE_SEPARATION_M / 2, 0.0, 0.0),
        )

        plain = _projector(np, 432, 267, Camera3D(), False)
        flipped = _projector(np, 432, 267, Camera3D(), True)

        self.assertLess(plain(bases[0])[0], plain(bases[1])[0])
        self.assertGreater(flipped(bases[0])[0], flipped(bases[1])[0])

    def test_mirroring_changes_what_is_drawn(self) -> None:
        plain = render(432, 267, arm_state(-70, 55))
        flipped = render(432, 267, arm_state(-70, 55), mirrored=True)

        self.assertFalse(np.array_equal(plain, flipped))

    def test_mirroring_the_body_view_swaps_the_arms_too(self) -> None:
        pose = tracked_pose()
        plain = render_body(432, 267, pose, BODY_INDICES)
        flipped = render_body(432, 267, pose, BODY_INDICES, mirrored=True)

        self.assertFalse(np.array_equal(plain, flipped))


class ArmOutlineTests(unittest.TestCase):
    def test_the_tool_is_one_stub_and_not_the_wrist_frame_cluster(self) -> None:
        """Three near-coincident wrist frames drew a hook at preview size."""
        points = ur7e_joint_points(UR_HOME_DEG, (0, 0, 0))
        outline = _arm_outline(points)

        self.assertEqual(len(outline), 5)
        self.assertEqual(outline[0], points[0])
        self.assertEqual(outline[-1], points[-1])
        self.assertNotIn(points[4], outline)
        self.assertNotIn(points[5], outline)

    def test_a_short_chain_is_passed_through_untouched(self) -> None:
        short = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.1))

        self.assertEqual(_arm_outline(short), short)


class OrthogonalViewTests(unittest.TestCase):
    """One perspective could not show reach across the table and tool height at once."""

    def test_the_plan_view_ignores_height_and_the_elevation_reads_it(self) -> None:
        floor, raised = (0.3, -0.2, 0.0), (0.3, -0.2, 0.8)

        top = _projector(np, 300, 200, TOP_CAMERA)
        front = _projector(np, 300, 200, FRONT_CAMERA)

        self.assertLess(abs(top(floor)[1] - top(raised)[1]), 4)
        self.assertGreater(abs(front(floor)[1] - front(raised)[1]), 40)

    def test_the_elevation_ignores_depth_and_the_plan_reads_it(self) -> None:
        near, far = (0.3, -0.7, 0.4), (0.3, 0.1, 0.4)

        top = _projector(np, 300, 200, TOP_CAMERA)
        front = _projector(np, 300, 200, FRONT_CAMERA)

        self.assertGreater(abs(top(near)[1] - top(far)[1]), 40)
        self.assertLess(abs(front(near)[1] - front(far)[1]), 4)

    def test_looking_straight_down_does_not_divide_by_a_zero_axis(self) -> None:
        """World up is useless as a reference when the camera looks along it."""
        project = _projector(np, 300, 200, TOP_CAMERA)

        self.assertIsNotNone(project((0.0, 0.0, 0.5)))

    def test_both_views_are_drawn_into_their_own_half(self) -> None:
        canvas = np.zeros((394, 262, 3), dtype=np.uint8)

        draw_workspace_views(cv2, np, canvas, arm_state(-70, 55), None, {})

        upper, lower = canvas[:190], canvas[204:]
        background = np.asarray(BACKGROUND, dtype=np.uint8)
        for half in (upper, lower):
            with self.subTest(half=half.shape):
                self.assertGreater(
                    np.count_nonzero(np.any(half != background, axis=2)), 400
                )
        self.assertFalse(np.array_equal(upper, lower[: upper.shape[0]]))

    def test_a_canvas_too_small_to_split_still_draws_one_view(self) -> None:
        canvas = np.zeros((12, 60, 3), dtype=np.uint8)

        draw_workspace_views(cv2, np, canvas, arm_state(-70, 55), None, {})

        background = np.asarray(BACKGROUND, dtype=np.uint8)
        self.assertGreater(np.count_nonzero(np.any(canvas != background, axis=2)), 0)
