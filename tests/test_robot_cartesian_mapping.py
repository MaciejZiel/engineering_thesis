import unittest

from vision_robot_arm.core.pose_state import LandmarkPoint, PoseState
from vision_robot_arm.robot.cartesian_mapping import tracked_tcp_target


def state(shoulder=(0.2, 0.0, 0.0), wrist=(0.5, 0.0, 0.0)) -> PoseState:
    points = {
        "right_shoulder": LandmarkPoint(*shoulder),
        "right_wrist": LandmarkPoint(*wrist),
    }
    return PoseState(1, [], None, {}, {}, {}, (), False, body_points=points)


class CartesianTrackingMappingTests(unittest.TestCase):
    def test_right_and_up_keep_their_direction(self) -> None:
        origin = tracked_tcp_target(state(), "right")
        right = tracked_tcp_target(state(wrist=(0.6, 0.0, 0.0)), "right")
        up = tracked_tcp_target(state(wrist=(0.5, 0.0, 0.2)), "right")

        assert origin is not None and right is not None and up is not None
        self.assertGreater(right[0], origin[0])
        self.assertGreater(up[2], origin[2])
        self.assertEqual(right[1], origin[1])

    def test_only_depth_is_mirrored(self) -> None:
        origin = tracked_tcp_target(state(), "right")
        toward_camera = tracked_tcp_target(state(wrist=(0.5, 0.2, 0.0)), "right")

        assert origin is not None and toward_camera is not None
        self.assertEqual(toward_camera[0], origin[0])
        self.assertEqual(toward_camera[2], origin[2])
        self.assertLess(toward_camera[1], origin[1])

    def test_unreliable_points_do_not_create_a_target(self) -> None:
        pose = state()
        pose.body_points["right_wrist"] = LandmarkPoint(0.5, 0.0, 0.0, 0.2)

        self.assertIsNone(tracked_tcp_target(pose, "right"))

    def test_target_is_clamped_to_preview_workspace(self) -> None:
        target = tracked_tcp_target(state(wrist=(10.0, 10.0, 10.0)), "right")

        assert target is not None
        self.assertEqual(target, (0.8, -0.72, 1.08))


if __name__ == "__main__":
    unittest.main()
