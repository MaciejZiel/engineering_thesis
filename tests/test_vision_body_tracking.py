import math
import unittest
from dataclasses import dataclass

from vision_robot_arm.vision.body_tracking import (
    BodyFrameStabilizer,
    build_body_frame,
    transform_hands_to_body,
    transform_pose_to_body,
)


@dataclass
class P:
    x: float
    y: float
    z: float
    visibility: float = 1.0


INDICES = {
    "LEFT_SHOULDER": 0,
    "RIGHT_SHOULDER": 1,
    "LEFT_HIP": 2,
    "RIGHT_HIP": 3,
    "LEFT_ELBOW": 4,
    "LEFT_WRIST": 5,
}


def front_facing_pose() -> tuple[list[P], list[P]]:
    # Anatomical right points towards negative camera X for a person facing camera.
    world = [
        P(0.2, 0.0, 0.0),
        P(-0.2, 0.0, 0.0),
        P(0.15, 0.5, 0.0),
        P(-0.15, 0.5, 0.0),
        P(0.35, 0.1, -0.25),
        P(0.40, 0.2, -0.50),
    ]
    image = [
        P(0.6, 0.3, 0.0),
        P(0.4, 0.3, 0.0),
        P(0.57, 0.65, 0.0),
        P(0.43, 0.65, 0.0),
        P(0.7, 0.4, -0.1),
        P(0.75, 0.5, -0.2),
    ]
    return image, world


class BodyFrameTests(unittest.TestCase):
    def test_axes_are_orthonormal_and_right_handed(self) -> None:
        image, world = front_facing_pose()
        frame = build_body_frame(image, world, INDICES, 0.55)
        self.assertIsNotNone(frame)
        assert frame is not None
        for axis in (frame.right, frame.forward, frame.up):
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in axis)), 1)
        self.assertAlmostEqual(sum(a*b for a, b in zip(frame.right, frame.forward)), 0)
        self.assertEqual(frame.source, "shoulders_hips")

    def test_camera_depth_becomes_forward_axis_without_leaking_into_x_or_z(self) -> None:
        image, world = front_facing_pose()
        frame = build_body_frame(image, world, INDICES, 0.55)
        assert frame is not None
        body = transform_pose_to_body(frame, image, world)
        shoulder, wrist = body[0], body[5]
        self.assertAlmostEqual(shoulder.y, 0.0)
        self.assertGreater(wrist.y, 0.45)
        self.assertAlmostEqual(wrist.z, -0.2)

    def test_translation_of_the_person_does_not_change_body_coordinates(self) -> None:
        image, original = front_facing_pose()
        shifted = [P(p.x + 3, p.y - 2, p.z + 4) for p in original]
        first_frame = build_body_frame(image, original, INDICES, 0.55)
        second_frame = build_body_frame(image, shifted, INDICES, 0.55)
        assert first_frame is not None and second_frame is not None
        first = transform_pose_to_body(first_frame, image, original)
        second = transform_pose_to_body(second_frame, image, shifted)
        for a, b in zip(first, second):
            self.assertAlmostEqual(a.x, b.x)
            self.assertAlmostEqual(a.y, b.y)
            self.assertAlmostEqual(a.z, b.z)

    def test_missing_hips_uses_explicit_camera_up_fallback(self) -> None:
        image, world = front_facing_pose()
        image[2].visibility = image[3].visibility = 0.0
        frame = build_body_frame(image, world, INDICES, 0.55)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.source, "shoulders_camera_up")

    def test_unreliable_shoulders_reject_the_entire_frame(self) -> None:
        image, world = front_facing_pose()
        image[0].visibility = 0.1
        self.assertIsNone(build_body_frame(image, world, INDICES, 0.55))

    def test_hand_depth_uses_the_same_body_frame_as_the_arm(self) -> None:
        image, world = front_facing_pose()
        frame = build_body_frame(image, world, INDICES, 0.55)
        assert frame is not None
        hand = transform_hands_to_body(frame, {"left": [P(0.4, 0.2, -0.6)]})
        self.assertGreater(hand["left"][0].y, 0.5)

    def test_non_finite_world_point_is_marked_unavailable(self) -> None:
        image, world = front_facing_pose()
        frame = build_body_frame(image, world, INDICES, 0.55)
        assert frame is not None
        world = list(world)
        world[5] = P(float("nan"), 0, 0)
        transformed = transform_pose_to_body(frame, image, world)
        self.assertTrue(math.isnan(transformed[5].x))
        self.assertEqual(transformed[5].visibility, 0.0)


class BodyFrameStabilizerTests(unittest.TestCase):
    def frame(self, forward=(0.0, 0.0, -1.0)):
        from vision_robot_arm.core.pose_state import BodyFrame3D

        return BodyFrame3D(
            (0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            forward,
            (0.0, -1.0, 0.0),
            "shoulders_hips",
        )

    def test_short_dropout_holds_frame_and_marks_source(self) -> None:
        stabilizer = BodyFrameStabilizer(0.35, max_hold_ms=250)
        stabilizer.update(self.frame(), 1000)

        held = stabilizer.update(None, 1200)

        self.assertIsNotNone(held)
        self.assertEqual(held.source, "shoulders_hips_held")

    def test_expired_dropout_removes_frame(self) -> None:
        stabilizer = BodyFrameStabilizer(0.35, max_hold_ms=250)
        stabilizer.update(self.frame(), 1000)

        self.assertIsNone(stabilizer.update(None, 1251))

    def test_axis_change_is_smoothed_without_losing_unit_length(self) -> None:
        stabilizer = BodyFrameStabilizer(0.35)
        stabilizer.update(self.frame(), 1000)
        changed = self.frame(forward=(0.0, 0.2, -0.98))

        result = stabilizer.update(changed, 1033)

        self.assertIsNotNone(result)
        self.assertLess(abs(result.forward[1]), 0.2)
        self.assertAlmostEqual(math.dist((0, 0, 0), result.forward), 1.0)


if __name__ == "__main__":
    unittest.main()
