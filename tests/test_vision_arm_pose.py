from dataclasses import dataclass
import unittest

from vision_robot_arm.vision.arm_pose import (
    IMAGE_DOWN,
    arm_elevation_angles,
    torso_down_vector,
    vector_angle,
)


@dataclass
class P:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


INDICES = {
    "LEFT_SHOULDER": 0,
    "RIGHT_SHOULDER": 1,
    "LEFT_ELBOW": 2,
    "RIGHT_ELBOW": 3,
    "LEFT_HIP": 4,
    "RIGHT_HIP": 5,
}


def pose(left_elbow: P, right_elbow: P, hips_visible: bool = False) -> list[P]:
    visibility = 1.0 if hips_visible else 0.1
    return [
        P(0.4, 0.4),
        P(0.6, 0.4),
        left_elbow,
        right_elbow,
        P(0.42, 0.8, visibility=visibility),
        P(0.58, 0.8, visibility=visibility),
    ]


class ElevationTests(unittest.TestCase):
    def test_hanging_arm_is_zero_degrees(self) -> None:
        landmarks = pose(P(0.4, 0.6), P(0.6, 0.6))

        angles = arm_elevation_angles(landmarks, INDICES)

        self.assertAlmostEqual(angles["left_shoulder_elevation"], 0.0)
        self.assertAlmostEqual(angles["right_shoulder_elevation"], 0.0)

    def test_horizontal_arm_is_ninety_degrees(self) -> None:
        landmarks = pose(P(0.2, 0.4), P(0.8, 0.4))

        angles = arm_elevation_angles(landmarks, INDICES, aspect_ratio=1.0)

        self.assertAlmostEqual(angles["left_shoulder_elevation"], 90.0)
        self.assertAlmostEqual(angles["right_shoulder_elevation"], 90.0)

    def test_raised_arm_is_one_hundred_eighty_degrees(self) -> None:
        landmarks = pose(P(0.4, 0.15), P(0.6, 0.15))

        angles = arm_elevation_angles(landmarks, INDICES)

        self.assertAlmostEqual(angles["left_shoulder_elevation"], 180.0)
        self.assertAlmostEqual(angles["right_shoulder_elevation"], 180.0)

    def test_works_without_visible_hips(self) -> None:
        landmarks = pose(P(0.4, 0.15), P(0.6, 0.15), hips_visible=False)

        self.assertIn("left_shoulder_elevation", arm_elevation_angles(landmarks, INDICES))

    def test_unreliable_elbow_is_skipped(self) -> None:
        landmarks = pose(P(0.4, 0.15, visibility=0.1), P(0.6, 0.15))

        angles = arm_elevation_angles(landmarks, INDICES)

        self.assertNotIn("left_shoulder_elevation", angles)
        self.assertIn("right_shoulder_elevation", angles)

    def test_elbow_on_the_shoulder_gives_no_angle(self) -> None:
        landmarks = pose(P(0.4, 0.4), P(0.6, 0.4))

        self.assertEqual(arm_elevation_angles(landmarks, INDICES), {})

    def test_arm_pointing_at_the_camera_is_left_unmeasured(self) -> None:
        landmarks = pose(P(0.41, 0.42), P(0.6, 0.6))

        angles = arm_elevation_angles(landmarks, INDICES)

        self.assertNotIn("left_shoulder_elevation", angles)
        self.assertIn("right_shoulder_elevation", angles)

    def test_a_distant_person_keeps_a_smaller_noise_floor(self) -> None:
        far = [P(0.48, 0.4), P(0.52, 0.4), P(0.48, 0.45), P(0.52, 0.45), P(0.5, 0.5), P(0.5, 0.5)]

        angles = arm_elevation_angles(far, INDICES)

        self.assertAlmostEqual(angles["left_shoulder_elevation"], 0.0)

    def test_wide_frames_stretch_the_arm_toward_horizontal(self) -> None:
        landmarks = pose(P(0.3, 0.3), P(0.6, 0.4))

        square = arm_elevation_angles(landmarks, INDICES, aspect_ratio=1.0)["left_shoulder_elevation"]
        wide = arm_elevation_angles(landmarks, INDICES, aspect_ratio=2.0)["left_shoulder_elevation"]

        self.assertAlmostEqual(square, 135.0)
        self.assertLess(wide, square)
        self.assertGreater(wide, 90.0)


class TorsoDownTests(unittest.TestCase):
    def test_uses_image_down_without_hips(self) -> None:
        self.assertEqual(torso_down_vector(pose(P(0.4, 0.6), P(0.6, 0.6)), INDICES), IMAGE_DOWN)

    def test_leaning_torso_tilts_the_reference(self) -> None:
        landmarks = pose(P(0.4, 0.6), P(0.6, 0.6), hips_visible=True)
        landmarks[4] = P(0.62, 0.8)
        landmarks[5] = P(0.78, 0.8)

        down = torso_down_vector(landmarks, INDICES)

        self.assertGreater(down[0], 0.0)
        self.assertGreater(down[1], 0.0)

    def test_leaning_body_keeps_arm_elevation_relative_to_torso(self) -> None:
        landmarks = pose(P(0.4, 0.6), P(0.6, 0.6), hips_visible=True)
        landmarks[4] = P(0.42, 0.8)
        landmarks[5] = P(0.58, 0.8)

        angles = arm_elevation_angles(landmarks, INDICES)

        self.assertAlmostEqual(angles["left_shoulder_elevation"], 0.0, delta=1.0)


class VectorAngleTests(unittest.TestCase):
    def test_returns_none_for_degenerate_vectors(self) -> None:
        self.assertIsNone(vector_angle((0.0, 0.0), (0.0, 1.0)))
        self.assertIsNone(vector_angle((float("nan"), 0.0), (0.0, 1.0)))

    def test_opposite_vectors_are_one_hundred_eighty_degrees(self) -> None:
        self.assertAlmostEqual(vector_angle((0.0, -1.0), (0.0, 1.0)), 180.0)


if __name__ == "__main__":
    unittest.main()
