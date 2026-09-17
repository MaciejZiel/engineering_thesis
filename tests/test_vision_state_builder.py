import unittest
from dataclasses import dataclass

from vision_robot_arm.vision.state_builder import PoseStateBuilder

POSE_LANDMARK_NAMES = (
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)
INDICES = {name: index for index, name in enumerate(POSE_LANDMARK_NAMES)}


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


def make_pose() -> list[FakeLandmark]:
    landmarks = [FakeLandmark(0.5, 0.5, visibility=0.0) for _ in POSE_LANDMARK_NAMES]
    landmarks[INDICES["RIGHT_SHOULDER"]] = FakeLandmark(0.6, 0.3)
    landmarks[INDICES["RIGHT_ELBOW"]] = FakeLandmark(0.6, 0.5)
    landmarks[INDICES["RIGHT_WRIST"]] = FakeLandmark(0.6, 0.7)
    landmarks[INDICES["RIGHT_INDEX"]] = FakeLandmark(0.6, 0.8, visibility=0.1)
    return landmarks


class PoseStateBuilderTests(unittest.TestCase):
    def test_world_only_mode_refuses_2d_body_angles(self) -> None:
        builder = PoseStateBuilder(INDICES, min_visibility=0.55, smoothing_alpha=1.0)
        state = builder.build(1, make_pose(), None, world_only=True)
        self.assertIsNone(state.angles["right_elbow"])
        self.assertNotIn("right_elbow", state.angle_sources)

    def test_world_only_state_preserves_3d_hands_and_measurement_source(self) -> None:
        builder = PoseStateBuilder(INDICES, min_visibility=0.55, smoothing_alpha=1.0)
        hand = [FakeLandmark(0.1, 0.2, 0.3)]
        state = builder.build(
            1,
            make_pose(),
            make_pose(),
            extra_angles={"right_wrist": 170.0},
            hand_tracking_enabled=True,
            hand_landmarks={"right": hand},
            hand_world_landmarks={"right": hand},
            world_only=True,
            extra_angle_sources={"right_wrist": "hand_world_3d_held"},
        )
        self.assertEqual(state.hand_world_landmarks["right"][0].z, 0.3)
        self.assertEqual(state.angle_sources["right_wrist"], "hand_world_3d_held")

    def test_extra_angles_replace_unreliable_pose_angles(self) -> None:
        builder = PoseStateBuilder(INDICES, min_visibility=0.55, smoothing_alpha=1.0)

        state = builder.build(1, make_pose(), None, extra_angles={"right_wrist": 150.0})

        self.assertEqual(state.angles["right_wrist"], 150.0)
        self.assertEqual(state.raw_angles["right_wrist"], 150.0)
        self.assertAlmostEqual(state.angles["right_elbow"], 180.0)

    def test_angles_measured_outside_the_pose_model_reach_the_state(self) -> None:
        builder = PoseStateBuilder(INDICES, min_visibility=0.55, smoothing_alpha=1.0)

        state = builder.build(
            1, make_pose(), None, extra_angles={"right_shoulder_elevation": 169.2}
        )

        self.assertAlmostEqual(state.angles["right_shoulder_elevation"], 169.2)

    def test_non_finite_extra_angles_are_ignored(self) -> None:
        builder = PoseStateBuilder(INDICES, min_visibility=0.55, smoothing_alpha=1.0)

        state = builder.build(
            1, make_pose(), None, extra_angles={"right_wrist": float("nan")}
        )

        self.assertIsNone(state.angles["right_wrist"])

    def test_hand_tracking_clears_the_pose_model_wrist_angle(self) -> None:
        builder = PoseStateBuilder(INDICES, min_visibility=0.55, smoothing_alpha=1.0)

        state = builder.build(1, make_pose(), None, hand_tracking_enabled=True)

        self.assertIsNone(state.angles["right_wrist"])

    def test_extra_angles_are_smoothed_like_other_angles(self) -> None:
        builder = PoseStateBuilder(INDICES, min_visibility=0.55, smoothing_alpha=0.5)
        builder.build(1000, make_pose(), None, extra_angles={"right_wrist": 100.0})

        state = builder.build(
            1033, make_pose(), None, extra_angles={"right_wrist": 120.0}
        )

        self.assertAlmostEqual(state.angles["right_wrist"], 110.0, delta=0.5)

    def test_without_extra_angle_unreliable_wrist_stays_missing(self) -> None:
        builder = PoseStateBuilder(INDICES, min_visibility=0.55, smoothing_alpha=1.0)

        state = builder.build(1, make_pose(), None)

        self.assertIsNone(state.angles["right_wrist"])

    def test_extra_gestures_are_appended(self) -> None:
        builder = PoseStateBuilder(INDICES, min_visibility=0.55, smoothing_alpha=1.0)

        state = builder.build(1, make_pose(), None, extra_gestures=("right_fist",))

        self.assertIn("right_fist", state.gestures)


if __name__ == "__main__":
    unittest.main()
