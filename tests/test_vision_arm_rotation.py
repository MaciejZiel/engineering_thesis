import unittest
from dataclasses import dataclass

from vision_robot_arm.vision.arm_rotation import (
    ArmRotationTracker,
    forearm_roll,
    shoulder_azimuth,
    wrist_deviation,
)


@dataclass
class P:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


# Shoulders at +/-0.2 m, hips half a metre below: the body frame built from
# these IS the world frame (X right, Y forward, Z up), so test geometry can be
# written directly in body coordinates.
INDICES = {
    "LEFT_SHOULDER": 0, "RIGHT_SHOULDER": 1,
    "LEFT_ELBOW": 2, "RIGHT_ELBOW": 3,
    "LEFT_WRIST": 4, "RIGHT_WRIST": 5,
    "LEFT_HIP": 6, "RIGHT_HIP": 7,
}


def body(left_elbow, right_elbow, left_wrist=None, right_wrist=None) -> list[P]:
    return [
        P(-0.2, 0.0, 0.0), P(0.2, 0.0, 0.0),
        left_elbow, right_elbow,
        left_wrist or P(-0.75, 0.0, 0.0), right_wrist or P(0.75, 0.0, 0.0),
        P(-0.1, 0.0, -0.5), P(0.1, 0.0, -0.5),
    ]


def hand(wrist: P, index_mcp: P, middle_mcp: P, pinky_mcp: P) -> list[P]:
    """21 MediaPipe hand points; the ones we do not use sit on the wrist."""
    points = [P(wrist.x, wrist.y, wrist.z) for _ in range(21)]
    points[5], points[9], points[17] = index_mcp, middle_mcp, pinky_mcp
    return points


def image_of(world: list[P]) -> list[P]:
    """Normalized image points carrying the world points' visibility.

    The body frame takes its geometry from the world points and only the
    visibility from the image ones, so the image coordinates can be anything
    inside the frame.
    """
    return [P(0.5, 0.5, 0.0, point.visibility) for point in world]


def track(tracker: ArmRotationTracker, pose: list[P], hands=None) -> dict[str, float]:
    return tracker.measure(image_of(pose), pose, hands or {}, INDICES, 0.5)


def measure(pose, hands=None):
    return track(ArmRotationTracker(), pose, hands)


class AzimuthTests(unittest.TestCase):
    def test_t_pose_is_zero_on_both_sides(self) -> None:
        angles = measure(body(P(-0.5, 0.0, 0.0), P(0.5, 0.0, 0.0)))

        self.assertAlmostEqual(angles["right_shoulder_azimuth"], 0.0, places=6)
        self.assertAlmostEqual(angles["left_shoulder_azimuth"], 0.0, places=6)

    def test_arms_swung_forward_are_plus_ninety_on_both_sides(self) -> None:
        angles = measure(body(P(-0.2, 0.3, 0.0), P(0.2, 0.3, 0.0)))

        self.assertAlmostEqual(angles["right_shoulder_azimuth"], 90.0, places=6)
        self.assertAlmostEqual(angles["left_shoulder_azimuth"], 90.0, places=6)

    def test_arm_swung_back_is_negative(self) -> None:
        angles = measure(body(P(-0.5, 0.0, 0.0), P(0.4, -0.2, 0.0)))

        self.assertLess(angles["right_shoulder_azimuth"], 0.0)

    def test_a_hanging_arm_has_no_azimuth(self) -> None:
        angles = measure(body(P(-0.5, 0.0, 0.0), P(0.2, 0.02, -0.3)))

        self.assertNotIn("right_shoulder_azimuth", angles)
        self.assertIn("left_shoulder_azimuth", angles)

    def test_invisible_elbow_yields_nothing(self) -> None:
        angles = measure(body(P(-0.5, 0.0, 0.0), P(0.5, 0.0, 0.0, visibility=0.1)))

        self.assertNotIn("right_shoulder_azimuth", angles)

    def test_pure_function_agrees_with_tracker(self) -> None:
        self.assertAlmostEqual(
            shoulder_azimuth(P(0.2, 0.0, 0.0), P(0.2, 0.3, 0.0), "right"), 90.0, places=6
        )
        self.assertAlmostEqual(
            shoulder_azimuth(P(-0.2, 0.0, 0.0), P(-0.2, 0.3, 0.0), "left"), 90.0, places=6
        )


class RollTests(unittest.TestCase):
    """Right forearm along +X, palm down: index MCP toward +Y, pinky toward -Y."""

    def palm_down(self, side: str) -> list[P]:
        s = 1.0 if side == "right" else -1.0
        return hand(P(s * 0.75, 0, 0), P(s * 0.83, 0.03, 0), P(s * 0.85, 0, 0), P(s * 0.83, -0.03, 0))

    def pronated(self, side: str) -> list[P]:
        # Hand rotated a quarter turn about the forearm: the palm now faces +Y.
        s = 1.0 if side == "right" else -1.0
        return hand(P(s * 0.75, 0, 0), P(s * 0.83, 0, 0.03), P(s * 0.85, 0, 0), P(s * 0.83, 0, -0.03))

    def test_palm_down_is_zero_for_both_hands(self) -> None:
        angles = measure(
            body(P(-0.5, 0, 0), P(0.5, 0, 0)),
            {"right": self.palm_down("right"), "left": self.palm_down("left")},
        )

        self.assertAlmostEqual(angles["right_forearm_roll"], 0.0, places=5)
        self.assertAlmostEqual(angles["left_forearm_roll"], 0.0, places=5)

    def test_the_same_pronation_reads_the_same_on_both_hands(self) -> None:
        angles = measure(
            body(P(-0.5, 0, 0), P(0.5, 0, 0)),
            {"right": self.pronated("right"), "left": self.pronated("left")},
        )

        self.assertAlmostEqual(angles["right_forearm_roll"], 90.0, places=5)
        self.assertAlmostEqual(angles["left_forearm_roll"], 90.0, places=5)

    def test_a_missing_hand_leaves_roll_and_deviation_unmeasured(self) -> None:
        angles = measure(body(P(-0.5, 0, 0), P(0.5, 0, 0)), {"right": self.palm_down("right")})

        self.assertIn("right_forearm_roll", angles)
        self.assertNotIn("left_forearm_roll", angles)
        self.assertNotIn("left_wrist_deviation", angles)

    def test_degenerate_palm_yields_nothing(self) -> None:
        flat = hand(P(0.75, 0, 0), P(0.83, 0, 0), P(0.85, 0, 0), P(0.83, 0, 0))
        angles = measure(body(P(-0.5, 0, 0), P(0.5, 0, 0)), {"right": flat})

        self.assertNotIn("right_forearm_roll", angles)

    def test_pure_function_reference_is_palm_down(self) -> None:
        self.assertAlmostEqual(forearm_roll((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), "right"), 0.0)
        self.assertAlmostEqual(forearm_roll((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "right"), 90.0)


class DeviationTests(unittest.TestCase):
    def straight(self, side: str) -> list[P]:
        s = 1.0 if side == "right" else -1.0
        return hand(P(s * 0.75, 0, 0), P(s * 0.83, 0.03, 0), P(s * 0.85, 0, 0), P(s * 0.83, -0.03, 0))

    def toward_index(self, side: str) -> list[P]:
        s = 1.0 if side == "right" else -1.0
        return hand(P(s * 0.75, 0, 0), P(s * 0.83, 0.05, 0), P(s * 0.85, 0.03, 0), P(s * 0.83, -0.01, 0))

    def test_a_straight_hand_has_no_deviation(self) -> None:
        angles = measure(
            body(P(-0.5, 0, 0), P(0.5, 0, 0)),
            {"right": self.straight("right"), "left": self.straight("left")},
        )

        self.assertAlmostEqual(angles["right_wrist_deviation"], 0.0, places=5)
        self.assertAlmostEqual(angles["left_wrist_deviation"], 0.0, places=5)

    def test_bending_toward_the_index_finger_is_positive_on_both_hands(self) -> None:
        angles = measure(
            body(P(-0.5, 0, 0), P(0.5, 0, 0)),
            {"right": self.toward_index("right"), "left": self.toward_index("left")},
        )

        self.assertGreater(angles["right_wrist_deviation"], 5.0)
        self.assertGreater(angles["left_wrist_deviation"], 5.0)
        self.assertAlmostEqual(
            angles["right_wrist_deviation"], angles["left_wrist_deviation"], places=5
        )

    def test_pure_function_needs_a_hand_direction(self) -> None:
        flat = hand(P(0.75, 0, 0), P(0.75, 0, 0), P(0.75, 0, 0), P(0.75, 0, 0))
        self.assertIsNone(wrist_deviation((1.0, 0.0, 0.0), flat, (0.0, 0.0, -1.0)))


class ContinuityTests(unittest.TestCase):
    def test_angles_do_not_jump_across_plus_minus_180(self) -> None:
        tracker = ArmRotationTracker()
        # Upper arm pointing straight back (-Y) from the right shoulder: -90.
        first = track(tracker, body(P(-0.5, 0, 0), P(0.2, -0.3, 0)))
        self.assertAlmostEqual(first["right_shoulder_azimuth"], -90.0, delta=1e-6)
        # Swing further, behind the body: the raw atan2 would flip to +170,
        # the tracker continues to -190.
        second = track(tracker, body(P(-0.5, 0, 0), P(0.2 - 0.3 * 0.9848, -0.3 * 0.1736, 0)))
        self.assertLess(second["right_shoulder_azimuth"], -170.0)
        self.assertGreater(second["right_shoulder_azimuth"], -200.0)

    def test_reset_forgets_the_previous_turn(self) -> None:
        tracker = ArmRotationTracker()
        track(tracker, body(P(-0.5, 0, 0), P(0.2, -0.3, 0)))
        tracker.reset()
        angles = track(tracker, body(P(-0.5, 0, 0), P(0.2, 0.3, 0)))

        self.assertAlmostEqual(angles["right_shoulder_azimuth"], 90.0, places=6)

    def test_no_world_landmarks_means_no_rotation_angles(self) -> None:
        pose = body(P(-0.5, 0, 0), P(0.5, 0, 0))
        self.assertEqual(ArmRotationTracker().measure(image_of(pose), None, {}, INDICES, 0.5), {})


if __name__ == "__main__":
    unittest.main()
