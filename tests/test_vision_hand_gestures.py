from dataclasses import dataclass
import math
import unittest

from vision_robot_arm.vision.hand_gestures import (
    assign_hand_sides,
    classify_hand,
    count_extended_fingers,
    HandGestureFilter,
    detect_hand_gestures,
    hand_wrist_angles,
    signed_wrist_deviation,
    WristAngleHold,
)


@dataclass
class P:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


EXTENDED_FINGER = ((0.0, 0.07), (0.0, 0.10), (0.0, 0.13), (0.0, 0.16))
CURLED_FINGER = ((0.0, 0.07), (0.012, 0.088), (0.02, 0.075), (0.015, 0.06))


def make_hand(wrist: tuple[float, float], extended: tuple[bool, bool, bool, bool]) -> list[P]:
    """Synthetic hand: straight fingers point up, curled fingers fold back toward the palm."""
    hand = [P(*wrist) for _ in range(21)]
    for thumb_index, step in enumerate((1, 2, 3, 4), start=1):
        hand[thumb_index] = P(wrist[0] - 0.012 * step, wrist[1] - 0.012 * step)
    finger_bases = (5, 9, 13, 17)
    for finger, (base_index, is_extended) in enumerate(zip(finger_bases, extended)):
        dx = (finger - 1.5) * 0.02
        shape = EXTENDED_FINGER if is_extended else CURLED_FINGER
        for offset, (x_offset, distance) in enumerate(shape):
            hand[base_index + offset] = P(wrist[0] + dx + x_offset, wrist[1] - distance)
    return hand


POSE_INDICES = {"LEFT_WRIST": 0, "RIGHT_WRIST": 1, "LEFT_ELBOW": 2, "RIGHT_ELBOW": 3}
POSE = [P(0.3, 0.6), P(0.7, 0.6), P(0.3, 0.9), P(0.7, 0.9)]


class FingerCountTests(unittest.TestCase):
    def test_open_hand_has_four_extended_fingers(self) -> None:
        hand = make_hand((0.5, 0.5), (True, True, True, True))

        self.assertEqual(count_extended_fingers(hand), 4)
        self.assertEqual(classify_hand(hand), "open")

    def test_fist_has_no_extended_fingers(self) -> None:
        hand = make_hand((0.5, 0.5), (False, False, False, False))

        self.assertEqual(count_extended_fingers(hand), 0)
        self.assertEqual(classify_hand(hand), "fist")

    def test_partial_hand_is_unclassified(self) -> None:
        hand = make_hand((0.5, 0.5), (True, True, False, False))

        self.assertIsNone(classify_hand(hand))

    def test_incomplete_landmarks_are_unclassified(self) -> None:
        self.assertIsNone(classify_hand([P(0.0, 0.0)] * 5))


class HandSideAssignmentTests(unittest.TestCase):
    def test_hands_are_matched_to_nearest_pose_wrist(self) -> None:
        left_hand = make_hand((0.32, 0.62), (True, True, True, True))
        right_hand = make_hand((0.69, 0.58), (False, False, False, False))

        sides = assign_hand_sides([right_hand, left_hand], POSE, POSE_INDICES)

        self.assertIs(sides["left"], left_hand)
        self.assertIs(sides["right"], right_hand)

    def test_far_hands_are_ignored(self) -> None:
        far_hand = make_hand((0.05, 0.05), (True, True, True, True))

        self.assertEqual(assign_hand_sides([far_hand], POSE, POSE_INDICES), {})

    def test_one_hand_is_not_assigned_to_both_sides(self) -> None:
        hand = make_hand((0.55, 0.6), (True, True, True, True))

        sides = assign_hand_sides([hand], POSE, POSE_INDICES, max_distance=0.5)

        self.assertEqual(list(sides), ["right"])

    def test_hand_equally_close_to_both_wrists_is_ambiguous(self) -> None:
        hand = make_hand((0.5, 0.6), (True, True, True, True))

        self.assertEqual(assign_hand_sides([hand], POSE, POSE_INDICES, max_distance=0.5), {})


class HandWristAngleTests(unittest.TestCase):
    def test_straight_hand_gives_180_degrees(self) -> None:
        hand = make_hand((0.7, 0.6), (True, True, True, True))
        hand[9] = P(0.7, 0.4)

        angles = hand_wrist_angles([hand], POSE, POSE_INDICES, aspect_ratio=1.0)

        self.assertAlmostEqual(angles["right_wrist"], 180.0)
        self.assertNotIn("left_wrist", angles)

    def test_sideways_bend_gives_ninety_degrees(self) -> None:
        hand = make_hand((0.3, 0.6), (False, False, False, False))
        hand[9] = P(0.5, 0.6)

        angles = hand_wrist_angles([hand], POSE, POSE_INDICES, aspect_ratio=1.0)

        self.assertAlmostEqual(abs(angles["left_wrist"] - 180.0), 90.0)

    def test_both_bend_directions_are_distinguished_for_a_vertical_forearm(self) -> None:
        """The arm hanging down or raised is the common case; the sign used to vanish there."""
        vertical = [P(0.3, 0.6), P(0.7, 0.6), P(0.3, 0.3), P(0.7, 0.3)]
        toward_the_body = make_hand((0.3, 0.6), (True, True, True, True))
        toward_the_body[9] = P(0.4, 0.75)
        away_from_the_body = make_hand((0.3, 0.6), (True, True, True, True))
        away_from_the_body[9] = P(0.2, 0.75)

        one = hand_wrist_angles([toward_the_body], vertical, POSE_INDICES, aspect_ratio=1.0)["left_wrist"]
        other = hand_wrist_angles([away_from_the_body], vertical, POSE_INDICES, aspect_ratio=1.0)["left_wrist"]

        self.assertNotAlmostEqual(one, other)
        self.assertAlmostEqual(one + other, 360.0, places=3)

    def test_both_wrists_report_the_same_anatomical_bend_the_same_way(self) -> None:
        """The arms are mirror images, so an unmirrored screen rotation inverts one of them."""
        pose = [P(0.65, 0.60), P(0.35, 0.60), P(0.65, 0.85), P(0.35, 0.85)]
        indices = {"LEFT_WRIST": 0, "RIGHT_WRIST": 1, "LEFT_ELBOW": 2, "RIGHT_ELBOW": 3}
        left = make_hand((0.65, 0.60), (True, True, True, True))
        left[9] = P(0.58, 0.45)
        right = make_hand((0.35, 0.60), (True, True, True, True))
        right[9] = P(0.42, 0.45)

        angles = hand_wrist_angles([left, right], pose, indices, aspect_ratio=16 / 9)

        self.assertAlmostEqual(angles["left_wrist"], angles["right_wrist"], places=6)
        self.assertNotAlmostEqual(angles["left_wrist"], 180.0)

    def test_a_hand_pointing_at_the_camera_is_left_unmeasured(self) -> None:
        """Its projection is a few pixels long, so its direction would be pure noise."""
        hand = make_hand((0.7, 0.6), (True, True, True, True))
        hand[9] = P(0.7, 0.6, -0.2)

        self.assertEqual(hand_wrist_angles([hand], POSE, POSE_INDICES, aspect_ratio=1.0), {})

    def test_pose_depth_no_longer_moves_the_reported_angle(self) -> None:
        """Pose z and hand z use different origins; mixing them swung the joint by tens of degrees."""
        flat = [P(0.3, 0.6), P(0.7, 0.6), P(0.3, 0.35), P(0.7, 0.35)]
        deep = [P(0.3, 0.6), P(0.7, 0.6), P(0.3, 0.35, 0.2), P(0.7, 0.35)]
        hand = make_hand((0.3, 0.6), (True, True, True, True))
        hand[9] = P(0.4, 0.7)

        without_depth = hand_wrist_angles([hand], flat, POSE_INDICES, aspect_ratio=1.0)["left_wrist"]
        with_depth = hand_wrist_angles([hand], deep, POSE_INDICES, aspect_ratio=1.0)["left_wrist"]

        self.assertAlmostEqual(without_depth, with_depth)

    def test_bend_direction_is_signed_whatever_the_forearm_direction(self) -> None:
        for forearm in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.7, -0.7, 0.0)):
            with self.subTest(forearm=forearm):
                angle = math.atan2(forearm[1], forearm[0])
                one = (math.cos(angle - math.radians(30)), math.sin(angle - math.radians(30)), 0.0)
                other = (math.cos(angle + math.radians(30)), math.sin(angle + math.radians(30)), 0.0)

                self.assertAlmostEqual(signed_wrist_deviation(forearm, one), 30.0)
                self.assertAlmostEqual(signed_wrist_deviation(forearm, other), -30.0)

    def test_degenerate_vectors_give_no_angle(self) -> None:
        forearm = (1.0, 0.0, 0.0)

        self.assertAlmostEqual(signed_wrist_deviation(forearm, (1.0, 0.0, 0.0)), 0.0)
        # A fully folded hand sits exactly on the boundary between the two directions.
        self.assertEqual(abs(signed_wrist_deviation(forearm, (-1.0, 0.0, 0.0))), 90.0)
        self.assertIsNone(signed_wrist_deviation(forearm, (0.0, 0.0, 0.0)))
        self.assertIsNone(signed_wrist_deviation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        self.assertIsNone(signed_wrist_deviation(forearm, (float("nan"), 0.0, 0.0)))

    def test_the_two_bend_directions_land_on_opposite_sides_of_180(self) -> None:
        horizontal_forearm_pose = [P(0.3, 0.6), P(0.7, 0.6), P(0.3, 0.9), P(0.4, 0.6)]
        hand = make_hand((0.7, 0.6), (True, True, True, True))
        hand[9] = P(0.9, 0.8)

        angles = hand_wrist_angles([hand], horizontal_forearm_pose, POSE_INDICES, aspect_ratio=1.0)

        self.assertAlmostEqual(angles["right_wrist"], 135.0)

    def test_missing_elbow_index_is_skipped(self) -> None:
        hand = make_hand((0.7, 0.6), (True, True, True, True))

        angles = hand_wrist_angles([hand], POSE[:2], {"LEFT_WRIST": 0, "RIGHT_WRIST": 1})

        self.assertEqual(angles, {})


class WristAngleHoldTests(unittest.TestCase):
    def test_holds_recent_value_and_forgets_old_one(self) -> None:
        hold = WristAngleHold(max_age_ms=500)

        self.assertEqual(hold.update({"right_wrist": 150.0}, 1000), {"right_wrist": 150.0})
        self.assertEqual(hold.update({}, 1400), {"right_wrist": 150.0})
        self.assertEqual(hold.update({}, 1600), {})

    def test_an_impossible_jump_is_slewed_instead_of_followed(self) -> None:
        """A foreshortened hand can flip the measured angle; no wrist moves 150 deg in 33 ms."""
        hold = WristAngleHold(max_rate_deg_s=240.0)
        hold.update({"left_wrist": 90.0}, 1000)

        after_one_frame = hold.update({"left_wrist": 240.0}, 1033)["left_wrist"]

        self.assertAlmostEqual(after_one_frame, 90.0 + 240.0 * 0.033, places=3)

    def test_a_plausible_change_passes_through_untouched(self) -> None:
        hold = WristAngleHold(max_rate_deg_s=240.0)
        hold.update({"left_wrist": 150.0}, 1000)

        self.assertAlmostEqual(hold.update({"left_wrist": 155.0}, 1033)["left_wrist"], 155.0)

    def test_the_first_value_after_a_reset_is_taken_as_it_is(self) -> None:
        hold = WristAngleHold(max_rate_deg_s=240.0)
        hold.update({"left_wrist": 90.0}, 1000)
        hold.reset()

        self.assertAlmostEqual(hold.update({"left_wrist": 240.0}, 1033)["left_wrist"], 240.0)

    def test_new_value_replaces_held_one(self) -> None:
        hold = WristAngleHold(max_age_ms=500)
        hold.update({"left_wrist": 150.0}, 0)

        self.assertEqual(hold.update({"left_wrist": 120.0}, 200), {"left_wrist": 120.0})

    def test_a_slow_but_steady_camera_still_confirms_a_gesture(self) -> None:
        """Below about 4 fps every frame looked like a dropout and nothing was ever confirmed."""
        gesture_filter = HandGestureFilter()
        timestamp = 0
        confirmed: tuple[str, ...] = ()

        for _ in range(10):
            timestamp += 300
            confirmed = gesture_filter.update(("left_fist",), timestamp)

        self.assertEqual(confirmed, ("left_fist",))

    def test_a_real_dropout_still_clears_the_confirmed_gesture(self) -> None:
        gesture_filter = HandGestureFilter()
        for step in range(1, 6):
            gesture_filter.update(("left_fist",), step * 33)

        after_gap = gesture_filter.update(("left_fist",), 5 * 33 + 3000)

        self.assertEqual(after_gap, ())

    def test_reset_forgets_everything(self) -> None:
        hold = WristAngleHold()
        hold.update({"left_wrist": 150.0}, 0)
        hold.reset()

        self.assertEqual(hold.update({}, 10), {})


class HandGestureTests(unittest.TestCase):
    def test_gestures_are_named_by_side(self) -> None:
        left_hand = make_hand((0.3, 0.6), (True, True, True, True))
        right_hand = make_hand((0.7, 0.6), (False, False, False, False))

        gestures = detect_hand_gestures([left_hand, right_hand], POSE, POSE_INDICES)

        self.assertEqual(gestures, ("left_hand_open", "right_fist"))

    def test_no_hands_gives_no_gestures(self) -> None:
        self.assertEqual(detect_hand_gestures([], POSE, POSE_INDICES), ())


if __name__ == "__main__":
    unittest.main()
