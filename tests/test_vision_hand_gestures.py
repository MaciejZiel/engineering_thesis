from dataclasses import dataclass
import unittest

from vision_robot_arm.vision.hand_gestures import (
    assign_hand_sides,
    classify_hand,
    count_extended_fingers,
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


def make_hand(wrist: tuple[float, float], extended: tuple[bool, bool, bool, bool]) -> list[P]:
    hand = [P(*wrist) for _ in range(21)]
    finger_bases = (5, 9, 13, 17)
    for finger, (base_index, is_extended) in enumerate(zip(finger_bases, extended)):
        dx = (finger - 1.5) * 0.02
        for offset, step in enumerate((1, 2, 3, 4)):
            index = base_index + offset
            if is_extended:
                distance = 0.04 + 0.03 * step
            else:
                distance = 0.07 if step == 2 else 0.05
            hand[index] = P(wrist[0] + dx, wrist[1] - distance)
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
        hand = make_hand((0.5, 0.6), (True, True, True, True))

        sides = assign_hand_sides([hand], POSE, POSE_INDICES, max_distance=0.5)

        self.assertEqual(len(sides), 1)


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

        self.assertAlmostEqual(angles["left_wrist"], 90.0)

    def test_hand_toward_camera_uses_depth(self) -> None:
        hand = make_hand((0.7, 0.6), (True, True, True, True))
        hand[9] = P(0.7, 0.6, -0.2)

        angles = hand_wrist_angles([hand], POSE, POSE_INDICES, aspect_ratio=1.0)

        self.assertAlmostEqual(angles["right_wrist"], 90.0)

    def test_bend_direction_is_signed(self) -> None:
        forearm = (1.0, 0.0, 0.0)

        self.assertAlmostEqual(signed_wrist_deviation(forearm, (1.0, -1.0, 0.0)), 45.0)
        self.assertAlmostEqual(signed_wrist_deviation(forearm, (1.0, 1.0, 0.0)), -45.0)
        self.assertAlmostEqual(signed_wrist_deviation(forearm, (1.0, 0.0, 0.0)), 0.0)
        self.assertEqual(signed_wrist_deviation(forearm, (-1.0, 0.0, 0.0)), 90.0)
        self.assertIsNone(signed_wrist_deviation(forearm, (0.0, 0.0, 0.0)))

    def test_bending_down_gives_more_than_180(self) -> None:
        horizontal_forearm_pose = [P(0.3, 0.6), P(0.7, 0.6), P(0.3, 0.9), P(0.4, 0.6)]
        hand = make_hand((0.7, 0.6), (True, True, True, True))
        hand[9] = P(0.9, 0.8)

        angles = hand_wrist_angles([hand], horizontal_forearm_pose, POSE_INDICES, aspect_ratio=1.0)

        self.assertAlmostEqual(angles["right_wrist"], 225.0)

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

    def test_new_value_replaces_held_one(self) -> None:
        hold = WristAngleHold(max_age_ms=500)
        hold.update({"left_wrist": 150.0}, 0)

        self.assertEqual(hold.update({"left_wrist": 120.0}, 100), {"left_wrist": 120.0})

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
