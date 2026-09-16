from dataclasses import dataclass
import unittest

import numpy as np

from vision_robot_arm.vision.drawing import draw_hands
from vision_robot_arm.vision.hand_gestures import assign_hand_sides


@dataclass
class P:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class FakeCv2:
    LINE_AA = 16

    def __init__(self) -> None:
        self.lines: list[tuple] = []
        self.circles: list[tuple] = []

    def line(self, frame, start, end, color, thickness, *args) -> None:
        self.lines.append((start, end, color, thickness))

    def circle(self, frame, center, radius, color, thickness, *args) -> None:
        self.circles.append((center, radius, color))


def make_hand(origin: tuple[float, float], span: float = 0.05) -> list[P]:
    return [P(origin[0] + span * (index % 5) / 4, origin[1] - span * (index // 5) / 4) for index in range(21)]


class DrawHandsTests(unittest.TestCase):
    def test_draws_palm_fingers_and_wrist_link(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        indices = {"RIGHT_WRIST": 0}
        landmarks = [P(0.5, 0.5)]
        hand = make_hand((0.5, 0.5))

        draw_hands(cv2, frame, {"right": hand}, landmarks, indices, min_visibility=0.55)

        thick_lines = [line for line in cv2.lines if line[3] > 2]
        self.assertEqual(len(thick_lines), 1)
        self.assertEqual(thick_lines[0][0], (640, 360))
        self.assertGreater(len(cv2.lines), 20)
        self.assertGreaterEqual(len(cv2.circles), 21)

    def test_skips_wrist_link_when_pose_wrist_unreliable(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        landmarks = [P(0.5, 0.5, visibility=0.1)]

        draw_hands(cv2, frame, {"right": make_hand((0.5, 0.5))}, landmarks, {"RIGHT_WRIST": 0}, 0.55)

        self.assertEqual([line for line in cv2.lines if line[3] > 2], [])

    def test_incomplete_hand_is_ignored(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        draw_hands(cv2, frame, {"right": [P(0.5, 0.5)] * 5}, [P(0.5, 0.5)], {"RIGHT_WRIST": 0}, 0.55)

        self.assertEqual(cv2.lines, [])


class AdaptiveHandMatchingTests(unittest.TestCase):
    def test_large_hand_matches_from_further_away(self) -> None:
        pose = [P(0.3, 0.6), P(0.7, 0.6)]
        indices = {"LEFT_WRIST": 0, "RIGHT_WRIST": 1}
        big_hand = [P(0.7, 0.85) for _ in range(21)]
        big_hand[9] = P(0.7, 0.55)

        sides = assign_hand_sides([big_hand], pose, indices)

        self.assertIs(sides.get("right"), big_hand)

    def test_small_hand_keeps_strict_distance(self) -> None:
        pose = [P(0.3, 0.6), P(0.7, 0.6)]
        indices = {"LEFT_WRIST": 0, "RIGHT_WRIST": 1}
        small_hand = [P(0.7, 0.85) for _ in range(21)]
        small_hand[9] = P(0.7, 0.82)

        self.assertEqual(assign_hand_sides([small_hand], pose, indices), {})


if __name__ == "__main__":
    unittest.main()
