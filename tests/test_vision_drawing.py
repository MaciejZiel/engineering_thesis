from dataclasses import dataclass
import unittest

import numpy as np

from vision_robot_arm.vision.drawing import (
    draw_joint_angle_labels,
    draw_overlay,
    draw_tracking_frame,
    joint_label,
    landmark_visibility_ratio,
)


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self) -> None:
        self.texts: list[tuple[str, tuple[int, int]]] = []
        self.fills = 0

    def getTextSize(self, text: str, font: int, scale: float, thickness: int) -> tuple[tuple[int, int], int]:
        return (int(len(text) * 10 * scale), int(20 * scale)), 2

    def addWeighted(self, *args: object) -> None:
        self.fills += 1

    def putText(self, frame: object, text: str, position: tuple[int, int], *args: object) -> None:
        self.texts.append((text, position))

    def line(self, *args: object) -> None:
        pass


class JointAngleLabelTests(unittest.TestCase):
    def test_labels_are_drawn_next_to_reliable_joints(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        indices = {"LEFT_SHOULDER": 0, "LEFT_ELBOW": 1, "LEFT_WRIST": 2, "RIGHT_ELBOW": 3}
        landmarks = [
            FakeLandmark(0.5, 0.5),
            FakeLandmark(0.25, 0.5),
            FakeLandmark(0.25, 0.8),
            FakeLandmark(0.75, 0.5, visibility=0.1),
        ]

        draw_joint_angle_labels(
            cv2,
            frame,
            landmarks,
            indices,
            {"left_elbow": 91.4, "right_elbow": 120.0, "left_shoulder": None},
            min_visibility=0.55,
            joints=("left_elbow", "right_elbow", "left_shoulder"),
        )

        self.assertEqual(cv2.texts, [("L elbow 91", (330, 350))])
        self.assertEqual(cv2.fills, 1)

    def test_joint_label_is_short(self) -> None:
        self.assertEqual(joint_label("right_shoulder"), "R shoulder")
        self.assertEqual(joint_label("left_elbow"), "L elbow")


class OverlayTests(unittest.TestCase):
    def test_overlay_draws_status_gestures_and_key_hint_once(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        draw_overlay(
            cv2,
            frame,
            "angles",
            person_detected=True,
            calibrated=True,
            recording=False,
            robot_label="sim",
            gestures=("right_hand_up",),
            status_lines=("sim shoulder= 90.0 ->  90.0",),
        )

        texts = [text for text, _ in cv2.texts]
        self.assertEqual(texts[0], "person   robot: sim   calibrated")
        self.assertEqual(texts[1], "gestures: right_hand_up")
        self.assertEqual(texts[2], "sim shoulder= 90.0 ->  90.0")
        self.assertTrue(texts[3].startswith("output: angles"))
        self.assertEqual(len(texts), 4)
        self.assertEqual(cv2.fills, 2)

    def test_hint_sits_at_the_bottom_of_the_frame(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        draw_overlay(cv2, frame, "angles", person_detected=False)

        _, hint_position = cv2.texts[-1]
        self.assertGreater(hint_position[1], 330)
        self.assertLess(hint_position[1], 360)


class TrackingFrameTests(unittest.TestCase):
    def test_visibility_ratio_counts_reliable_landmarks(self) -> None:
        landmarks = [
            FakeLandmark(0.2, 0.2, visibility=0.9),
            FakeLandmark(0.4, 0.4, visibility=0.6),
            FakeLandmark(0.6, 0.6, visibility=0.2),
            FakeLandmark(0.8, 0.8, visibility=0.1),
        ]

        self.assertEqual(landmark_visibility_ratio(landmarks, 0.55), 0.5)

    def test_draws_tracking_label_for_reliable_pose(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        landmarks = [
            FakeLandmark(0.2, 0.2),
            FakeLandmark(0.8, 0.2),
            FakeLandmark(0.2, 0.8),
            FakeLandmark(0.8, 0.8),
        ]

        ratio = draw_tracking_frame(cv2, frame, landmarks, 0.55)

        self.assertEqual(ratio, 1.0)
        self.assertTrue(any(text == "TRACKED 100%" for text, _ in cv2.texts))


if __name__ == "__main__":
    unittest.main()
