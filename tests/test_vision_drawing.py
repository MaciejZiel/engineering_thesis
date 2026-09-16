from dataclasses import dataclass
import unittest

from vision_robot_arm.vision.drawing import draw_joint_angle_labels, joint_label


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class FakeFrame:
    shape = (100, 200, 3)


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self) -> None:
        self.texts: list[tuple[str, tuple[int, int]]] = []

    def putText(self, frame: object, text: str, position: tuple[int, int], *args: object) -> None:
        self.texts.append((text, position))


class JointAngleLabelTests(unittest.TestCase):
    def test_labels_are_drawn_next_to_reliable_joints(self) -> None:
        cv2 = FakeCv2()
        indices = {"LEFT_SHOULDER": 0, "LEFT_ELBOW": 1, "LEFT_WRIST": 2, "RIGHT_ELBOW": 3}
        landmarks = [
            FakeLandmark(0.5, 0.5),
            FakeLandmark(0.25, 0.5),
            FakeLandmark(0.25, 0.8),
            FakeLandmark(0.75, 0.5, visibility=0.1),
        ]

        draw_joint_angle_labels(
            cv2,
            FakeFrame(),
            landmarks,
            indices,
            {"left_elbow": 91.4, "right_elbow": 120.0, "left_shoulder": None},
            min_visibility=0.55,
            joints=("left_elbow", "right_elbow", "left_shoulder"),
        )

        labels = {text for text, _ in cv2.texts}
        self.assertEqual(labels, {"L elbow 91"})
        self.assertEqual(cv2.texts[0][1], (60, 40))

    def test_joint_label_is_short(self) -> None:
        self.assertEqual(joint_label("right_shoulder"), "R shoulder")
        self.assertEqual(joint_label("left_elbow"), "L elbow")


if __name__ == "__main__":
    unittest.main()
