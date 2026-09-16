import unittest

import numpy as np

from vision_robot_arm.robot.backend import DebugBackend, TargetTracker
from vision_robot_arm.robot.targets import GRIPPER_CLOSE, GRIPPER_OPEN, ArmState, JointTargets
from vision_robot_arm.robot.visualization import arm_points, draw_arm_panel


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def getTextSize(self, text: str, font: int, scale: float, thickness: int) -> tuple[tuple[int, int], int]:
        return (int(len(text) * 10 * scale), int(20 * scale)), 2

    def addWeighted(self, *args: object) -> None:
        self.calls.append(("addWeighted", args))

    def rectangle(self, *args: object) -> None:
        self.calls.append(("rectangle", args))

    def line(self, *args: object) -> None:
        self.calls.append(("line", args))

    def circle(self, *args: object) -> None:
        self.calls.append(("circle", args))

    def putText(self, *args: object) -> None:
        self.calls.append(("putText", args))


class ArmPointsTests(unittest.TestCase):
    def test_arm_hanging_down_at_zero_degrees(self) -> None:
        base, elbow, wrist = arm_points(0.0, 180.0, (100, 100), 50.0)

        self.assertEqual(base, (100, 100))
        self.assertEqual(elbow, (100, 150))
        self.assertEqual(wrist, (100, 200))

    def test_arm_horizontal_at_ninety_degrees(self) -> None:
        _, elbow, wrist = arm_points(90.0, 180.0, (100, 100), 50.0)

        self.assertEqual(elbow, (150, 100))
        self.assertEqual(wrist, (200, 100))

    def test_bent_elbow_folds_forearm_upwards(self) -> None:
        _, elbow, wrist = arm_points(90.0, 90.0, (100, 100), 50.0)

        self.assertEqual(elbow, (150, 100))
        self.assertEqual(wrist, (150, 50))


class DrawArmPanelTests(unittest.TestCase):
    def test_draws_panel_arms_gripper_and_text(self) -> None:
        cv2 = FakeCv2()
        state = ArmState(
            joints={"shoulder": 90.0, "elbow": 150.0},
            targets={"shoulder": 120.0, "elbow": 150.0},
            gripper=GRIPPER_CLOSE,
            lift_mode=True,
        )

        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        draw_arm_panel(cv2, frame, state=state, origin=(10, 20), size=200)

        kinds = [kind for kind, _ in cv2.calls]
        self.assertEqual(kinds.count("addWeighted"), 1)
        self.assertEqual(kinds.count("rectangle"), 1)
        self.assertEqual(kinds.count("line"), 6)
        texts = [args[1] for kind, args in cv2.calls if kind == "putText"]
        self.assertIn("shoulder  90.0 -> 120.0", texts)
        self.assertIn("elbow 150.0 -> 150.0", texts)
        self.assertIn("gripper close   lift on", texts)

    def test_missing_joints_are_reported_as_not_available(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        state = ArmState(joints={}, targets={}, gripper=GRIPPER_OPEN, lift_mode=False)

        draw_arm_panel(cv2, frame, state=state, origin=(0, 0))

        texts = [args[1] for kind, args in cv2.calls if kind == "putText"]
        self.assertIn("shoulder n/a", texts)


class TargetTrackerTests(unittest.TestCase):
    def test_no_state_before_first_targets(self) -> None:
        self.assertIsNone(TargetTracker().arm_state())
        self.assertIsNone(DebugBackend(print_interval=1.0).arm_state())

    def test_latches_joints_and_gripper(self) -> None:
        tracker = TargetTracker()
        tracker.update(JointTargets(timestamp_ms=1, joints={"shoulder": 30.0}, gripper=GRIPPER_CLOSE))
        tracker.update(JointTargets(timestamp_ms=2, joints={"elbow": 100.0}, lift_mode=True))

        state = tracker.arm_state()

        self.assertEqual(state.joints, {"shoulder": 30.0, "elbow": 100.0})
        self.assertEqual(state.targets, state.joints)
        self.assertEqual(state.gripper, GRIPPER_CLOSE)
        self.assertTrue(state.lift_mode)
        self.assertTrue(state.settled)


if __name__ == "__main__":
    unittest.main()
