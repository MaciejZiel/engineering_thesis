import unittest

import numpy as np

from vision_robot_arm.robot.backend import DebugBackend, TargetTracker
from vision_robot_arm.robot.targets import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    ArmState,
    ArmTargets,
    JointTargets,
    RobotState,
)
from vision_robot_arm.robot.visualization import arm_points, draw_arm_panel, draw_simulation


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def getTextSize(self, text: str, font: int, scale: float, thickness: int) -> tuple[tuple[int, int], int]:
        return (int(len(text) * 10 * scale), int(20 * scale)), 2

    def rectangle(self, *args: object) -> None:
        self.calls.append(("rectangle", args))

    def line(self, *args: object) -> None:
        self.calls.append(("line", args))

    def circle(self, *args: object) -> None:
        self.calls.append(("circle", args))

    def putText(self, *args: object) -> None:
        self.calls.append(("putText", args))

    def texts(self) -> list[str]:
        return [args[1] for kind, args in self.calls if kind == "putText"]

    def count(self, kind: str) -> int:
        return sum(1 for call_kind, _ in self.calls if call_kind == kind)


def arm_state(shoulder: float, elbow: float, wrist: float, gripper: str = GRIPPER_OPEN) -> ArmState:
    joints = {"shoulder": shoulder, "elbow": elbow, "wrist_1": wrist}
    return ArmState(joints=joints, targets=dict(joints), gripper=gripper)


class ArmPointsTests(unittest.TestCase):
    def test_arm_hanging_down_at_minus_180_degrees(self) -> None:
        base, elbow, wrist, tip = arm_points(-180.0, 0.0, 0.0, (100, 100), 50.0)

        self.assertEqual(base, (100, 100))
        self.assertEqual(elbow, (100, 150))
        self.assertEqual(wrist, (100, 200))
        self.assertEqual(tip, (100, 230))

    def test_arm_horizontal_at_minus_ninety_degrees(self) -> None:
        _, elbow, wrist, _ = arm_points(-90.0, 0.0, 0.0, (100, 100), 50.0)

        self.assertEqual(elbow, (150, 100))
        self.assertEqual(wrist, (200, 100))

    def test_bent_elbow_and_wrist_fold_upwards(self) -> None:
        _, elbow, wrist, tip = arm_points(-90.0, 90.0, 90.0, (100, 100), 50.0)

        self.assertEqual(elbow, (150, 100))
        self.assertEqual(wrist, (150, 50))
        self.assertEqual(tip, (120, 50))

    def test_mirrored_arm_points_the_other_way(self) -> None:
        _, elbow, wrist, _ = arm_points(-90.0, 0.0, 0.0, (100, 100), 50.0, mirror=True)

        self.assertEqual(elbow, (50, 100))
        self.assertEqual(wrist, (0, 100))


class DrawArmPanelTests(unittest.TestCase):
    def test_draws_title_badge_sketch_and_readout(self) -> None:
        cv2 = FakeCv2()
        state = ArmState(
            joints={"shoulder": -90.0, "elbow": 30.0, "wrist_1": 10.0},
            targets={"shoulder": -60.0, "elbow": 30.0, "wrist_1": 10.0},
            gripper=GRIPPER_CLOSE,
        )
        frame = np.zeros((540, 960, 3), dtype=np.uint8)

        draw_arm_panel(cv2, frame, state, (10, 20), (300, 460), title="RIGHT UR7e")

        texts = cv2.texts()
        self.assertIn("RIGHT UR7e", texts)
        self.assertIn("GRIP CLOSED", texts)
        self.assertEqual(texts.count("S") + texts.count("E") + texts.count("W1"), 3)
        self.assertIn("shoulder", texts)
        self.assertIn("wrist 1", texts)
        self.assertIn(" -90.0", texts)
        self.assertIn(" -60.0", texts)
        self.assertGreaterEqual(cv2.count("circle"), 3)
        self.assertGreater(cv2.count("line"), 10)

    def test_missing_state_shows_placeholders(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((540, 960, 3), dtype=np.uint8)

        draw_arm_panel(cv2, frame, None, (0, 0), (300, 400))

        texts = cv2.texts()
        self.assertIn("NO DATA", texts)
        self.assertEqual(texts.count("n/a"), 6)

    def test_open_gripper_badge(self) -> None:
        cv2 = FakeCv2()
        frame = np.zeros((540, 960, 3), dtype=np.uint8)

        draw_arm_panel(cv2, frame, arm_state(-90.0, 0.0, 0.0), (0, 0), (300, 400))

        self.assertIn("GRIP OPEN", cv2.texts())


class DrawSimulationTests(unittest.TestCase):
    def test_draws_left_and_right_panels_and_footer(self) -> None:
        cv2 = FakeCv2()
        canvas = np.full((460, 618, 3), 255, dtype=np.uint8)
        state = RobotState(
            arms={"right": arm_state(-90.0, 0.0, 0.0), "left": arm_state(-135.0, 90.0, 60.0)},
            lift_mode=True,
        )

        draw_simulation(cv2, canvas, state)

        texts = cv2.texts()
        self.assertIn("LEFT UR7e", texts)
        self.assertIn("RIGHT UR7e", texts)
        self.assertIn("LIFT ON", texts)
        self.assertLess(texts.index("LEFT UR7e"), texts.index("RIGHT UR7e"))
        self.assertEqual(int(canvas[0, 0, 0]), 23)

    def test_draws_placeholders_without_state(self) -> None:
        cv2 = FakeCv2()
        canvas = np.zeros((300, 480, 3), dtype=np.uint8)

        draw_simulation(cv2, canvas, None)

        texts = cv2.texts()
        self.assertEqual(texts.count("n/a"), 12)
        self.assertEqual(texts.count("NO DATA"), 2)
        self.assertIn("LIFT OFF", texts)

    def test_tiny_canvas_does_not_crash(self) -> None:
        cv2 = FakeCv2()
        canvas = np.zeros((20, 30, 3), dtype=np.uint8)

        draw_simulation(cv2, canvas, None)


class TargetTrackerTests(unittest.TestCase):
    def test_no_state_before_first_targets(self) -> None:
        self.assertIsNone(TargetTracker().robot_state())
        self.assertIsNone(DebugBackend(print_interval=1.0).robot_state())

    def test_latches_joints_and_grippers_per_arm(self) -> None:
        tracker = TargetTracker()
        tracker.update(
            JointTargets(
                timestamp_ms=1,
                arms={"right": ArmTargets(joints={"shoulder": 30.0}, gripper=GRIPPER_CLOSE)},
            )
        )
        tracker.update(
            JointTargets(
                timestamp_ms=2,
                arms={"right": ArmTargets(joints={"wrist_1": 100.0}), "left": ArmTargets(joints={"elbow": 60.0})},
                lift_mode=True,
            )
        )

        state = tracker.robot_state()

        self.assertEqual(state.arm("right").joints, {"shoulder": 30.0, "wrist_1": 100.0})
        self.assertEqual(state.arm("right").targets, state.arm("right").joints)
        self.assertEqual(state.arm("right").gripper, GRIPPER_CLOSE)
        self.assertEqual(state.arm("left").joints, {"elbow": 60.0})
        self.assertEqual(state.arm("left").gripper, GRIPPER_OPEN)
        self.assertTrue(state.lift_mode)
        self.assertTrue(state.arm("right").settled)


if __name__ == "__main__":
    unittest.main()
