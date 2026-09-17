import unittest

import cv2
import numpy as np

from vision_robot_arm.core.config import ANGLE_MODE, BOTH_MODE, LANDMARK_MODE
from vision_robot_arm.vision.dashboard import (
    ACTION_CALIBRATE,
    ACTION_DETAILS,
    ACTION_JOG_NEGATIVE,
    ACTION_JOG_POSITIVE,
    ACTION_RECORD,
    ACTION_STOP,
    DashboardUi,
    Rect,
    backend_label,
    cycle_output_mode,
    fit_inside,
)
from vision_robot_arm.vision.ui_style import Painter, font


class RectTests(unittest.TestCase):
    def test_contains_includes_top_left_and_excludes_bottom_right(self) -> None:
        rect = Rect(10, 20, 100, 50)

        self.assertTrue(rect.contains(10, 20))
        self.assertTrue(rect.contains(109, 69))
        self.assertFalse(rect.contains(110, 70))


class FitInsideTests(unittest.TestCase):
    def test_wide_image_is_letterboxed_vertically(self) -> None:
        fitted = fit_inside((1920, 1080), Rect(10, 20, 800, 600))

        self.assertEqual(fitted, Rect(10, 95, 800, 450))

    def test_tall_image_is_letterboxed_horizontally(self) -> None:
        fitted = fit_inside((640, 480), Rect(0, 0, 800, 300))

        self.assertEqual(fitted, Rect(200, 0, 400, 300))


class OutputModeTests(unittest.TestCase):
    def test_cycles_through_all_output_modes(self) -> None:
        self.assertEqual(cycle_output_mode(ANGLE_MODE), LANDMARK_MODE)
        self.assertEqual(cycle_output_mode(LANDMARK_MODE), BOTH_MODE)
        self.assertEqual(cycle_output_mode(BOTH_MODE), ANGLE_MODE)

    def test_unknown_mode_returns_angles(self) -> None:
        self.assertEqual(cycle_output_mode("unknown"), ANGLE_MODE)


class DashboardInteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = DashboardUi(cv2, np, "test")
        self.ui._canvas_size = (960, 540)
        self.camera = np.zeros((480, 640, 3), np.uint8)
        self.render()

    def render(self, detected: bool = True, recording: bool = False) -> np.ndarray:
        return self.ui.render(
            self.camera,
            self.camera,
            mode="angles",
            person_detected=detected,
            calibrated=False,
            recording=recording,
            robot_label="sim",
            gestures=(),
            status_lines=("robot diagnostic",),
            tracking_quality=0.8,
            fps=30,
            source_label="CAM 0",
        )

    def point(self, action: str) -> tuple[int, int]:
        rect = next(b.rect for b in self.ui.buttons if b.action == action)
        return rect.x + rect.width // 2, rect.y + rect.height // 2

    def click(self, action: str) -> None:
        x, y = self.point(action)
        self.ui._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        self.ui._on_mouse(cv2.EVENT_LBUTTONUP, x, y, 0, None)

    def test_disabled_calibration_cannot_be_clicked_or_keyboard_focused(self) -> None:
        self.render(detected=False)
        self.click(ACTION_CALIBRATE)
        self.assertIsNone(self.ui.consume_action())
        for _ in self.ui.buttons:
            self.ui.handle_key(9)
            self.assertNotEqual(self.ui._focus, ACTION_CALIBRATE)
        self.render(detected=True)
        self.click(ACTION_CALIBRATE)
        self.assertEqual(self.ui.consume_action(), ACTION_CALIBRATE)

    def test_pose_without_usable_angles_does_not_enable_calibration(self) -> None:
        self.ui.render(
            self.camera,
            self.camera,
            mode="angles",
            person_detected=True,
            can_calibrate=False,
            calibrated=False,
            recording=False,
            robot_label="sim",
            gestures=(),
            status_lines=(),
            tracking_quality=0.1,
            fps=30,
            source_label="CAM 0",
        )
        self.click(ACTION_CALIBRATE)
        self.assertIsNone(self.ui.consume_action())

    def test_release_outside_button_cancels_action(self) -> None:
        x, y = self.point(ACTION_RECORD)
        self.ui._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        self.ui._on_mouse(cv2.EVENT_LBUTTONUP, 0, 0, 0, None)
        self.assertIsNone(self.ui.consume_action())

    def test_release_without_press_does_not_trigger_action(self) -> None:
        x, y = self.point(ACTION_RECORD)
        self.ui._on_mouse(cv2.EVENT_LBUTTONUP, x, y, 0, None)
        self.assertIsNone(self.ui.consume_action())

    def test_button_dispatches_once_and_recording_changes_its_label(self) -> None:
        self.click(ACTION_RECORD)
        self.assertEqual(self.ui.consume_action(), ACTION_RECORD)
        self.assertIsNone(self.ui.consume_action())
        self.render(recording=True)
        button = next(b for b in self.ui.buttons if b.action == ACTION_RECORD)
        self.assertEqual(button.label, "Stop recording")
        self.assertTrue(button.active)

    def test_details_are_local_ui_state_and_keyboard_can_close_them(self) -> None:
        self.click(ACTION_DETAILS)
        self.assertTrue(self.ui._details)
        self.assertIsNone(self.ui.consume_action())
        self.ui.handle_key(ord("d"))
        self.assertFalse(self.ui._details)

    def test_keyboard_focus_can_activate_recording(self) -> None:
        for _ in self.ui.buttons:
            self.ui.handle_key(9)
            if self.ui._focus == ACTION_RECORD:
                break
        self.ui.handle_key(13)
        self.assertEqual(self.ui.consume_action(), ACTION_RECORD)

    def test_hover_changes_the_rendered_control(self) -> None:
        before = self.render()
        x, y = self.point(ACTION_RECORD)
        self.ui._on_mouse(cv2.EVENT_MOUSEMOVE, x, y, 0, None)
        after = self.render()
        self.assertFalse(np.array_equal(before, after))

    def test_backend_name_does_not_claim_hardware_feedback(self) -> None:
        self.assertEqual(backend_label("sim"), "Simulation")
        self.assertEqual(backend_label("ur"), "URScript output")
        self.assertEqual(backend_label("off"), "Preview only")

    def test_commissioning_jog_is_active_only_while_pointer_is_held_inside(self) -> None:
        self.ui.render(
            self.camera,
            self.camera,
            mode="angles",
            person_detected=True,
            calibrated=False,
            recording=False,
            robot_label="ur",
            gestures=(),
            status_lines=(),
            tracking_quality=0.8,
            fps=30,
            source_label="CAM 0",
            control_label="Disarm commissioning",
            commissioning_joint="shoulder",
        )
        actions = {button.action for button in self.ui.buttons}
        self.assertTrue(
            {ACTION_JOG_NEGATIVE, ACTION_JOG_POSITIVE, ACTION_STOP}.issubset(actions)
        )

        x, y = self.point(ACTION_JOG_NEGATIVE)
        self.ui._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        self.assertEqual(self.ui.held_action, ACTION_JOG_NEGATIVE)
        self.ui._on_mouse(cv2.EVENT_MOUSEMOVE, 0, 0, 0, None)
        self.assertIsNone(self.ui.held_action)
        self.ui._on_mouse(cv2.EVENT_LBUTTONUP, 0, 0, 0, None)
        self.assertIsNone(self.ui.consume_action())


class TypographyTests(unittest.TestCase):
    def test_fonts_are_bundled_and_labels_fit_their_available_width(self) -> None:
        self.assertEqual(font(14).getname()[0], "Lato")
        canvas = np.zeros((200, 300, 3), np.uint8)
        painter = Painter(cv2, np, canvas)
        label = painter.elide("A very long recording filename with Unicode — łódź", 100)
        self.assertTrue(label.endswith("…"))
        self.assertLessEqual(painter.measure(label), 100)
        painter.text(label, -5, -5, width=100)
        self.assertGreater(int(canvas.max()), 0)
        self.assertTrue(np.all(canvas[:, 100:] == 0))


if __name__ == "__main__":
    unittest.main()
