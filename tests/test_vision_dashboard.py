import unittest

from vision_robot_arm.core.config import ANGLE_MODE, BOTH_MODE, LANDMARK_MODE
from vision_robot_arm.vision.dashboard import Rect, cycle_output_mode, fit_inside


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


if __name__ == "__main__":
    unittest.main()
