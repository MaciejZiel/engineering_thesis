import unittest

from vision_robot_arm.core.pose_state import LandmarkPoint
from vision_robot_arm.vision.smoothing import AngleSmoother, LandmarkSmoother


class AngleSmootherTests(unittest.TestCase):
    def test_uses_configured_alpha(self) -> None:
        smoother = AngleSmoother(alpha=0.5)

        self.assertEqual(smoother.update({"left_elbow": 10.0}), {"left_elbow": 10.0})
        self.assertEqual(smoother.update({"left_elbow": 20.0}), {"left_elbow": 15.0})

    def test_low_alpha_smooths_more_than_high_alpha(self) -> None:
        slow = AngleSmoother(alpha=0.1)
        fast = AngleSmoother(alpha=0.9)
        for smoother in (slow, fast):
            smoother.update({"left_elbow": 0.0})

        slow_value = slow.update({"left_elbow": 100.0})["left_elbow"]
        fast_value = fast.update({"left_elbow": 100.0})["left_elbow"]

        self.assertAlmostEqual(slow_value, 10.0)
        self.assertAlmostEqual(fast_value, 90.0)

    def test_missing_angles_pass_through_as_none(self) -> None:
        smoother = AngleSmoother(alpha=0.5)
        smoother.update({"left_elbow": 10.0})

        smoothed = smoother.update({"left_elbow": None})

        self.assertIsNone(smoothed["left_elbow"])

    def test_reset_forgets_previous_values(self) -> None:
        smoother = AngleSmoother(alpha=0.5)
        smoother.update({"left_elbow": 10.0})
        smoother.reset()

        self.assertEqual(smoother.update({"left_elbow": 20.0}), {"left_elbow": 20.0})


class LandmarkSmootherTests(unittest.TestCase):
    def test_smooths_coordinates_and_keeps_current_visibility(self) -> None:
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.update([LandmarkPoint(0.0, 0.0, 0.0, visibility=0.9)])

        smoothed = smoother.update([LandmarkPoint(1.0, 2.0, 4.0, visibility=0.8)])

        self.assertEqual(smoothed, [LandmarkPoint(0.5, 1.0, 2.0, visibility=0.8)])

    def test_low_visibility_points_pass_through_unsmoothed(self) -> None:
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.update([LandmarkPoint(0.0, 0.0, 0.0, visibility=0.9)])

        smoothed = smoother.update([LandmarkPoint(1.0, 2.0, 4.0, visibility=0.3)])

        self.assertEqual(smoothed, [LandmarkPoint(1.0, 2.0, 4.0, visibility=0.3)])

    def test_restarts_when_landmark_count_changes(self) -> None:
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.update([LandmarkPoint(0.0, 0.0, 0.0)])
        new_landmarks = [LandmarkPoint(1.0, 1.0, 1.0), LandmarkPoint(2.0, 2.0, 2.0)]

        self.assertEqual(smoother.update(new_landmarks), new_landmarks)


if __name__ == "__main__":
    unittest.main()
