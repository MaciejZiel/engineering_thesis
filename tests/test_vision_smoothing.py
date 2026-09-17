import random
import statistics
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


class StillJointTests(unittest.TestCase):
    """Landmark noise used to be passed on as motion, so the robot twitched while you held still."""

    NOISE_DEG = 1.5

    def noisy_run(self, smoother: AngleSmoother, speed_deg_s: float) -> tuple[list[float], list[float]]:
        random.seed(11)
        timestamp, truth = 0, 120.0
        output, error = [], []
        for step in range(400):
            timestamp += 33
            truth += speed_deg_s * 0.033
            value = smoother.update({"a": truth + random.gauss(0, self.NOISE_DEG)}, timestamp)["a"]
            if step > 60:
                output.append(value)
                error.append(abs(truth - value))
        return output, error

    def test_a_still_joint_settles_far_below_the_measurement_noise(self) -> None:
        output, _ = self.noisy_run(AngleSmoother(0.35), speed_deg_s=0.0)

        self.assertLess(statistics.pstdev(output), self.NOISE_DEG / 3)

    def test_a_still_joint_stops_moving_at_all_through_the_dead_band(self) -> None:
        output, _ = self.noisy_run(AngleSmoother(0.35), speed_deg_s=0.0)

        self.assertLess(max(output) - min(output), 1.5)

    def test_a_sweeping_joint_keeps_the_configured_response(self) -> None:
        """Calming a still arm must not cost anything while it is actually moving."""
        _, adaptive = self.noisy_run(AngleSmoother(0.35), speed_deg_s=60.0)
        _, plain = self.noisy_run(AngleSmoother(0.35, motion_floor_deg_s=0.0), speed_deg_s=60.0)

        self.assertLess(statistics.mean(adaptive), statistics.mean(plain) + 0.5)

    def test_slow_deliberate_motion_is_still_followed(self) -> None:
        _, error = self.noisy_run(AngleSmoother(0.35), speed_deg_s=20.0)

        self.assertLess(statistics.mean(error), 3.0)


class LandmarkSmootherTests(unittest.TestCase):
    def test_same_elapsed_time_has_same_response_at_different_frame_rates(self):
        values = []
        for interval in (20, 50, 100):
            smoother = LandmarkSmoother(0.35, noise_floor=0)
            smoother.update([LandmarkPoint(0, 0, 0)], 0)
            for timestamp in range(interval, 301, interval):
                value = smoother.update([LandmarkPoint(1, 0, 0)], timestamp)
            values.append(value[0].x)
        self.assertAlmostEqual(values[0], values[1])
        self.assertAlmostEqual(values[1], values[2])

    def test_stale_and_rewound_landmarks_are_not_blended(self):
        for timestamp in (0, 1000):
            smoother = LandmarkSmoother(0.1)
            smoother.update([LandmarkPoint(0, 0, 0)], 100)
            result = smoother.update([LandmarkPoint(1, 0, 0)], timestamp)
            self.assertEqual(result[0].x, 1)

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

    def test_the_configured_visibility_threshold_is_honoured(self) -> None:
        """A lower --visibility-threshold must also mean those landmarks get smoothed."""
        smoother = LandmarkSmoother(alpha=0.5, min_visibility=0.3)
        smoother.update([LandmarkPoint(0.0, 0.0, 0.0, visibility=0.5)])

        smoothed = smoother.update([LandmarkPoint(1.0, 0.0, 0.0, visibility=0.5)])

        self.assertAlmostEqual(smoothed[0].x, 0.5)

    def test_restarts_when_landmark_count_changes(self) -> None:
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.update([LandmarkPoint(0.0, 0.0, 0.0)])
        new_landmarks = [LandmarkPoint(1.0, 1.0, 1.0), LandmarkPoint(2.0, 2.0, 2.0)]

        self.assertEqual(smoother.update(new_landmarks), new_landmarks)

    def test_metric_outlier_is_limited_by_physical_speed(self) -> None:
        smoother = LandmarkSmoother(alpha=1.0, noise_floor=0, max_speed=1.0)
        smoother.update([LandmarkPoint(0.0, 0.0, 0.0)], 1000)

        result = smoother.update([LandmarkPoint(0.0, 0.0, 2.0)], 1100)

        self.assertAlmostEqual(result[0].z, 0.1)

    def test_plausible_metric_motion_is_not_limited(self) -> None:
        smoother = LandmarkSmoother(alpha=1.0, noise_floor=0, max_speed=4.0)
        smoother.update([LandmarkPoint(0.0, 0.0, 0.0)], 1000)

        result = smoother.update([LandmarkPoint(0.0, 0.0, 0.1)], 1100)

        self.assertAlmostEqual(result[0].z, 0.1)


if __name__ == "__main__":
    unittest.main()
