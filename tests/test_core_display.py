import unittest

from vision_robot_arm.core.display import DisplayMetrics, preferred_dashboard_size


class DisplayMetricsTests(unittest.TestCase):
    def test_reports_dpi_scale(self) -> None:
        self.assertEqual(DisplayMetrics(2560, 1440, dpi=144).scale, 1.5)


class PreferredDashboardSizeTests(unittest.TestCase):
    def test_uses_ninety_percent_of_full_hd_work_area(self) -> None:
        size = preferred_dashboard_size(DisplayMetrics(1920, 1040))

        self.assertEqual(size, (1728, 936))

    def test_adapts_to_ultrawide_monitor(self) -> None:
        size = preferred_dashboard_size(DisplayMetrics(3440, 1400))

        self.assertEqual(size, (3096, 1260))

    def test_never_exceeds_a_small_display(self) -> None:
        size = preferred_dashboard_size(DisplayMetrics(800, 480))

        self.assertEqual(size, (800, 480))

    def test_rejects_unreasonable_coverage(self) -> None:
        with self.assertRaises(ValueError):
            preferred_dashboard_size(DisplayMetrics(1920, 1080), coverage=0.2)


if __name__ == "__main__":
    unittest.main()
