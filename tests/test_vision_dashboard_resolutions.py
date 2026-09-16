import unittest

import cv2
import numpy as np

from vision_robot_arm.vision.dashboard import DashboardUi


COMMON_DASHBOARD_SIZES = (
    (1280, 720),
    (1728, 936),
    (2304, 1260),
    (3096, 1253),
    (3456, 1944),
)


class DashboardResolutionTests(unittest.TestCase):
    def test_renders_common_display_sizes_without_stretching_surface(self) -> None:
        camera = np.zeros((480, 640, 3), dtype=np.uint8)
        simulation = np.zeros((540, 960, 3), dtype=np.uint8)

        for width, height in COMMON_DASHBOARD_SIZES:
            with self.subTest(width=width, height=height):
                ui = DashboardUi(cv2, np, "resolution test")
                ui._canvas_size = (width, height)

                result = ui.render(
                    camera,
                    simulation,
                    mode="angles",
                    person_detected=True,
                    calibrated=True,
                    recording=False,
                    robot_label="sim",
                    gestures=("right_hand_open",),
                    status_lines=("robot connected",),
                    tracking_quality=0.9,
                    fps=30.0,
                    source_label="CAM 0",
                )

                self.assertEqual(result.shape, (height, width, 3))
                self.assertEqual(result.dtype, np.uint8)
                self.assertGreater(int(result.max()), 0)


if __name__ == "__main__":
    unittest.main()
