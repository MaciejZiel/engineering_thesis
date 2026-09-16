import unittest

import cv2
import numpy as np

from vision_robot_arm.vision.dashboard import DashboardUi, dashboard_layout

COMMON_DASHBOARD_SIZES = (
    (640, 480),
    (960, 540),
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

                layout = dashboard_layout(width, height)
                for rect in (
                    layout.camera,
                    layout.preview,
                    layout.status,
                    layout.footer,
                ):
                    self.assertGreater(rect.width, 0)
                    self.assertGreater(rect.height, 0)
                    self.assertGreaterEqual(rect.x, 0)
                    self.assertGreaterEqual(rect.y, 0)
                    self.assertLessEqual(rect.right, width)
                    self.assertLessEqual(rect.bottom, height)
                self.assertLess(layout.camera.right, layout.preview.x)
                self.assertLess(layout.preview.bottom, layout.status.y)
                self.assertLess(layout.status.bottom, layout.footer.y)
                for i, button in enumerate(ui.buttons):
                    self.assertLessEqual(button.rect.right, width)
                    self.assertLessEqual(button.rect.bottom, height)
                    for other in ui.buttons[i + 1 :]:
                        a, b = button.rect, other.rect
                        self.assertFalse(
                            a.x < b.right
                            and a.right > b.x
                            and a.y < b.bottom
                            and a.bottom > b.y
                        )


if __name__ == "__main__":
    unittest.main()
