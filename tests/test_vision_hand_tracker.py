import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from vision_robot_arm.core.config import AppConfig
from vision_robot_arm.vision.hand_tracker import HandTracker


class HandTrackerConfigurationTests(unittest.TestCase):
    def test_hand_thresholds_are_independent_and_not_silently_capped(self):
        options = Mock()
        deps = SimpleNamespace(
            base_options=Mock(),
            vision=SimpleNamespace(
                HandLandmarkerOptions=options,
                RunningMode=SimpleNamespace(VIDEO="video"),
                HandLandmarker=Mock(),
            ),
        )
        tracker = HandTracker(deps, AppConfig(
            hand_detection_confidence=0.85, hand_presence_confidence=0.75
        ))
        self.assertEqual(options.call_args.kwargs["min_hand_detection_confidence"], 0.85)
        self.assertEqual(options.call_args.kwargs["min_hand_presence_confidence"], 0.75)
        tracker.close()

    def test_invalid_thresholds_are_rejected_before_inference(self):
        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(SystemExit):
                AppConfig(hand_detection_confidence=value).validate()
