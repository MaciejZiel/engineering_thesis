import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.mapping import RobotMapper
from vision_robot_arm.vision.calibration import PoseCalibration


class CalibrationProfileTests(unittest.TestCase):
    def test_stable_capture_requires_time_and_rejects_motion(self):
        calibration = PoseCalibration()
        samples = [(i * 100, {"right_elbow": 100 + (i % 2)}) for i in range(8)]
        self.assertEqual(calibration.capture_stable(samples[:3]), 0)
        self.assertEqual(calibration.capture_stable(samples, ("left_elbow",)), 0)
        self.assertEqual(calibration.capture_stable(samples), 1)
        self.assertAlmostEqual(calibration.relative_angles({"right_elbow": 100.5})["right_elbow"], 0)
        moving = [(i * 100, {"right_elbow": 100 + i * 5}) for i in range(8)]
        self.assertEqual(calibration.capture_stable(moving), 0)

    def test_profiles_roundtrip_and_invalid_load_preserves_current_profile(self):
        calibration = PoseCalibration()
        calibration.capture({"right_elbow": 120})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            calibration.save(path)
            calibration.reset()
            self.assertFalse(calibration.calibrated)
            calibration.load(path)
            self.assertEqual(calibration.relative_angles({"right_elbow": 130})["right_elbow"], 10)
            path.write_text('{"version": 1, "neutral_angles": {"right_elbow": NaN}}')
            with self.assertRaises(ValueError):
                calibration.load(path)
            self.assertEqual(calibration.relative_angles({"right_elbow": 130})["right_elbow"], 10)

    def test_mapper_uses_calibrated_offsets_around_robot_home(self):
        state = PoseState(1, [], None, {}, {"right_elbow": 120}, {"right_elbow": 0}, (), True)
        mapper = RobotMapper(RobotConfig())
        self.assertEqual(mapper.map(state).arm("right").joints["elbow"], 0)
        moved = replace(state, relative_angles={"right_elbow": -15})
        self.assertEqual(mapper.map(moved).arm("right").joints["elbow"], 15)

    def test_empty_capture_does_not_mark_calibration_ready(self):
        calibration = PoseCalibration()
        self.assertEqual(calibration.capture({"elbow": float("nan")}), 0)
        self.assertFalse(calibration.calibrated)
