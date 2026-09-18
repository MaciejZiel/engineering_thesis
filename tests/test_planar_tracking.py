import unittest
from dataclasses import replace

from vision_robot_arm.core.pose_state import LandmarkPoint
from vision_robot_arm.vision.planar_tracking import PlanarArmTracker
from vision_robot_arm.vision.state_builder import PoseStateBuilder


class PlanarTrackingTests(unittest.TestCase):
    indices = {"RIGHT_SHOULDER": 0, "RIGHT_ELBOW": 1, "RIGHT_WRIST": 2}

    def measure(self, points, hands=None, aspect=1, mirrored=False):
        return PlanarArmTracker().measure(points, hands or {}, self.indices, aspect, 0.55, mirrored)

    def test_depth_does_not_affect_any_arm_angle(self):
        points = [LandmarkPoint(.2, .5, 0), LandmarkPoint(.5, .5, 0), LandmarkPoint(.5, .2, 0)]
        hands = {"right": [LandmarkPoint(.5, .2, 0)] * 9 + [LandmarkPoint(.6, .2, 0)]}
        before = self.measure(points, hands)
        after = self.measure([replace(p, z=100*(i+1)) for i,p in enumerate(points)], {"right": [replace(p, z=-100) for p in hands["right"]]})
        self.assertEqual(before, after)
        self.assertEqual(before, {"right_shoulder_elevation": 0, "right_elbow": 90, "right_wrist": -90})

    def test_mirror_reverses_bend_and_aspect_ratio_corrects_angles(self):
        points = [LandmarkPoint(.1, .5, 0), LandmarkPoint(.3, .3, 0), LandmarkPoint(.5, .3, 0)]
        measured = self.measure(points, aspect=2)
        self.assertAlmostEqual(measured["right_shoulder_elevation"], 26.565051177)
        self.assertAlmostEqual(self.measure(points, aspect=2, mirrored=True)["right_elbow"], -measured["right_elbow"])

    def test_missing_wrist_does_not_fall_back_to_unsigned_or_world_angles(self):
        points = [LandmarkPoint(.1, .5, 0), LandmarkPoint(.3, .3, 0), LandmarkPoint(.5, .3, 0, 0)]
        angles = self.measure(points)
        indices = dict(self.indices, LEFT_SHOULDER=0, LEFT_ELBOW=1, LEFT_WRIST=2)
        state = PoseStateBuilder(indices, .55, 1).build(1, points, points, extra_angles=angles, planar=True)
        self.assertIsNone(state.angles["right_elbow"])
        self.assertIsNone(state.angles["right_wrist"])
        self.assertIsNone(state.world_landmarks)

    def test_horizontal_left_crossing_has_no_360_degree_jump(self):
        tracker = PlanarArmTracker()
        def sample(y):
            return tracker.measure([LandmarkPoint(.5,.5,0), LandmarkPoint(.1,y,0), LandmarkPoint(.1,.2,0)], {}, self.indices, 1, .55, False)
        first = sample(.49)["right_shoulder_elevation"]
        second = sample(.51)["right_shoulder_elevation"]
        self.assertLess(abs(second-first), 3)
