import json
import math
import random
import statistics
import tempfile
import unittest
from pathlib import Path

from vision_robot_arm.core.pose_state import LandmarkPoint
from vision_robot_arm.vision.skeleton import (
    BONES,
    STAGES,
    Skeleton,
    SkeletonCalibrator,
    SkeletonError,
    load_skeleton,
    measure_bones,
    save_skeleton,
)

INDICES = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
}

UPPER_ARM = 0.30
FOREARM = 0.26


def body(
    left_elbow, left_wrist, right_elbow, right_wrist, shoulders=0.19, visibility=1.0
):
    points = [LandmarkPoint(0.0, 0.0, 0.0, visibility=0.0) for _ in range(17)]
    points[11] = LandmarkPoint(-shoulders, 0.0, 0.0, visibility=visibility)
    points[12] = LandmarkPoint(shoulders, 0.0, 0.0, visibility=visibility)
    points[13] = LandmarkPoint(*left_elbow, visibility=visibility)
    points[15] = LandmarkPoint(*left_wrist, visibility=visibility)
    points[14] = LandmarkPoint(*right_elbow, visibility=visibility)
    points[16] = LandmarkPoint(*right_wrist, visibility=visibility)
    return points


def t_pose(noise: float = 0.0, seed: int | None = None) -> list[LandmarkPoint]:
    """Arms straight out sideways, with optional noise on the depth axis."""
    rng = random.Random(seed)

    def z() -> float:
        return rng.gauss(0.0, noise) if noise else 0.0

    return body(
        left_elbow=(-0.19 - UPPER_ARM, 0.0, z()),
        left_wrist=(-0.19 - UPPER_ARM - FOREARM, 0.0, z()),
        right_elbow=(0.19 + UPPER_ARM, 0.0, z()),
        right_wrist=(0.19 + UPPER_ARM + FOREARM, 0.0, z()),
    )


def bent_pose() -> list[LandmarkPoint]:
    return body(
        left_elbow=(-0.19 - 0.10, 0.28, 0.0),
        left_wrist=(-0.19 - 0.12, 0.28 - 0.25, 0.0),
        right_elbow=(0.19 + 0.10, 0.28, 0.0),
        right_wrist=(0.19 + 0.12, 0.28 - 0.25, 0.0),
    )


def arms_up_pose() -> list[LandmarkPoint]:
    return body(
        left_elbow=(-0.22, -0.28, 0.0),
        left_wrist=(-0.24, -0.54, 0.0),
        right_elbow=(0.22, -0.28, 0.0),
        right_wrist=(0.24, -0.54, 0.0),
    )


def arms_down_pose() -> list[LandmarkPoint]:
    return body(
        left_elbow=(-0.21, 0.30, 0.0),
        left_wrist=(-0.22, 0.56, 0.0),
        right_elbow=(0.21, 0.30, 0.0),
        right_wrist=(0.22, 0.56, 0.0),
    )


POSE_FOR_STAGE = {
    "t_pose": t_pose,
    "bent_arms": bent_pose,
    "arms_up": arms_up_pose,
    "arms_down": arms_down_pose,
}


def elbow_angle(points: list[LandmarkPoint]) -> float:
    shoulder, elbow, wrist = points[11], points[13], points[15]
    first = (shoulder.x - elbow.x, shoulder.y - elbow.y, shoulder.z - elbow.z)
    second = (wrist.x - elbow.x, wrist.y - elbow.y, wrist.z - elbow.z)
    dot = sum(a * b for a, b in zip(first, second))
    sizes = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / sizes))))


class MeasureTests(unittest.TestCase):
    def test_a_t_pose_measures_both_arm_bones(self) -> None:
        measured = measure_bones(t_pose(), INDICES)

        self.assertAlmostEqual(measured["left_upper_arm"], UPPER_ARM, places=6)
        self.assertAlmostEqual(measured["right_forearm"], FOREARM, places=6)

    def test_a_bone_pointing_at_the_camera_is_not_measured(self) -> None:
        """Along the depth axis the length is almost entirely guesswork."""
        points = body(
            left_elbow=(-0.19, 0.0, 0.30),
            left_wrist=(-0.19, 0.0, 0.56),
            right_elbow=(0.19, 0.0, 0.0),
            right_wrist=(0.19 + 0.26, 0.0, 0.0),
        )

        measured = measure_bones(points, INDICES)

        self.assertNotIn("left_upper_arm", measured)
        self.assertIn("right_forearm", measured)

    def test_invisible_landmarks_are_not_measured(self) -> None:
        self.assertEqual(measure_bones(t_pose(), INDICES, min_visibility=1.1), {})

    def test_an_implausible_bone_is_rejected(self) -> None:
        points = body(
            left_elbow=(-3.0, 0.0, 0.0),
            left_wrist=(-3.3, 0.0, 0.0),
            right_elbow=(0.19, 0.0, 0.0),
            right_wrist=(0.19, 0.0, 0.0),
        )

        measured = measure_bones(points, INDICES)

        self.assertNotIn("left_upper_arm", measured)
        self.assertNotIn("right_forearm", measured)


class ConstrainTests(unittest.TestCase):
    def skeleton(self) -> Skeleton:
        return Skeleton(
            {
                "left_upper_arm": UPPER_ARM,
                "left_forearm": FOREARM,
                "right_upper_arm": UPPER_ARM,
                "right_forearm": FOREARM,
            }
        )

    def test_a_stretched_bone_is_pulled_back_to_its_measured_length(self) -> None:
        points = t_pose()
        points[13] = LandmarkPoint(-0.19 - UPPER_ARM, 0.0, 0.22)

        corrected = self.skeleton().constrain(points, INDICES)

        length = math.dist(
            (corrected[11].x, corrected[11].y, corrected[11].z),
            (corrected[13].x, corrected[13].y, corrected[13].z),
        )
        self.assertAlmostEqual(length, UPPER_ARM, places=6)

    def test_the_well_seen_coordinates_are_left_alone(self) -> None:
        points = t_pose()
        points[13] = LandmarkPoint(-0.44, 0.03, 0.19)

        corrected = self.skeleton().constrain(points, INDICES)

        self.assertAlmostEqual(corrected[13].x, -0.44)
        self.assertAlmostEqual(corrected[13].y, 0.03)
        self.assertNotAlmostEqual(corrected[13].z, 0.19)

    def test_a_limb_too_long_across_the_image_is_scaled_instead(self) -> None:
        """Depth cannot explain it, so do not pretend that it can."""
        points = t_pose()
        points[13] = LandmarkPoint(-0.19 - 0.9, 0.0, 0.0)

        corrected = self.skeleton().constrain(points, INDICES)

        length = math.dist(
            (corrected[11].x, corrected[11].y, corrected[11].z),
            (corrected[13].x, corrected[13].y, corrected[13].z),
        )
        self.assertAlmostEqual(length, UPPER_ARM, places=6)

    def test_bones_it_never_measured_are_left_untouched(self) -> None:
        partial = Skeleton({"left_upper_arm": UPPER_ARM})
        points = t_pose()
        points[15] = LandmarkPoint(-0.9, 0.0, 0.4)

        corrected = partial.constrain(points, INDICES)

        self.assertEqual(corrected[15], points[15])

    def test_a_missing_landmark_does_not_break_the_chain(self) -> None:
        self.skeleton().constrain([], INDICES)

    def test_holding_the_bones_steadies_the_elbow_angle(self) -> None:
        """The payoff: depth noise is what makes a still arm's angle wander."""
        skeleton = self.skeleton()
        truth = elbow_angle(t_pose())
        raw_error, held_error = [], []
        for seed in range(200):
            noisy = t_pose(noise=0.035, seed=seed)
            raw_error.append(abs(elbow_angle(noisy) - truth))
            held_error.append(abs(elbow_angle(skeleton.constrain(noisy, INDICES)) - truth))

        self.assertLess(statistics.mean(held_error), statistics.mean(raw_error) / 2)


class CalibratorTests(unittest.TestCase):
    def run_stage(
        self, calibrator: SkeletonCalibrator, pose, start_ms: int, frames: int = 30
    ) -> int:
        timestamp = start_ms
        for _ in range(frames):
            timestamp += 33
            calibrator.update(timestamp, pose(), INDICES)
        return timestamp

    def calibrate(self) -> SkeletonCalibrator:
        calibrator = SkeletonCalibrator()
        timestamp = 0
        for stage in STAGES:
            timestamp = self.run_stage(
                calibrator, POSE_FOR_STAGE[stage.name], timestamp
            )
        return calibrator

    def test_the_routine_walks_every_stage_and_ends_with_a_skeleton(self) -> None:
        calibrator = self.calibrate()

        self.assertTrue(calibrator.finished)
        self.assertAlmostEqual(
            calibrator.skeleton.length("left_upper_arm"), UPPER_ARM, places=2
        )
        self.assertAlmostEqual(
            calibrator.skeleton.length("right_forearm"), FOREARM, places=2
        )

    def test_it_reports_which_pose_to_strike(self) -> None:
        calibrator = SkeletonCalibrator()

        progress = calibrator.update(33, arms_down_pose(), INDICES)

        self.assertEqual(progress.stage, "t_pose")
        self.assertIn("T-pose", progress.prompt)
        self.assertEqual(progress.samples, 0)
        self.assertFalse(progress.finished)

    def test_a_pose_passed_through_on_the_way_somewhere_else_does_not_count(
        self,
    ) -> None:
        """Otherwise a limb swinging past the target pose calibrates the person."""
        calibrator = SkeletonCalibrator()
        timestamp = 0
        for _ in range(20):
            timestamp += 33
            calibrator.update(timestamp, t_pose(), INDICES)
            timestamp += 33
            progress = calibrator.update(timestamp, arms_down_pose(), INDICES)

        self.assertEqual(progress.samples, 0)
        self.assertFalse(calibrator.finished)

    def test_a_stage_needs_time_and_not_only_frames(self) -> None:
        calibrator = SkeletonCalibrator()
        timestamp = 0
        for _ in range(40):
            timestamp += 1
            progress = calibrator.update(timestamp, t_pose(), INDICES)

        self.assertEqual(progress.stage, "t_pose")

    def test_losing_the_person_restarts_the_stage(self) -> None:
        calibrator = SkeletonCalibrator()
        timestamp = 0
        for _ in range(8):
            timestamp += 33
            calibrator.update(timestamp, t_pose(), INDICES)

        timestamp += 33
        progress = calibrator.update(timestamp, None, INDICES)

        self.assertEqual(progress.samples, 0)

    def test_a_reset_starts_the_whole_routine_again(self) -> None:
        calibrator = self.calibrate()
        calibrator.reset()

        self.assertFalse(calibrator.finished)
        self.assertIsNone(calibrator.skeleton)
        self.assertEqual(calibrator.update(33, None, INDICES).stage, "t_pose")

    def test_the_status_line_names_the_stage_and_the_progress(self) -> None:
        calibrator = SkeletonCalibrator()

        line = calibrator.update(33, t_pose(), INDICES).status_line()

        self.assertIn("t_pose", line)
        self.assertIn("/12", line)


class ProfileTests(unittest.TestCase):
    def test_a_saved_skeleton_loads_back_unchanged(self) -> None:
        skeleton = Skeleton({"left_upper_arm": 0.31, "right_forearm": 0.25})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles" / "skeleton.json"
            save_skeleton(skeleton, path)

            self.assertEqual(load_skeleton(path).bones, skeleton.bones)

    def test_an_unknown_bone_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skeleton.json"
            path.write_text(
                json.dumps({"version": 1, "bones": {"tail": 0.3}}), encoding="utf-8"
            )

            with self.assertRaises(SkeletonError):
                load_skeleton(path)

    def test_an_impossible_length_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skeleton.json"
            path.write_text(
                json.dumps({"version": 1, "bones": {"left_forearm": 9.0}}),
                encoding="utf-8",
            )

            with self.assertRaises(SkeletonError):
                load_skeleton(path)

    def test_a_missing_file_is_reported_as_a_profile_problem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SkeletonError):
                load_skeleton(Path(directory) / "absent.json")

    def test_every_bone_name_has_two_landmarks(self) -> None:
        for name, endpoints in BONES.items():
            with self.subTest(bone=name):
                self.assertEqual(len(endpoints), 2)


if __name__ == "__main__":
    unittest.main()
