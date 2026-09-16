import math
from typing import Any

from vision_robot_arm.vision.calibration import PoseCalibration
from vision_robot_arm.vision.gestures import detect_gestures
from vision_robot_arm.vision.metrics import calculate_angles
from vision_robot_arm.core.pose_state import LandmarkPoint, PoseState
from vision_robot_arm.vision.smoothing import AngleSmoother, LandmarkSmoother


class PoseStateBuilder:
    def __init__(
        self,
        indices: dict[str, int],
        min_visibility: float,
        smoothing_alpha: float,
    ) -> None:
        self._indices = indices
        self._min_visibility = min_visibility
        self._landmark_smoother = LandmarkSmoother(smoothing_alpha, min_visibility)
        self._world_landmark_smoother = LandmarkSmoother(smoothing_alpha, min_visibility)
        self._angle_smoother = AngleSmoother(smoothing_alpha)
        self._calibration = PoseCalibration()

    @property
    def calibrated(self) -> bool:
        return self._calibration.calibrated

    def build(
        self,
        timestamp_ms: int,
        landmarks: list[Any],
        world_landmarks: list[Any] | None,
        extra_gestures: tuple[str, ...] = (),
        extra_angles: dict[str, float] | None = None,
        *,
        aspect_ratio: float = 1.0,
        hand_tracking_enabled: bool = False,
    ) -> PoseState:
        raw_landmarks = [LandmarkPoint.from_landmark(landmark) for landmark in landmarks]
        smoothed_landmarks = self._landmark_smoother.update(raw_landmarks)

        smoothed_world_landmarks = None
        if world_landmarks is not None:
            raw_world_landmarks = [
                LandmarkPoint.from_landmark(landmark)
                for landmark in world_landmarks
            ]
            smoothed_world_landmarks = self._world_landmark_smoother.update(
                raw_world_landmarks
            )

        raw_angles = calculate_angles(
            raw_landmarks,
            self._indices,
            self._min_visibility,
            aspect_ratio=aspect_ratio,
            world_landmarks=world_landmarks,
        )
        # Smooth once in angle space. Filtering coordinates AND angles caused
        # extra latency and distorted joint geometry during movement.
        if hand_tracking_enabled:
            raw_angles.update(left_wrist=None, right_wrist=None)
        if extra_angles:
            raw_angles.update({name: value for name, value in extra_angles.items()
                               if value is not None and math.isfinite(value)})
        smoothed_angles = self._angle_smoother.update(raw_angles, timestamp_ms)
        relative_angles = self._calibration.relative_angles(smoothed_angles)
        gestures = detect_gestures(
            smoothed_landmarks,
            smoothed_angles,
            self._indices,
            self._min_visibility,
        ) + tuple(extra_gestures)

        return PoseState(
            timestamp_ms=timestamp_ms,
            landmarks=smoothed_landmarks,
            world_landmarks=smoothed_world_landmarks,
            raw_angles=raw_angles,
            angles=smoothed_angles,
            relative_angles=relative_angles,
            gestures=gestures,
            calibrated=self._calibration.calibrated,
        )

    def capture_calibration(self, state: PoseState) -> int:
        return self._calibration.capture(state.angles)

    def reset_tracking(self) -> None:
        self._landmark_smoother.reset()
        self._world_landmark_smoother.reset()
        self._angle_smoother.reset()
