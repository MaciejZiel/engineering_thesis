import math
from collections import deque
from pathlib import Path
from typing import Any

from vision_robot_arm.core.pose_state import LandmarkPoint, PoseState
from vision_robot_arm.vision.calibration import PoseCalibration
from vision_robot_arm.vision.body_tracking import (
    BodyFrameStabilizer,
    build_body_frame,
    transform_hands_to_body,
    transform_pose_to_body,
)
from vision_robot_arm.vision.gestures import detect_gestures
from vision_robot_arm.vision.metrics import calculate_angles
from vision_robot_arm.vision.smoothing import (
    MAX_WORLD_LANDMARK_SPEED_M_S,
    AngleSmoother,
    LandmarkSmoother,
)


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
        self._world_landmark_smoother = LandmarkSmoother(
            smoothing_alpha,
            min_visibility,
            max_speed=MAX_WORLD_LANDMARK_SPEED_M_S,
        )
        self._angle_smoother = AngleSmoother(smoothing_alpha)
        self._body_frame_stabilizer = BodyFrameStabilizer(smoothing_alpha)
        self._calibration = PoseCalibration()
        self._calibration_samples = deque()

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
        hand_landmarks: dict[str, list[Any]] | None = None,
        hand_world_landmarks: dict[str, list[Any]] | None = None,
        world_only: bool = False,
        extra_angle_sources: dict[str, str] | None = None,
    ) -> PoseState:
        raw_landmarks = [
            LandmarkPoint.from_landmark(landmark) for landmark in landmarks
        ]
        smoothed_landmarks = self._landmark_smoother.update(raw_landmarks, timestamp_ms)

        smoothed_world_landmarks = None
        if world_landmarks is not None:
            raw_world_landmarks = [
                LandmarkPoint.from_landmark(landmark) for landmark in world_landmarks
            ]
            smoothed_world_landmarks = self._world_landmark_smoother.update(
                raw_world_landmarks, timestamp_ms
            )

        raw_angles = calculate_angles(
            raw_landmarks,
            self._indices,
            self._min_visibility,
            aspect_ratio=aspect_ratio,
            world_landmarks=world_landmarks,
        )
        if world_only and world_landmarks is None:
            raw_angles = {name: None for name in raw_angles}
        # Smooth once in angle space. Filtering coordinates AND angles caused
        # extra latency and distorted joint geometry during movement.
        if hand_tracking_enabled:
            raw_angles.update(left_wrist=None, right_wrist=None)
        if extra_angles:
            raw_angles.update(
                {
                    name: value
                    for name, value in extra_angles.items()
                    if value is not None and math.isfinite(value)
                }
            )
        smoothed_angles = self._angle_smoother.update(raw_angles, timestamp_ms)
        if (
            self._calibration_samples
            and timestamp_ms <= self._calibration_samples[-1][0]
        ):
            self._calibration_samples.clear()
        self._calibration_samples.append((timestamp_ms, dict(raw_angles)))
        while (
            self._calibration_samples
            and timestamp_ms - self._calibration_samples[0][0] > 800
        ):
            self._calibration_samples.popleft()
        relative_angles = self._calibration.relative_angles(smoothed_angles)
        gestures = detect_gestures(
            smoothed_landmarks,
            smoothed_angles,
            self._indices,
            self._min_visibility,
        ) + tuple(extra_gestures)
        angle_sources = {
            name: (
                "hand_world_3d"
                if name.endswith("_wrist") and hand_tracking_enabled
                else "pose_world_3d"
                if world_landmarks is not None
                else "image_2d"
            )
            for name, value in raw_angles.items()
            if value is not None
        }
        angle_sources.update(
            {
                name: source
                for name, source in (extra_angle_sources or {}).items()
                if raw_angles.get(name) is not None
            }
        )

        frozen_hands = _freeze_hands(hand_landmarks)
        frozen_world_hands = _freeze_hands(hand_world_landmarks)
        body_frame = self._body_frame_stabilizer.update(
            build_body_frame(
                smoothed_landmarks,
                smoothed_world_landmarks,
                self._indices,
                self._min_visibility,
            ),
            timestamp_ms,
        )
        body_landmarks = (
            transform_pose_to_body(
                body_frame, smoothed_landmarks, smoothed_world_landmarks
            )
            if body_frame is not None and smoothed_world_landmarks is not None
            else None
        )
        hand_body_landmarks = (
            transform_hands_to_body(body_frame, frozen_world_hands)
            if body_frame is not None
            else {}
        )
        body_points = _named_body_points(body_landmarks, self._indices)

        return PoseState(
            timestamp_ms=timestamp_ms,
            landmarks=smoothed_landmarks,
            world_landmarks=smoothed_world_landmarks,
            raw_angles=raw_angles,
            angles=smoothed_angles,
            relative_angles=relative_angles,
            gestures=gestures,
            calibrated=self._calibration.calibrated,
            hand_landmarks=frozen_hands,
            hand_world_landmarks=frozen_world_hands,
            angle_sources=angle_sources,
            body_frame=body_frame,
            body_landmarks=body_landmarks,
            hand_body_landmarks=hand_body_landmarks,
            body_points=body_points,
        )

    def capture_calibration(self, state: PoseState, required=()) -> int:
        return self._calibration.capture_stable(
            list(self._calibration_samples), required
        )

    def reset_calibration(self) -> None:
        self._calibration.reset()

    def save_calibration(self, path: Path) -> None:
        self._calibration.save(path)

    def load_calibration(self, path: Path) -> None:
        self._calibration.load(path)

    def reset_tracking(self) -> None:
        self._calibration_samples.clear()
        self._landmark_smoother.reset()
        self._world_landmark_smoother.reset()
        self._angle_smoother.reset()
        self._body_frame_stabilizer.reset()


def _freeze_hands(
    hands: dict[str, list[Any]] | None,
) -> dict[str, tuple[LandmarkPoint, ...]]:
    return {
        side: tuple(
            point
            if isinstance(point, LandmarkPoint)
            else LandmarkPoint.from_landmark(point)
            for point in landmarks
        )
        for side, landmarks in (hands or {}).items()
    }


def _named_body_points(
    body_landmarks: list[LandmarkPoint] | None, indices: dict[str, int]
) -> dict[str, LandmarkPoint]:
    if body_landmarks is None:
        return {}
    points = {}
    for side in ("left", "right"):
        for joint in ("shoulder", "elbow", "wrist"):
            index = indices.get(f"{side.upper()}_{joint.upper()}")
            if index is not None and 0 <= index < len(body_landmarks):
                points[f"{side}_{joint}"] = body_landmarks[index]
    return points
