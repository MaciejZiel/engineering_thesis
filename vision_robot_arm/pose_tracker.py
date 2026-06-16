from dataclasses import dataclass
from typing import Any

from vision_robot_arm.config import AppConfig
from vision_robot_arm.runtime import RuntimeDeps


@dataclass(frozen=True)
class PoseDetection:
    landmarks: list[Any] | None
    world_landmarks: list[Any] | None

    @property
    def has_pose(self) -> bool:
        return self.landmarks is not None


class PoseTracker:
    def __init__(self, deps: RuntimeDeps, config: AppConfig) -> None:
        self._deps = deps
        options = deps.vision.PoseLandmarkerOptions(
            base_options=deps.base_options(model_asset_path=str(config.model_path.resolve())),
            running_mode=deps.vision.RunningMode.VIDEO,
            num_poses=config.num_poses,
            min_pose_detection_confidence=config.min_detection_confidence,
            min_pose_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = deps.vision.PoseLandmarker.create_from_options(options)

    def detect(self, rgb_frame: Any, timestamp_ms: int) -> PoseDetection:
        image = self._deps.mp.Image(
            image_format=self._deps.mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return PoseDetection(landmarks=None, world_landmarks=None)

        world_landmarks = (
            result.pose_world_landmarks[0]
            if result.pose_world_landmarks
            else None
        )
        return PoseDetection(
            landmarks=result.pose_landmarks[0],
            world_landmarks=world_landmarks,
        )

    def close(self) -> None:
        self._landmarker.close()
