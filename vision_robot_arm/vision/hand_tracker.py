from dataclasses import dataclass
from typing import Any

from vision_robot_arm.core.config import AppConfig
from vision_robot_arm.core.runtime import RuntimeDeps

HandLandmarks = list[Any]


@dataclass(frozen=True)
class HandDetection:
    landmarks: list[HandLandmarks]
    world_landmarks: list[HandLandmarks]


class HandTracker:
    def __init__(self, deps: RuntimeDeps, config: AppConfig) -> None:
        self._deps = deps
        options = deps.vision.HandLandmarkerOptions(
            base_options=deps.base_options(
                model_asset_path=str(config.hand_model_path.resolve())
            ),
            running_mode=deps.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = deps.vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame: Any, timestamp_ms: int) -> list[HandLandmarks]:
        return self.detect_frame(rgb_frame, timestamp_ms).landmarks

    def detect_frame(self, rgb_frame: Any, timestamp_ms: int) -> HandDetection:
        image = self._deps.mp.Image(
            image_format=self._deps.mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return HandDetection(
            landmarks=[list(hand) for hand in result.hand_landmarks],
            world_landmarks=[list(hand) for hand in result.hand_world_landmarks],
        )

    def close(self) -> None:
        self._landmarker.close()
