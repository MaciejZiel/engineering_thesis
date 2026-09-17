import csv
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO
from uuid import uuid4

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.vision.arm_pose import ELEVATION_ANGLE_NAMES
from vision_robot_arm.vision.metrics import ANGLE_DEFINITIONS

RECORDED_ANGLES = (*ANGLE_DEFINITIONS, *ELEVATION_ANGLE_NAMES)
HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)


class CsvPoseRecorder:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._path: Path | None = None
        self.last_error: str | None = None
        self._last_flush = 0.0

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def path(self) -> Path | None:
        return self._path

    def start(self, landmark_names: dict[int, str]) -> Path:
        if self.is_recording and self._path is not None:
            return self._path

        self._output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._path = (
            self._output_dir / f"pose_recording_{timestamp}_{uuid4().hex[:8]}.csv"
        )
        self.last_error = None
        try:
            self._file = self._path.open("x", newline="", encoding="utf-8")
            fieldnames = self._build_fieldnames(landmark_names)
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()
            self._file.flush()
            self._last_flush = time.monotonic()
        except OSError:
            self._abort()
            raise
        return self._path

    def stop(self) -> Path | None:
        stopped_path = self._path
        file = self._file
        self._file, self._writer, self._path = None, None, None
        if file is not None:
            try:
                file.flush()
            finally:
                file.close()
        return stopped_path

    def toggle(self, landmark_names: dict[int, str]) -> tuple[bool, Path | None]:
        try:
            if self.is_recording:
                return False, self.stop()
            return True, self.start(landmark_names)
        except OSError as error:
            self.last_error = f"Recording failed: {error}"
            self._abort()
            return False, None

    def _abort(self) -> None:
        file = self._file
        self._file, self._writer = None, None
        if file is not None:
            try:
                file.close()
            except OSError:
                pass

    def write_state(self, state: PoseState, landmark_names: dict[int, str]) -> None:
        if self._writer is None or self._file is None:
            return

        row: dict[str, str | int | float | bool] = {
            "timestamp_ms": state.timestamp_ms,
            "calibrated": state.calibrated,
            "gestures": "|".join(state.gestures),
            "pose_world_frame": "body_relative_m" if state.world_landmarks else "",
            "hand_world_frame": "pose_wrist_anchored_m"
            if state.hand_world_landmarks
            else "",
            "control_frame": (
                "shoulder_center_m:x_right,y_forward,z_up"
                if state.body_frame is not None
                else ""
            ),
            "control_frame_source": (
                state.body_frame.source if state.body_frame is not None else ""
            ),
        }

        for name in RECORDED_ANGLES:
            row[f"angle_{name}"] = _value_or_blank(state.angles.get(name))
            row[f"relative_{name}"] = _value_or_blank(state.relative_angles.get(name))
            row[f"source_{name}"] = state.angle_sources.get(name, "")

        for index, landmark in enumerate(state.landmarks):
            name = landmark_names.get(index, str(index))
            row[f"{name}_x"] = landmark.x
            row[f"{name}_y"] = landmark.y
            row[f"{name}_z"] = landmark.z
            row[f"{name}_visibility"] = landmark.visibility

        if state.world_landmarks is not None:
            for index, landmark in enumerate(state.world_landmarks):
                name = landmark_names.get(index, str(index))
                row[f"{name}_world_x"] = landmark.x
                row[f"{name}_world_y"] = landmark.y
                row[f"{name}_world_z"] = landmark.z

        if state.body_landmarks is not None:
            for index, landmark in enumerate(state.body_landmarks):
                name = landmark_names.get(index, str(index))
                row[f"{name}_body_x"] = landmark.x
                row[f"{name}_body_y"] = landmark.y
                row[f"{name}_body_z"] = landmark.z

        for side in ("left", "right"):
            image_hand = state.hand_landmarks.get(side, ())
            world_hand = state.hand_world_landmarks.get(side, ())
            body_hand = state.hand_body_landmarks.get(side, ())
            for index, name in enumerate(HAND_LANDMARK_NAMES):
                prefix = f"{side}_hand_{name}"
                if index < len(image_hand):
                    point = image_hand[index]
                    row[f"{prefix}_x"] = point.x
                    row[f"{prefix}_y"] = point.y
                    row[f"{prefix}_z"] = point.z
                if index < len(world_hand):
                    point = world_hand[index]
                    row[f"{prefix}_world_x"] = point.x
                    row[f"{prefix}_world_y"] = point.y
                    row[f"{prefix}_world_z"] = point.z
                if index < len(body_hand):
                    point = body_hand[index]
                    row[f"{prefix}_body_x"] = point.x
                    row[f"{prefix}_body_y"] = point.y
                    row[f"{prefix}_body_z"] = point.z

        try:
            self._writer.writerow(row)
            now = time.monotonic()
            if now - self._last_flush >= 1.0:
                self._file.flush()
                self._last_flush = now
        except OSError as error:
            self.last_error = f"Recording failed: {error}"
            self._abort()

    def _build_fieldnames(self, landmark_names: dict[int, str]) -> list[str]:
        fieldnames = [
            "timestamp_ms",
            "calibrated",
            "gestures",
            "pose_world_frame",
            "hand_world_frame",
            "control_frame",
            "control_frame_source",
        ]
        for name in RECORDED_ANGLES:
            fieldnames.append(f"angle_{name}")
            fieldnames.append(f"relative_{name}")
            fieldnames.append(f"source_{name}")

        for index in sorted(landmark_names):
            name = landmark_names[index]
            fieldnames.extend(
                [
                    f"{name}_x",
                    f"{name}_y",
                    f"{name}_z",
                    f"{name}_visibility",
                    f"{name}_world_x",
                    f"{name}_world_y",
                    f"{name}_world_z",
                    f"{name}_body_x",
                    f"{name}_body_y",
                    f"{name}_body_z",
                ]
            )
        for side in ("left", "right"):
            for name in HAND_LANDMARK_NAMES:
                prefix = f"{side}_hand_{name}"
                fieldnames.extend(
                    f"{prefix}_{suffix}"
                    for suffix in (
                        "x", "y", "z", "world_x", "world_y", "world_z",
                        "body_x", "body_y", "body_z",
                    )
                )
        return fieldnames


def _value_or_blank(value: float | None) -> float | str:
    if value is None:
        return ""
    return value
