import csv
from datetime import datetime
from pathlib import Path
from typing import TextIO

from vision_robot_arm.vision.metrics import ANGLE_DEFINITIONS
from vision_robot_arm.core.pose_state import PoseState


class CsvPoseRecorder:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._path: Path | None = None

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self._output_dir / f"pose_recording_{timestamp}.csv"
        self._file = self._path.open("w", newline="", encoding="utf-8")
        fieldnames = self._build_fieldnames(landmark_names)
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()
        return self._path

    def stop(self) -> Path | None:
        stopped_path = self._path
        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None
        self._writer = None
        self._path = None
        return stopped_path

    def toggle(self, landmark_names: dict[int, str]) -> tuple[bool, Path | None]:
        if self.is_recording:
            return False, self.stop()
        return True, self.start(landmark_names)

    def write_state(self, state: PoseState, landmark_names: dict[int, str]) -> None:
        if self._writer is None or self._file is None:
            return

        row: dict[str, str | int | float | bool] = {
            "timestamp_ms": state.timestamp_ms,
            "calibrated": state.calibrated,
            "gestures": "|".join(state.gestures),
        }

        for name in ANGLE_DEFINITIONS:
            row[f"angle_{name}"] = _value_or_blank(state.angles.get(name))
            row[f"relative_{name}"] = _value_or_blank(state.relative_angles.get(name))

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

        self._writer.writerow(row)
        self._file.flush()

    def _build_fieldnames(self, landmark_names: dict[int, str]) -> list[str]:
        fieldnames = ["timestamp_ms", "calibrated", "gestures"]
        for name in ANGLE_DEFINITIONS:
            fieldnames.append(f"angle_{name}")
            fieldnames.append(f"relative_{name}")

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
                ]
            )
        return fieldnames


def _value_or_blank(value: float | None) -> float | str:
    if value is None:
        return ""
    return value
