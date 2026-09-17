import json
import math
import os
import statistics
import tempfile
from pathlib import Path


class PoseCalibration:
    def __init__(self) -> None:
        self._neutral_angles: dict[str, float] | None = None

    @property
    def calibrated(self) -> bool:
        return self._neutral_angles is not None

    def capture(self, angles: dict[str, float | None]) -> int:
        neutral = {
            name: value
            for name, value in angles.items()
            if value is not None and math.isfinite(value)
        }
        if not neutral:
            return 0
        self._neutral_angles = neutral
        return len(neutral)

    def capture_stable(self, samples, required=()) -> int:
        if len(samples) < 5 or samples[-1][0] - samples[0][0] < 600:
            return 0
        common = set.intersection(*(set(angles) for _, angles in samples))
        neutral = {}
        for name in common:
            values = [angles[name] for _, angles in samples]
            if any(value is None or not math.isfinite(value) for value in values):
                continue
            if max(values) - min(values) <= 3.0:
                neutral[name] = statistics.median(values)
        if not set(required).issubset(neutral):
            return 0
        return self.capture(neutral)

    def reset(self) -> None:
        self._neutral_angles = None

    def save(self, path: Path) -> None:
        if not self.calibrated:
            raise ValueError("Capture a stable calibration before saving.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             delete=False) as file:
                temporary = Path(file.name)
                json.dump({"version": 1, "neutral_angles": self._neutral_angles}, file, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported calibration profile.")
        angles = data.get("neutral_angles")
        if not isinstance(angles, dict) or not angles or any(
            not isinstance(name, str) or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not 0 <= value <= 360 for name, value in angles.items()
        ):
            raise ValueError("Invalid calibration angles.")
        self._neutral_angles = dict(angles)

    def relative_angles(
        self,
        angles: dict[str, float | None],
    ) -> dict[str, float | None]:
        if self._neutral_angles is None:
            return {name: None for name in angles}

        relative: dict[str, float | None] = {}
        for name, value in angles.items():
            neutral_value = self._neutral_angles.get(name)
            if value is None or neutral_value is None:
                relative[name] = None
            else:
                relative[name] = value - neutral_value
        return relative
