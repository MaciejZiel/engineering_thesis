import math

from vision_robot_arm.core.pose_state import LandmarkPoint


class LowPassValueFilter:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._previous: float | None = None

    def update(self, value: float, alpha: float | None = None) -> float:
        if self._previous is None:
            self._previous = value
            return value

        effective_alpha = self._alpha if alpha is None else alpha
        smoothed = effective_alpha * value + (1.0 - effective_alpha) * self._previous
        self._previous = smoothed
        return smoothed

    def reset(self) -> None:
        self._previous = None


class LandmarkSmoother:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._previous: list[LandmarkPoint] | None = None

    def update(self, landmarks: list[LandmarkPoint]) -> list[LandmarkPoint]:
        if self._previous is None or len(self._previous) != len(landmarks):
            self._previous = landmarks
            return landmarks

        smoothed = [
            LandmarkPoint(
                x=self._smooth(current.x, previous.x),
                y=self._smooth(current.y, previous.y),
                z=self._smooth(current.z, previous.z),
                visibility=current.visibility,
            ) if self._valid(current) and self._valid(previous) else current
            for current, previous in zip(landmarks, self._previous)
        ]
        self._previous = smoothed
        return smoothed

    def reset(self) -> None:
        self._previous = None

    def _smooth(self, current: float, previous: float) -> float:
        return self._alpha * current + (1.0 - self._alpha) * previous

    @staticmethod
    def _valid(point: LandmarkPoint) -> bool:
        return point.visibility >= 0.55 and all(math.isfinite(v) for v in (point.x, point.y, point.z))


class AngleSmoother:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._filters: dict[str, LowPassValueFilter] = {}
        self._last_timestamp_ms: int | None = None

    def update(self, angles: dict[str, float | None], timestamp_ms: int | None = None) -> dict[str, float | None]:
        alpha = self._alpha
        if timestamp_ms is not None and self._last_timestamp_ms is not None:
            elapsed = timestamp_ms - self._last_timestamp_ms
            if elapsed <= 0 or elapsed > 500:
                self.reset()
            else:
                # Alpha describes smoothing at 30 Hz; low-FPS cameras must not lag seconds behind.
                alpha = 1 - (1-self._alpha)**(elapsed / (1000/30))
        self._last_timestamp_ms = timestamp_ms
        smoothed: dict[str, float | None] = {}
        for name in self._filters.keys() - angles.keys():
            self._filters[name].reset()
        for name, value in angles.items():
            if value is None or not math.isfinite(value):
                smoothed[name] = None
                if name in self._filters:
                    self._filters[name].reset()
                continue

            value_filter = self._filters.setdefault(name, LowPassValueFilter(self._alpha))
            smoothed[name] = value_filter.update(value, alpha)
        return smoothed

    def reset(self) -> None:
        self._last_timestamp_ms = None
        for value_filter in self._filters.values():
            value_filter.reset()
