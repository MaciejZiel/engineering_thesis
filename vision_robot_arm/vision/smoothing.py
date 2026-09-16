from vision_robot_arm.core.pose_state import LandmarkPoint


class LowPassValueFilter:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._previous: float | None = None

    def update(self, value: float) -> float:
        if self._previous is None:
            self._previous = value
            return value

        smoothed = self._alpha * value + (1.0 - self._alpha) * self._previous
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
            )
            for current, previous in zip(landmarks, self._previous)
        ]
        self._previous = smoothed
        return smoothed

    def reset(self) -> None:
        self._previous = None

    def _smooth(self, current: float, previous: float) -> float:
        return self._alpha * current + (1.0 - self._alpha) * previous


class AngleSmoother:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._filters: dict[str, LowPassValueFilter] = {}

    def update(self, angles: dict[str, float | None]) -> dict[str, float | None]:
        smoothed: dict[str, float | None] = {}
        for name, value in angles.items():
            if value is None:
                smoothed[name] = None
                continue

            value_filter = self._filters.setdefault(name, LowPassValueFilter(self._alpha))
            smoothed[name] = value_filter.update(value)
        return smoothed

    def reset(self) -> None:
        for value_filter in self._filters.values():
            value_filter.reset()
