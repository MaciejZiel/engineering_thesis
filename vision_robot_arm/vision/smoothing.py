import math

from vision_robot_arm.core.pose_state import LandmarkPoint

# Landmark noise at rest, in normalized image units and in degrees of joint angle.
LANDMARK_NOISE = 0.004
ANGLE_NOISE_DEG = 2.0
# Below this joint speed a reading is mostly noise; a real arm sweeps far faster.
MOTION_FLOOR_DEG_S = 100.0
# How much of the configured response survives when nothing is moving.
MIN_RESPONSE = 0.10
# Velocity has to be averaged over several frames, or it is just noise again.
VELOCITY_ALPHA = 0.15


def taper(alpha: float, change: float, noise_floor: float) -> float:
    """Fade the filter towards a hold for changes that are indistinguishable from noise."""
    if noise_floor <= 0.0 or change >= noise_floor:
        return alpha
    return alpha * (change / noise_floor) ** 2


class LowPassValueFilter:
    """Low pass whose response follows how fast the value is actually moving.

    A single frame-to-frame difference cannot tell 1.5 deg of landmark noise from a real
    arm at 60 deg/s, because at 30 fps both are about two degrees. Velocity averaged over
    several frames can: noise cancels itself, motion does not. When the estimate says the
    joint is still, the filter holds; when it says the arm is sweeping, the full
    configured response is used, so nothing is added to the lag that matters.
    """

    def __init__(self, alpha: float, motion_floor: float = 0.0) -> None:
        self._alpha = alpha
        self._motion_floor = motion_floor
        self._previous: float | None = None
        self._velocity: float | None = None

    def update(self, value: float, alpha: float | None = None, elapsed_s: float | None = None) -> float:
        if self._previous is None:
            self._previous = value
            return value

        effective_alpha = self._alpha if alpha is None else alpha
        effective_alpha *= self._response(value, elapsed_s)
        smoothed = effective_alpha * value + (1.0 - effective_alpha) * self._previous
        self._previous = smoothed
        return smoothed

    def _response(self, value: float, elapsed_s: float | None) -> float:
        if self._motion_floor <= 0.0 or not elapsed_s or elapsed_s <= 0.0:
            return 1.0
        velocity = (value - self._previous) / elapsed_s
        if self._velocity is None:
            self._velocity = velocity
        else:
            self._velocity = VELOCITY_ALPHA * velocity + (1.0 - VELOCITY_ALPHA) * self._velocity
        return max(MIN_RESPONSE, min(1.0, abs(self._velocity) / self._motion_floor))

    def reset(self) -> None:
        self._previous = None
        self._velocity = None


class LandmarkSmoother:
    def __init__(
        self,
        alpha: float,
        min_visibility: float = 0.55,
        noise_floor: float = LANDMARK_NOISE,
    ) -> None:
        self._alpha = alpha
        self._min_visibility = min_visibility
        self._noise_floor = noise_floor
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
        alpha = taper(self._alpha, abs(current - previous), self._noise_floor)
        return alpha * current + (1.0 - alpha) * previous

    def _valid(self, point: LandmarkPoint) -> bool:
        return point.visibility >= self._min_visibility and all(
            math.isfinite(value) for value in (point.x, point.y, point.z)
        )


class AngleSmoother:
    def __init__(self, alpha: float, motion_floor_deg_s: float = MOTION_FLOOR_DEG_S) -> None:
        self._alpha = alpha
        self._motion_floor_deg_s = motion_floor_deg_s
        self._filters: dict[str, LowPassValueFilter] = {}
        self._last_timestamp_ms: int | None = None

    def update(self, angles: dict[str, float | None], timestamp_ms: int | None = None) -> dict[str, float | None]:
        alpha = self._alpha
        elapsed_s: float | None = None
        if timestamp_ms is not None and self._last_timestamp_ms is not None:
            elapsed = timestamp_ms - self._last_timestamp_ms
            if elapsed <= 0 or elapsed > 500:
                self.reset()
            else:
                # Alpha describes smoothing at 30 Hz; low-FPS cameras must not lag seconds behind.
                alpha = 1 - (1-self._alpha)**(elapsed / (1000/30))
                elapsed_s = elapsed / 1000.0
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

            value_filter = self._filters.setdefault(
                name, LowPassValueFilter(self._alpha, self._motion_floor_deg_s)
            )
            smoothed[name] = value_filter.update(value, alpha, elapsed_s)
        return smoothed

    def reset(self) -> None:
        self._last_timestamp_ms = None
        for value_filter in self._filters.values():
            value_filter.reset()
