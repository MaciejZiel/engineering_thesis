from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    z: float
    visibility: float = 1.0

    @classmethod
    def from_landmark(cls, landmark: Any) -> "LandmarkPoint":
        return cls(
            x=float(landmark.x),
            y=float(landmark.y),
            z=float(landmark.z),
            visibility=float(getattr(landmark, "visibility", 1.0)),
        )


@dataclass(frozen=True)
class PoseState:
    timestamp_ms: int
    landmarks: list[LandmarkPoint]
    world_landmarks: list[LandmarkPoint] | None
    raw_angles: dict[str, float | None]
    angles: dict[str, float | None]
    relative_angles: dict[str, float | None]
    gestures: tuple[str, ...]
    calibrated: bool

    @property
    def display_angles(self) -> dict[str, float | None]:
        if self.calibrated:
            return self.relative_angles
        return self.angles
