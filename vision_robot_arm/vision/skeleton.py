"""A per-person skeleton: measure the bones once, then hold them fixed every frame.

MediaPipe estimates the direction of a limb across the image well and its depth
badly. The measured length of an upper arm therefore swings from frame to frame,
and every angle derived from it swings with it. Measuring the bones once, while
the person holds poses where the limbs lie across the camera rather than along
it, gives lengths worth trusting; re-solving only the depth of each joint against
those lengths keeps the well-seen part of the estimate and replaces the rest.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from vision_robot_arm.core.pose_state import LandmarkPoint

# name -> (parent landmark, child landmark)
BONES: dict[str, tuple[str, str]] = {
    "shoulders": ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    "left_upper_arm": ("LEFT_SHOULDER", "LEFT_ELBOW"),
    "left_forearm": ("LEFT_ELBOW", "LEFT_WRIST"),
    "right_upper_arm": ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    "right_forearm": ("RIGHT_ELBOW", "RIGHT_WRIST"),
}

# Solved outward from the shoulder, so a corrected elbow carries the wrist with it.
CHAINS: tuple[tuple[str, ...], ...] = (
    ("left_upper_arm", "left_forearm"),
    ("right_upper_arm", "right_forearm"),
)

ARM_BONES = ("left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm")

# A bone pointing at the camera is measured almost entirely in the noisy axis.
MAX_DEPTH_FRACTION = 0.6
MIN_BONE_M = 0.05
MAX_BONE_M = 1.2
MIN_VISIBILITY = 0.6
# Enough samples to make a median mean something, over enough time to exclude one
# lucky instant.
SAMPLES_PER_STAGE = 12
STAGE_HOLD_MS = 700


class SkeletonError(ValueError):
    """The stored profile cannot be used."""


@dataclass(frozen=True)
class Skeleton:
    """Bone lengths in metres, measured once for one person."""

    bones: dict[str, float]

    def length(self, bone: str) -> float | None:
        return self.bones.get(bone)

    def constrain(
        self,
        points: list[LandmarkPoint],
        indices: dict[str, int],
        depth_axis: int = 2,
    ) -> list[LandmarkPoint]:
        """Re-solve each joint's depth so every bone keeps its measured length.

        `depth_axis` selects the coordinate the tracker guesses worst: 2 (z) for
        MediaPipe world landmarks, 1 (y) for the body frame, where y is forward.
        """
        corrected = list(points)
        for chain in CHAINS:
            for bone in chain:
                length = self.bones.get(bone)
                if length is None:
                    continue
                parent_name, child_name = BONES[bone]
                parent = _at(corrected, indices, parent_name)
                child = _at(corrected, indices, child_name)
                if parent is None or child is None:
                    continue
                solved = _solve_depth(parent, child, length, depth_axis)
                if solved is not None:
                    corrected[indices[child_name]] = solved
        return corrected


def _at(
    points: list[LandmarkPoint], indices: dict[str, int], name: str
) -> LandmarkPoint | None:
    index = indices.get(name)
    if index is None or not 0 <= index < len(points):
        return None
    point = points[index]
    if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
        return None
    return point


def _coordinates(point: LandmarkPoint) -> list[float]:
    return [float(point.x), float(point.y), float(point.z)]


def _solve_depth(
    parent: LandmarkPoint,
    child: LandmarkPoint,
    length: float,
    depth_axis: int,
) -> LandmarkPoint | None:
    """Keep the two well-seen coordinates, put the bone's remaining reach in depth."""
    parent_values = _coordinates(parent)
    child_values = _coordinates(child)
    offset = [c - p for c, p in zip(child_values, parent_values)]
    planar = math.sqrt(
        sum(value * value for axis, value in enumerate(offset) if axis != depth_axis)
    )
    if planar <= 0.0 and offset[depth_axis] == 0.0:
        return None
    moved = list(child_values)
    if planar > length:
        # The limb already spans more than its length across the image, where the
        # estimate is good. Depth cannot explain that, so lay the bone flat and
        # bring the image-plane reach in to match.
        scale = length / planar
        for axis in range(3):
            if axis != depth_axis:
                moved[axis] = parent_values[axis] + offset[axis] * scale
        moved[depth_axis] = parent_values[depth_axis]
    else:
        sign = -1.0 if offset[depth_axis] < 0 else 1.0
        depth = math.sqrt(max(length * length - planar * planar, 0.0))
        moved[depth_axis] = parent_values[depth_axis] + sign * depth
    return LandmarkPoint(moved[0], moved[1], moved[2], visibility=child.visibility)


def measure_bones(
    points: list[LandmarkPoint],
    indices: dict[str, int],
    min_visibility: float = MIN_VISIBILITY,
    depth_axis: int = 2,
) -> dict[str, float]:
    """Bone lengths from one frame, keeping only the ones this frame can measure."""
    measured: dict[str, float] = {}
    for bone, (parent_name, child_name) in BONES.items():
        parent = _at(points, indices, parent_name)
        child = _at(points, indices, child_name)
        if parent is None or child is None:
            continue
        if min(parent.visibility, child.visibility) < min_visibility:
            continue
        offset = [c - p for c, p in zip(_coordinates(child), _coordinates(parent))]
        length = math.sqrt(sum(value * value for value in offset))
        if not MIN_BONE_M <= length <= MAX_BONE_M:
            continue
        if abs(offset[depth_axis]) > length * MAX_DEPTH_FRACTION:
            # Mostly measured along the axis the tracker guesses worst.
            continue
        measured[bone] = length
    return measured


@dataclass(frozen=True)
class Stage:
    name: str
    prompt: str
    accepts: Callable[[dict[str, LandmarkPoint]], bool]


def _named(
    points: list[LandmarkPoint], indices: dict[str, int]
) -> dict[str, LandmarkPoint]:
    named = {}
    for name in (
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
    ):
        point = _at(points, indices, name)
        if point is not None:
            named[name] = point
    return named


def _both_arms(body: dict[str, LandmarkPoint], test: Callable[[str], bool]) -> bool:
    return all(
        all(f"{side}_{joint}" in body for joint in ("SHOULDER", "ELBOW", "WRIST"))
        and test(side)
        for side in ("LEFT", "RIGHT")
    )


def _t_pose(body: dict[str, LandmarkPoint]) -> bool:
    """Arms out sideways: the one pose where both arm bones lie across the camera."""

    def straight(side: str) -> bool:
        shoulder, elbow, wrist = (
            body[f"{side}_SHOULDER"],
            body[f"{side}_ELBOW"],
            body[f"{side}_WRIST"],
        )
        reach = abs(wrist.x - shoulder.x)
        return (
            reach > 0.30
            and abs(elbow.y - shoulder.y) < 0.18
            and abs(wrist.y - shoulder.y) < 0.20
        )

    return _both_arms(body, straight)


def _bent_arms(body: dict[str, LandmarkPoint]) -> bool:
    """Elbows bent, hands up: separates the forearm from the upper arm."""

    def bent(side: str) -> bool:
        shoulder, elbow, wrist = (
            body[f"{side}_SHOULDER"],
            body[f"{side}_ELBOW"],
            body[f"{side}_WRIST"],
        )
        return wrist.y < elbow.y - 0.12 and elbow.y > shoulder.y - 0.05

    return _both_arms(body, bent)


def _arms_up(body: dict[str, LandmarkPoint]) -> bool:
    def raised(side: str) -> bool:
        return body[f"{side}_WRIST"].y < body[f"{side}_SHOULDER"].y - 0.25

    return _both_arms(body, raised)


def _arms_down(body: dict[str, LandmarkPoint]) -> bool:
    def lowered(side: str) -> bool:
        return body[f"{side}_WRIST"].y > body[f"{side}_SHOULDER"].y + 0.30

    return _both_arms(body, lowered)


# Image coordinates put y downwards, so "up" is a smaller y.
STAGES: tuple[Stage, ...] = (
    Stage("t_pose", "Hold a T-pose: both arms straight out to the sides", _t_pose),
    Stage("bent_arms", "Bend both elbows, hands up beside your head", _bent_arms),
    Stage("arms_up", "Reach both arms straight up", _arms_up),
    Stage("arms_down", "Let both arms hang down at your sides", _arms_down),
)


@dataclass
class CalibrationProgress:
    stage: str
    prompt: str
    samples: int
    required: int
    finished: bool
    skeleton: Skeleton | None = None

    def status_line(self) -> str:
        if self.finished:
            return "skeleton: calibrated"
        return f"skeleton {self.stage} [{self.samples}/{self.required}]: {self.prompt}"


class SkeletonCalibrator:
    """Walk the person through the poses and keep the bones each one measures well."""

    def __init__(
        self,
        stages: Iterable[Stage] = STAGES,
        samples_per_stage: int = SAMPLES_PER_STAGE,
        hold_ms: int = STAGE_HOLD_MS,
        min_visibility: float = MIN_VISIBILITY,
        depth_axis: int = 2,
    ) -> None:
        self._stages = tuple(stages)
        self._samples_per_stage = samples_per_stage
        self._hold_ms = hold_ms
        self._min_visibility = min_visibility
        self._depth_axis = depth_axis
        self._index = 0
        self._collected: dict[str, list[float]] = {}
        self._stage_samples: list[float] = []
        self._stage_started_ms: int | None = None
        self._skeleton: Skeleton | None = None

    @property
    def finished(self) -> bool:
        return self._skeleton is not None

    @property
    def skeleton(self) -> Skeleton | None:
        return self._skeleton

    def reset(self) -> None:
        self._index = 0
        self._collected.clear()
        self._stage_samples.clear()
        self._stage_started_ms = None
        self._skeleton = None

    def update(
        self,
        timestamp_ms: int,
        points: list[LandmarkPoint] | None,
        indices: dict[str, int],
    ) -> CalibrationProgress:
        if self._skeleton is not None:
            return self._progress(finished=True)
        stage = self._stages[self._index]
        if points is None:
            self._restart_stage()
            return self._progress()
        body = _named(points, indices)
        if not stage.accepts(body):
            self._restart_stage()
            return self._progress()

        measured = measure_bones(
            points, indices, self._min_visibility, self._depth_axis
        )
        if not measured:
            return self._progress()
        if self._stage_started_ms is None:
            self._stage_started_ms = timestamp_ms
        for bone, length in measured.items():
            self._collected.setdefault(bone, []).append(length)
        self._stage_samples.append(timestamp_ms)

        held = timestamp_ms - self._stage_started_ms
        if len(self._stage_samples) >= self._samples_per_stage and held >= self._hold_ms:
            self._advance()
        return self._progress(finished=self._skeleton is not None)

    def _restart_stage(self) -> None:
        """A pose has to be held, not passed through on the way somewhere else."""
        self._stage_samples.clear()
        self._stage_started_ms = None

    def _advance(self) -> None:
        self._index += 1
        self._restart_stage()
        if self._index < len(self._stages):
            return
        self._index = len(self._stages) - 1
        bones = {
            bone: statistics.median(lengths)
            for bone, lengths in self._collected.items()
            if len(lengths) >= self._samples_per_stage
        }
        if all(bone in bones for bone in ARM_BONES):
            self._skeleton = Skeleton(bones)
        else:
            # Not enough of the body was ever measured well; start over rather
            # than hand out a skeleton with invented bones.
            self.reset()

    def _progress(self, finished: bool = False) -> CalibrationProgress:
        stage = self._stages[self._index]
        return CalibrationProgress(
            stage=stage.name,
            prompt=stage.prompt,
            samples=len(self._stage_samples),
            required=self._samples_per_stage,
            finished=finished,
            skeleton=self._skeleton,
        )


def save_skeleton(skeleton: Skeleton, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(
                {"version": 1, "bones": skeleton.bones}, file, allow_nan=False
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_skeleton(path: Path) -> Skeleton:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SkeletonError(f"Could not read the skeleton profile: {error}") from error
    if not isinstance(data, dict) or data.get("version") != 1:
        raise SkeletonError("Unsupported skeleton profile.")
    bones = data.get("bones")
    if not isinstance(bones, dict) or not bones:
        raise SkeletonError("The skeleton profile holds no bones.")
    cleaned: dict[str, float] = {}
    for name, value in bones.items():
        if name not in BONES:
            raise SkeletonError(f"Unknown bone in the skeleton profile: {name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SkeletonError(f"Bone {name} is not a number.")
        length = float(value)
        if not math.isfinite(length) or not MIN_BONE_M <= length <= MAX_BONE_M:
            raise SkeletonError(f"Bone {name} is not a plausible length: {value}")
        cleaned[name] = length
    return Skeleton(cleaned)
