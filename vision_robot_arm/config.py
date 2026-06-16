from dataclasses import dataclass
from pathlib import Path


ANGLE_MODE = "angles"
LANDMARK_MODE = "landmarks"
BOTH_MODE = "both"

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
)
DEFAULT_RECORDING_DIR = Path(__file__).resolve().parent.parent / "recordings"


@dataclass(frozen=True)
class AppConfig:
    camera: int = 0
    width: int = 0
    height: int = 0
    print_interval: float = 0.5
    visibility_threshold: float = 0.55
    smoothing_alpha: float = 0.35
    recording_dir: Path = DEFAULT_RECORDING_DIR
    model_path: Path = DEFAULT_MODEL_PATH
    num_poses: int = 1
    min_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def validate(self) -> None:
        if self.print_interval <= 0:
            raise SystemExit("--print-interval must be greater than 0")
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise SystemExit("--visibility-threshold must be between 0 and 1")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise SystemExit("--smoothing-alpha must be greater than 0 and at most 1")
        if self.num_poses <= 0:
            raise SystemExit("--num-poses must be greater than 0")

        model_path = self.model_path.resolve()
        if not model_path.exists():
            raise SystemExit(
                f"Pose model not found: {model_path}\n"
                "Expected the default model at models/pose_landmarker_lite.task."
            )
