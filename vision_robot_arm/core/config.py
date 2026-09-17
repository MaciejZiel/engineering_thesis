from dataclasses import dataclass
from pathlib import Path

from vision_robot_arm.robot.config import RobotConfig

ANGLE_MODE = "angles"
LANDMARK_MODE = "landmarks"
BOTH_MODE = "both"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
DEFAULT_HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DEFAULT_RECORDING_DIR = PROJECT_ROOT / "recordings"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class AppConfig:
    camera: int | str = 0
    camera_backend: str = "auto"
    camera_format: str = "auto"
    width: int = 1920
    height: int = 1080
    camera_fps: float = 30.0
    inference_width: int = 960
    inference_height: int = 540
    mirror: bool = True
    print_interval: float = 0.5
    visibility_threshold: float = 0.55
    smoothing_alpha: float = 0.35
    recording_dir: Path = DEFAULT_RECORDING_DIR
    model_path: Path = DEFAULT_MODEL_PATH
    hand_model_path: Path = DEFAULT_HAND_MODEL_PATH
    hands: bool = True
    video_path: Path | None = None
    loop_video: bool = False
    robot: RobotConfig = RobotConfig()
    test_mode: bool = False
    num_poses: int = 1
    min_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    hand_detection_confidence: float = 0.4
    hand_presence_confidence: float = 0.4

    def validate(self) -> None:
        if self.print_interval <= 0:
            raise SystemExit("--print-interval must be greater than 0")
        if not 1.0 <= self.camera_fps <= 120.0:
            raise SystemExit("--fps must be between 1 and 120")
        if isinstance(self.camera, int) and self.camera < 0:
            raise SystemExit("--camera index must be 0 or greater")
        valid_backends = ("auto", "v4l2", "dshow", "msmf", "avfoundation", "any")
        if self.camera_backend.lower() not in valid_backends:
            raise SystemExit(
                f"--camera-backend must be one of: {', '.join(valid_backends)}"
            )
        valid_formats = ("auto", "mjpg", "yuyv", "nv12", "h264")
        if self.camera_format.lower() not in valid_formats:
            raise SystemExit(
                f"--camera-format must be one of: {', '.join(valid_formats)}"
            )
        if self.inference_width <= 0 or self.inference_height <= 0:
            raise SystemExit(
                "--inference-width and --inference-height must be greater than 0"
            )
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise SystemExit("--visibility-threshold must be between 0 and 1")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise SystemExit("--smoothing-alpha must be greater than 0 and at most 1")
        if self.num_poses <= 0:
            raise SystemExit("--num-poses must be greater than 0")
        for flag, value in (
            ("--min-detection-confidence", self.min_detection_confidence),
            ("--min-pose-presence-confidence", self.min_pose_presence_confidence),
            ("--min-tracking-confidence", self.min_tracking_confidence),
            ("--hand-detection-confidence", self.hand_detection_confidence),
            ("--hand-presence-confidence", self.hand_presence_confidence),
        ):
            # MediaPipe aborts the process on an out-of-range value instead of raising.
            if not 0.0 <= value <= 1.0:
                raise SystemExit(f"{flag} must be between 0 and 1")
        self.robot.validate()
        if self.video_path is not None and not self.video_path.exists():
            raise SystemExit(f"Video file not found: {self.video_path.resolve()}")

        model_path = self.model_path.resolve()
        if not model_path.exists():
            raise SystemExit(
                f"Pose model not found: {model_path}\n"
                "Expected the default model at models/pose_landmarker_lite.task."
            )
        hand_model_path = self.hand_model_path.resolve()
        if self.hands and not hand_model_path.exists():
            raise SystemExit(
                f"Hand model not found: {hand_model_path}\n"
                f"Download it from {HAND_MODEL_URL} or run with --no-hands."
            )
