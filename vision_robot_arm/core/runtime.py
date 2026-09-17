from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeDeps:
    cv2: Any
    mp: Any
    np: Any
    vision: Any
    base_options: Any


def load_camera_dependency() -> Any:
    """Load OpenCV without importing MediaPipe or constructing models."""
    try:
        import cv2  # type: ignore
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: opencv-contrib-python\n"
            "Install it with: python -m pip install -r requirements.txt"
        ) from error
    return cv2


def load_runtime_dependencies() -> RuntimeDeps:
    missing = []
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None
        missing.append("opencv-contrib-python")

    try:
        import mediapipe as mp  # type: ignore
    except ImportError:
        mp = None
        missing.append("mediapipe")

    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None
        missing.append("numpy")

    try:
        from mediapipe.tasks.python import vision  # type: ignore
        from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
    except ImportError:
        vision = None
        BaseOptions = None
        if "mediapipe" not in missing:
            missing.append("mediapipe")

    if missing:
        packages = ", ".join(missing)
        raise SystemExit(
            f"Missing dependencies: {packages}\n"
            "Install them with:\n"
            "  python -m pip install -r requirements.txt"
        )

    return RuntimeDeps(
        cv2=cv2,
        mp=mp,
        np=np,
        vision=vision,
        base_options=BaseOptions,
    )
