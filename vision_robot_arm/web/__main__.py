"""Run: python -m vision_robot_arm.web [--camera | --demo] [--port 8765]."""
import argparse
import threading
import time
from pathlib import Path

from .bridge import WebBridge


def run_demo(bridge):
    import cv2
    import numpy as np
    from vision_robot_arm.robot.targets import ArmState, RobotState, full_joint_pose
    joints = full_joint_pose({"shoulder": -65, "elbow": 80, "wrist_1": -100})
    robots = RobotState({side: ArmState(joints, joints, "open") for side in ("left", "right")}, False)
    image = np.full((540, 960, 3), (27, 26, 25), np.uint8)
    while not bridge.closed:
        bridge.publish(cv2, image, None, robots, fps=0, quality=0, calibrated=False,
                       recording=False, source="Demonstration", message="Illustrative robot pose · no camera or robot connected",
                       skeleton_status="", mirrored=True)
        time.sleep(1 / 15)


def main():
    parser = argparse.ArgumentParser(description="Motion Twin local web workspace (simulation only)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", action="store_true", help="Use the real camera and existing tracking pipeline")
    source.add_argument("--demo", action="store_true", help="Read-only illustration without camera or robot")
    parser.add_argument("--port", type=int, default=8765)
    args, tracking_args = parser.parse_known_args()
    from .server import create_app
    import uvicorn
    bridge = WebBridge(demo=not args.camera)

    def worker():
        try:
            if bridge.demo:
                run_demo(bridge)
            else:
                from vision_robot_arm.app import run_app
                from vision_robot_arm.cli import parse_args
                from dataclasses import replace
                config = parse_args(tracking_args)
                if config.robot.backend not in ("none", "sim", "debug"):
                    raise ValueError("Web launch supports simulation only. Use main.py for hardware workflows.")
                config = replace(config, robot=replace(config.robot, backend="sim"))
                config.validate()
                result = run_app(config, web=bridge)
                if not bridge.closed:
                    bridge.fail("Camera session ended. Restart the Python launcher to reconnect." if result == 0 else "Camera capture stopped. Check the device and restart the launcher.")
        except (Exception, SystemExit) as error:
            bridge.fail(error)

    worker_thread = threading.Thread(target=worker, daemon=True, name="tracking")
    worker_thread.start()
    root = Path(__file__).resolve().parents[2].parent / "frontend" / "dist"
    print(f"Motion Twin: http://127.0.0.1:{args.port} ({'demo' if bridge.demo else 'camera + simulation'})")
    try:
        uvicorn.run(create_app(bridge, root), host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        bridge.closed = True
        worker_thread.join(timeout=5)


if __name__ == "__main__":
    main()
