"""Bounded snapshot exchange between the existing frame loop and local UI."""
from __future__ import annotations

import json
import secrets
import threading
import time
from collections import deque

from vision_robot_arm.robot.kinematics import ur7e_joint_points
from vision_robot_arm.robot.cartesian_mapping import BASE_SEPARATION_M

COMMAND_KEYS = {"calibrate": "c", "skeleton": "b", "save_calibration": "k",
                "load_calibration": "l", "reset_calibration": "x", "record": "r"}


class WebBridge:
    def __init__(self, demo=False):
        self.lock = threading.Lock()
        self.commands = deque(maxlen=8)
        self.token = secrets.token_urlsafe(32)
        self.demo = demo
        self.closed = False
        self.sequence = 0
        self.updated = time.monotonic()
        self.jpeg = b""
        self.last_publish = 0.0
        self.state = {"schema_version": 1, "sequence": 0, "status": "starting",
                      "demo": demo, "token": self.token, "message": "Starting the tracking engine…"}

    def fail(self, message):
        with self.lock:
            self.state = {**self.state, "status": "error", "message": str(message)}
            self.commands.clear()

    def snapshot(self):
        with self.lock:
            return {**self.state, "age_ms": round((time.monotonic() - self.updated) * 1000)}

    def command(self, name):
        with self.lock:
            if name not in COMMAND_KEYS:
                raise ValueError("Unknown command")
            if self.demo:
                raise ValueError("Demo is read-only. Start with --camera to use this action.")
            if self.closed or self.state["status"] != "running" or time.monotonic() - self.updated > 2:
                raise ValueError("Tracking engine is unavailable")
            if name in ("calibrate", "skeleton") and not self.state.get("detected"):
                raise ValueError("Step into the camera view before calibrating")
            if len(self.commands) >= self.commands.maxlen:
                raise ValueError("An action is already pending")
            self.commands.append((time.monotonic(), ord(COMMAND_KEYS[name])))

    def poll_key(self):
        with self.lock:
            while self.commands:
                timestamp, key = self.commands.popleft()
                if time.monotonic() - timestamp < 1:
                    return key
        return 255

    def publish(self, cv2, frame, pose, robots, **metadata):
        now = time.monotonic()
        if now - self.last_publish < 1 / 15:
            return
        self.last_publish = now
        height, width = frame.shape[:2]
        preview = cv2.resize(frame, (960, max(1, round(height * 960 / width)))) if width > 960 else frame
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        arms = {}
        if robots:
            for side, arm in robots.arms.items():
                base = ((-1 if side == "left" else 1) * BASE_SEPARATION_M / 2, 0, 0)
                arms[side] = {"joints_deg": arm.joints, "targets_deg": arm.targets,
                              "points_m": ur7e_joint_points(arm.joints, base),
                              "target_points_m": ur7e_joint_points(arm.targets, base),
                              "gripper": arm.gripper, "tcp_target_m": arm.tcp_target}
        body = [[p.x, p.y, p.z, p.visibility] for p in pose.body_landmarks] if pose and pose.body_landmarks else []
        with self.lock:
            self.sequence += 1
            self.updated = now
            state = {"schema_version": 1, "sequence": self.sequence, "status": "running",
                     "demo": self.demo, "token": self.token, "detected": pose is not None,
                     "body_m": body, "arms": arms, "angles_deg": pose.angles if pose else {},
                     "gestures": pose.gestures if pose else [], "resolution": [width, height],
                     "coordinate_frame": "body: X right, Y forward, Z up; metres",
                     **metadata}
            # Reject invalid floats at the boundary rather than emitting invalid JSON.
            self.state = json.loads(json.dumps(state, allow_nan=False))
            if ok:
                self.jpeg = encoded.tobytes()
