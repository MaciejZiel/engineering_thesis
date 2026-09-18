"""Flat camera-plane and joint-angle diagrams for the laboratory tests."""

import math

from vision_robot_arm.vision.drawing import draw_stick_figure, draw_hands
from vision_robot_arm.core.pose_state import mirror_landmarks
from vision_robot_arm.vision.ui_style import BACKGROUND, ACCENT, MUTED, TEXT


def draw_planar_body(cv2, np, canvas, state, indices, *, mirrored=False):
    canvas[:] = BACKGROUND
    if state is None:
        return
    points = mirror_landmarks(state.landmarks) if mirrored else state.landmarks
    hands = {side: mirror_landmarks(hand) if mirrored else hand for side, hand in state.hand_landmarks.items()}
    draw_stick_figure(cv2, canvas, points, indices, 0.55)
    draw_hands(cv2, canvas, hands, points, indices, 0.55)


def draw_planar_robot(cv2, np, canvas, robot, state, indices, *, mirrored=False):
    """Joint chain schematic only: it is not a calibrated robot TCP projection."""
    canvas[:] = BACKGROUND
    h, w = canvas.shape[:2]
    cv2.putText(canvas, "2D joint diagram (not TCP position)", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, MUTED, 1, cv2.LINE_AA)
    if robot is None:
        return
    for i, (side, arm) in enumerate(robot.arms.items()):
        cx = int(w * (i + 0.5) / len(robot.arms))
        cy = int(h * 0.58)
        length = min(h * 0.17, w / len(robot.arms) * 0.16)
        for joints, color, thickness in ((arm.targets, MUTED, 2), (arm.joints, ACCENT, 4)):
            x, y, angle = cx, cy, 0.0
            for joint, scale in (("shoulder", 1.0), ("elbow", 1.0), ("wrist_1", 0.45)):
                angle += math.radians(joints.get(joint, 0.0))
                nx, ny = x + length*scale*math.cos(angle), y - length*scale*math.sin(angle)
                cv2.line(canvas, (int(x), int(y)), (int(nx), int(ny)), color, thickness, cv2.LINE_AA)
                cv2.circle(canvas, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
                x, y = nx, ny
        cv2.putText(canvas, side.upper(), (cx - 30, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
        values = " / ".join(f"{arm.joints.get(j, 0):+.0f}" for j in ("shoulder", "elbow", "wrist_1"))
        cv2.putText(canvas, values, (max(8, cx - 80), h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.43, TEXT, 1, cv2.LINE_AA)
