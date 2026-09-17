"""UR7e forward kinematics and a continuity-preserving position IK solver."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from vision_robot_arm.robot.targets import JOINT_NAMES, full_joint_pose

Point3 = tuple[float, float, float]

# Official UR5e/UR7e standard DH parameters, metres/radians.
UR7E_DH = (
    (0.0, 0.1625, math.pi / 2),
    (-0.425, 0.0, 0.0),
    (-0.3922, 0.0, 0.0),
    (0.0, 0.1333, math.pi / 2),
    (0.0, 0.0997, -math.pi / 2),
    (0.0, 0.0996, 0.0),
)

POSITION_JOINTS = ("base", "shoulder", "elbow")
DEFAULT_IK_LIMITS = {
    "base": (-180.0, 180.0),
    "shoulder": (-180.0, 0.0),
    "elbow": (-160.0, 160.0),
}


def ur7e_joint_points(
    joints_deg: dict[str, float], base: Point3 = (0.0, 0.0, 0.0)
) -> tuple[Point3, ...]:
    """Return base plus all six UR7e joint-frame origins."""
    pose = full_joint_pose(joints_deg)
    transform = _identity()
    points = [base]
    for name, (a, d, alpha) in zip(JOINT_NAMES, UR7E_DH):
        transform = _matmul(transform, _dh(math.radians(pose[name]), d, a, alpha))
        point = _transform_point(transform, (0.0, 0.0, 0.0))
        points.append(
            (point[0] + base[0], point[1] + base[1], point[2] + base[2])
        )
    return tuple(points)


def solve_position_ik(
    target: Point3,
    initial_joints: dict[str, float],
    base: Point3 = (0.0, 0.0, 0.0),
    *,
    tolerance_m: float = 0.008,
    max_iterations: int = 60,
) -> dict[str, float] | None:
    """Place the TCP using base/shoulder/elbow while preserving wrist orientation.

    Seeding every frame from the preceding solution selects a continuous IK branch.
    Unreachable points return ``None`` instead of exposing a misleading nearest pose.
    """
    if not all(math.isfinite(value) for value in target):
        return None
    joints = full_joint_pose(initial_joints)
    for name, (minimum, maximum) in DEFAULT_IK_LIMITS.items():
        joints[name] = _clamp(joints[name], minimum, maximum)

    target_vector = np.asarray(target, dtype=float)
    for _ in range(max_iterations):
        current = np.asarray(ur7e_joint_points(joints, base)[-1], dtype=float)
        error = target_vector - current
        if float(np.linalg.norm(error)) <= tolerance_m:
            return {name: float(value) for name, value in joints.items()}

        jacobian_columns = []
        epsilon_deg = 0.1
        for name in POSITION_JOINTS:
            perturbed = dict(joints)
            perturbed[name] += epsilon_deg
            endpoint = np.asarray(
                ur7e_joint_points(perturbed, base)[-1], dtype=float
            )
            jacobian_columns.append(
                (endpoint - current) / math.radians(epsilon_deg)
            )
        jacobian = np.asarray(jacobian_columns, dtype=float).T
        damping = 0.02
        try:
            delta_rad = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(3), error
            )
        except np.linalg.LinAlgError:
            return None
        delta_deg = np.clip(np.degrees(delta_rad), -5.0, 5.0)
        for name, delta in zip(POSITION_JOINTS, delta_deg):
            minimum, maximum = DEFAULT_IK_LIMITS[name]
            joints[name] = _clamp(joints[name] + float(delta), minimum, maximum)

    endpoint = ur7e_joint_points(joints, base)[-1]
    if math.dist(endpoint, target) <= tolerance_m:
        return {name: float(value) for name, value in joints.items()}
    return None


def _identity() -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(4))
        for row in range(4)
    )


def _dh(
    theta: float, d: float, a: float, alpha: float
) -> tuple[tuple[float, ...], ...]:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return (
        (ct, -st * ca, st * sa, a * ct),
        (st, ct * ca, -ct * sa, a * st),
        (0.0, sa, ca, d),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matmul(
    first: Iterable[Iterable[float]], second: Iterable[Iterable[float]]
) -> tuple[tuple[float, ...], ...]:
    first_rows, second_rows = tuple(map(tuple, first)), tuple(map(tuple, second))
    columns = tuple(zip(*second_rows))
    return tuple(
        tuple(sum(a * b for a, b in zip(row, column)) for column in columns)
        for row in first_rows
    )


def _transform_point(
    transform: tuple[tuple[float, ...], ...], point: Point3
) -> Point3:
    vector = (*point, 1.0)
    return tuple(
        sum(row[index] * vector[index] for index in range(4))
        for row in transform[:3]
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
