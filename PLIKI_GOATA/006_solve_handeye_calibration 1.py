"""Solve eye-in-hand calibration and save the best camera/flange transform."""

import cv2
import numpy as np


DATASET_PATH = "handeye_dataset.npz"
RESULT_PATH = "handeye_result.npz"


HAND_EYE_METHODS = [
    ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
    ("PARK", cv2.CALIB_HAND_EYE_PARK),
    ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
    ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
    ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
]


def make_transform(R, t):
    """Build a 4x4 homogeneous transform from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R)
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def fixed_target_residuals(T_cam2flange, R_flange2base, t_flange2base,
                           R_marker2cam, t_marker2cam):
    """Return base-frame marker poses implied by each sample.

    For an eye-in-hand setup:
        base_T_marker = base_T_flange * T_cam2flange * cam_T_marker

    If calibration is good and the marker was fixed in the robot base/world,
    these positions should cluster tightly.
    """
    positions = []

    for R_bf, t_bf, R_cm, t_cm in zip(
        R_flange2base, t_flange2base, R_marker2cam, t_marker2cam
    ):
        T_base_from_flange = make_transform(R_bf, t_bf)
        T_cam_from_marker = make_transform(R_cm, t_cm)
        T_base_from_marker = (
            T_base_from_flange @ T_cam2flange @ T_cam_from_marker
        )
        positions.append(T_base_from_marker[:3, 3])

    positions = np.asarray(positions)
    mean_position = positions.mean(axis=0)
    errors = np.linalg.norm(positions - mean_position, axis=1)

    return mean_position, positions.std(axis=0), errors


def main():
    data = np.load(DATASET_PATH)

    # UR pose with TCP set to zero is flange -> base, i.e. base_T_flange.
    R_flange2base = data["R_flange2base"]
    t_flange2base = data["t_flange2base"]

    # ArUco pose from OpenCV is marker -> camera, i.e. cam_T_marker.
    R_marker2cam = data["R_marker2cam"]
    t_marker2cam = data["t_marker2cam"]

    if len(R_flange2base) != len(R_marker2cam):
        raise ValueError("Dataset size mismatch")
    if len(R_flange2base) < 6:
        raise ValueError("Too few samples for calibration")

    R_gripper2base = [R for R in R_flange2base]
    t_gripper2base = [t.reshape(3, 1) for t in t_flange2base]
    R_target2cam = [R for R in R_marker2cam]
    t_target2cam = [t.reshape(3, 1) for t in t_marker2cam]

    # Compare several OpenCV solvers and keep the one with the tightest target cluster.
    results = []

    print("\n=== HAND-EYE METHOD COMPARISON ===")

    for method_name, method_id in HAND_EYE_METHODS:
        try:
            # OpenCV returns camera -> gripper/flange.
            R_cam2flange, t_cam2flange = cv2.calibrateHandEye(
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
                method=method_id,
            )
        except cv2.error as exc:
            print(f"{method_name:10s} failed: {exc}")
            continue

        T_cam2flange = make_transform(R_cam2flange, t_cam2flange)
        marker_mean, marker_std, marker_errors = fixed_target_residuals(
            T_cam2flange,
            R_flange2base,
            t_flange2base,
            R_marker2cam,
            t_marker2cam,
        )

        rms_error = float(np.sqrt(np.mean(marker_errors ** 2)))
        max_error = float(np.max(marker_errors))

        results.append(
            {
                "method_name": method_name,
                "R_cam2flange": R_cam2flange,
                "t_cam2flange": t_cam2flange,
                "T_cam2flange": T_cam2flange,
                "T_flange2cam": np.linalg.inv(T_cam2flange),
                "marker_mean": marker_mean,
                "marker_std": marker_std,
                "rms_error": rms_error,
                "max_error": max_error,
            }
        )

        print(
            f"{method_name:10s} "
            f"t_cam2flange[m]={t_cam2flange.reshape(3)} "
            f"target_RMS={rms_error * 1000:.1f} mm "
            f"target_MAX={max_error * 1000:.1f} mm"
        )

    if not results:
        raise RuntimeError("All hand-eye methods failed")

    best = min(results, key=lambda item: item["rms_error"])

    R_cam2flange = best["R_cam2flange"]
    t_cam2flange = best["t_cam2flange"]
    T_cam2flange = best["T_cam2flange"]
    T_flange2cam = best["T_flange2cam"]

    print("\n=== BEST RESULT ===")
    print(f"Method: {best['method_name']}")
    print(f"Fixed-marker RMS error: {best['rms_error'] * 1000:.1f} mm")
    print(f"Fixed-marker max error: {best['max_error'] * 1000:.1f} mm")
    print(f"Fixed-marker base position mean [m]: {best['marker_mean']}")
    print(f"Fixed-marker base position std [mm]: {best['marker_std'] * 1000}")

    print("\n=== CAM2FLANGE ===")
    print("Translation, camera origin in flange frame [m]:")
    print(t_cam2flange.reshape(3))
    print("Rotation matrix:")
    print(R_cam2flange)
    print("Homogeneous matrix:")
    print(T_cam2flange)

    print("\n=== FLANGE2CAM ===")
    print(T_flange2cam)

    np.savez(
        RESULT_PATH,
        method=best["method_name"],
        R_cam2flange=R_cam2flange,
        t_cam2flange=t_cam2flange,
        T_cam2flange=T_cam2flange,
        T_flange2cam=T_flange2cam,
    )

    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
