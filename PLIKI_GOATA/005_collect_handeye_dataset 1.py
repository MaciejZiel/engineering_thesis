"""Collect paired robot flange poses and ArUco marker poses for hand-eye calibration."""

import cv2
import numpy as np
import time
import rtde_receive
import rtde_control

# =========================
# CONFIG
# =========================
ROBOT_IP = "10.20.3.136"
CAM_ID = 2
N_SAMPLES = 25

MARKER_LENGTH = 0.032

Z_MIN = 0.15
Z_MAX = 1.20

BORDER_MARGIN = 120

MIN_ROT_DELTA = 0.15  # rad - minimum orientation change between samples

# =========================
# INIT CAMERA
# =========================
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

mtx = np.load("camera_matrix.npy")
dist = np.load("dist_coeffs.npy")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()

# =========================
# ROBOT
# =========================
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

rtde_c.freedriveMode()
rtde_c.setTcp([0, 0, 0, 0, 0, 0])

print("Freedrive ON (flange frame)")

# =========================
# STORAGE
# =========================
R_flange2base = []
t_flange2base = []

R_marker2cam = []
t_marker2cam = []

# for validation diversity
last_R = None

# =========================
# HELPERS
# =========================
def pose_to_RT(pose):
    """Convert a UR TCP pose vector into rotation matrix and translation."""
    x, y, z, rx, ry, rz = pose
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    t = np.array([x, y, z])
    return R, t


def rotation_distance(R1, R2):
    """Return angular distance between two rotation matrices in radians."""
    R_rel = R1.T @ R2
    trace = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return np.arccos(trace)


def valid_detection(tvec, center, frame_shape):
    """Reject marker detections that are too near/far or close to the image edge."""
    h, w = frame_shape[:2]
    x, y = center

    if tvec[2] < Z_MIN or tvec[2] > Z_MAX:
        return False, "Z out of range"

    if (x < BORDER_MARGIN or x > w - BORDER_MARGIN or
        y < BORDER_MARGIN or y > h - BORDER_MARGIN):
        return False, "marker too close border"

    return True, "OK"


# =========================
# LOOP
# =========================
print("\nHAND-EYE DATA COLLECTION")
print("ENTER = capture | q = quit")

i = 0

while i < N_SAMPLES:

    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    detected = False
    valid = False
    reason = ""

    rvec = None
    tvec = None

    if ids is not None:

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, mtx, dist
        )

        rvec = rvecs[0][0]
        tvec = tvecs[0][0]

        center = corners[0][0].mean(axis=0).astype(int)

        valid, reason = valid_detection(tvec, center, frame.shape)

        R_marker, _ = cv2.Rodrigues(rvec)

        detected = True

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)

        cv2.putText(frame, reason, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255) if valid else (0, 0, 255), 2)

    # Read flange pose while the robot is in freedrive mode.
    tcp_pose = rtde_r.getActualTCPPose()
    Rb, tb = pose_to_RT(tcp_pose)

    # Good hand-eye data needs varied robot orientations, not only translations.
    if last_R is not None and detected:
        rot_delta = rotation_distance(last_R, Rb)
    else:
        rot_delta = 999

    # =========================
    # UI
    # =========================
    cv2.putText(frame, f"Samples: {i}/{N_SAMPLES}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, f"rot delta: {rot_delta:.3f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    status = "GOOD" if (valid and rot_delta > MIN_ROT_DELTA) else "BAD"
    color = (0, 255, 0) if status == "GOOD" else (0, 0, 255)

    cv2.putText(frame, status, (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Hand-Eye Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # CAPTURE
    # =========================
    if key == 13:

        if not detected:
            print("Rejected: no marker")
            continue

        if not valid:
            print("Rejected:", reason)
            continue

        if last_R is not None and rot_delta < MIN_ROT_DELTA:
            print("Rejected: not enough robot rotation diversity")
            continue

        # Robot pose: flange -> base.
        R_flange2base.append(Rb)
        t_flange2base.append(tb)

        # Camera observation from OpenCV: marker -> camera.
        R_marker2cam.append(R_marker)
        t_marker2cam.append(tvec.reshape(3))

        last_R = Rb.copy()

        print(f"Captured {i}")

        i += 1
        time.sleep(0.2)

    elif key == ord('q'):
        break

# =========================
# SAVE DATA
# =========================
np.savez(
    "handeye_dataset.npz",
    R_flange2base=R_flange2base,
    t_flange2base=t_flange2base,
    R_marker2cam=R_marker2cam,
    t_marker2cam=t_marker2cam
)

print("\nDATA SAVED: handeye_dataset.npz")

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

rtde_c.endFreedriveMode()
rtde_c.stopScript()
