"""Capture ChArUco observations and save OpenCV camera calibration files."""

import os

os.environ["OPENCV_VIDEOIO_PRIORITY_OBSENSOR"] = "0"

import cv2
import numpy as np


# =====================================================
# ChArUco calibration settings for the 1280x720 capture pipeline.
# =====================================================

# ===== BOARD GEOMETRY =====
squares_x = 9
squares_y = 6

square_length = 0.032   # 32 mm
marker_length = 0.024   # 75% of the square size

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    dictionary
)

# ===== CAPTURED OBSERVATIONS =====
all_charuco_corners = []
all_charuco_ids = []

# ===== CAMERA INIT =====
cap = cv2.VideoCapture(2, cv2.CAP_MSMF)

# Keep the same resolution used by the runtime camera scripts.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# MJPG is usually faster and more stable for USB cameras on Windows.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Use a small buffer to reduce live preview latency.
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("SPACE = capture frame")
print("ENTER = run calibration")
print("ESC   = exit")

flash = 0
img_size = None

# =====================================================
# CAPTURE LOOP
# =====================================================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame grab failed")
        continue

    view = frame.copy()

    # Convert once and run all marker/ChArUco detection on grayscale data.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_size = gray.shape[::-1]

    # ==========================================
    # MARKER DETECTION
    # ==========================================
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, dictionary
    )

    valid_frame = False

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(view, corners, ids)

        retval, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )

        # Require enough interpolated ChArUco corners for a useful sample.
        if retval is not None and retval > 10:
            valid_frame = True

            cv2.aruco.drawDetectedCornersCharuco(
                view,
                charuco_corners,
                charuco_ids,
                (0, 255, 0)
            )

            cv2.putText(
                view,
                f"CORNERS: {retval}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    if not valid_frame:
        cv2.putText(
            view,
            "NO DETECTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Show how many valid calibration views have been stored.
    cv2.putText(
        view,
        f"Saved frames: {len(all_charuco_corners)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    if flash > 0:
        cv2.putText(
            view,
            "CAPTURED",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )
        flash -= 1

    cv2.imshow("ChArUco Calibration 1280x720", view)

    key = cv2.waitKey(1) & 0xFF

    # SPACE captures only frames with a valid ChArUco detection.
    if key == 32 and valid_frame:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        flash = 10
        print("Captured:", len(all_charuco_corners))

    # ENTER starts calibration with the collected samples.
    elif key == 13:
        break

    # ESC exits without saving calibration files.
    elif key == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# =====================================================
# CALIBRATION
# =====================================================

if len(all_charuco_corners) < 12:
    print("Too few frames. Minimum is 12.")
    exit()

print("\nCalibrating...")

ret, cameraMatrix, distCoeffs, rvecs, tvecs = \
    cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None
    )

print("\n==============================")
print("RMS Error:", ret)
print("==============================")

print("\nCamera Matrix:\n")
print(cameraMatrix)

print("\nDistortion Coefficients:\n")
print(distCoeffs)

# Save files consumed by ArucoCamera and the hand-eye collection script.
np.save("camera_matrix.npy", cameraMatrix)
np.save("dist_coeffs.npy", distCoeffs)

print("\nSaved files:")
print("camera_matrix.npy")
print("dist_coeffs.npy")
