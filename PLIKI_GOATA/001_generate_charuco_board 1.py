"""Generate a printable ChArUco board image for camera calibration."""

import cv2

# Board geometry must match the values used by the calibration script.
board = cv2.aruco.CharucoBoard(
    (9, 6),     # number of chessboard squares (NOT inner corners)
    0.024,      # square length (meters)
    0.018,      # marker length (meters)
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
)

img = board.generateImage((2000, 1500))

# Save a high-resolution image that can be printed or embedded in a PDF.
cv2.imwrite("charuco_board.png", img)
