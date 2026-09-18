"""Generate a printable 2x3 sheet of styled ArUco markers with IDs 0-5."""

import cv2
import numpy as np

# ==========================================
# SINGLE MARKER PARAMETERS
# ==========================================
size = 640
marker_size = 330
circle_radius = 320
inner_white_radius = 280

dark_red = (0, 0, 200)

cols = 2
rows = 3
margin = 40

# ==========================================
# ARUCO DICTIONARY
# ==========================================
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

# ==========================================
# FUNCTION TO GENERATE A SINGLE MARKER
# ==========================================
def generate_marker(marker_id):
    """Create one marker tile with the project visual style and ID label."""

    marker = cv2.aruco.generateImageMarker(
        aruco_dict, marker_id, marker_size
    )

    marker_bgr = np.ones(
        (marker_size, marker_size, 3),
        dtype=np.uint8
    ) * 255

    marker_bgr[marker == 0] = dark_red

    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    cx = size // 2
    cy = size // 2

    # Draw the red background circle used for the printed object markers.
    cv2.circle(img, (cx, cy), circle_radius, dark_red, -1)

    # Add a white lower semicircle for the PJAIT-style marker background.
    cv2.ellipse(
        img,
        (cx, cy),
        (inner_white_radius, inner_white_radius),
        0,
        180,
        360,
        (255, 255, 255),
        -1
    )

    # Place the ArUco code on a white square for reliable detection.
    pad = 30
    cv2.rectangle(
        img,
        (cx - marker_size//2 - pad, cy - marker_size//2 - pad),
        (cx + marker_size//2 + pad, cy + marker_size//2 + pad),
        (255, 255, 255),
        -1
    )

    # Paste the generated ArUco marker into the center of the tile.
    x1 = cx - marker_size // 2
    y1 = cy - marker_size // 2
    x2 = x1 + marker_size
    y2 = y1 + marker_size

    img[y1:y2, x1:x2] = marker_bgr

    # ======================================
    # NUMBER AT THE BOTTOM (centered, white)
    # ======================================
    text = str(marker_id)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 3.0
    thickness = 6

    (tw, th), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )

    tx = cx - tw // 2
    ty = size - 45

    cv2.putText(
        img,
        text,
        (tx, ty),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return img

# Generate marker IDs expected by the box-lifting demo.
tiles = []

for marker_id in range(6):
    tiles.append(generate_marker(marker_id))

# Compose all marker tiles into one printable 2x3 sheet.
sheet_h = rows * size + (rows + 1) * margin
sheet_w = cols * size + (cols + 1) * margin

sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 255

for i, tile in enumerate(tiles):

    r = i // cols
    c = i % cols

    x = margin + c * (size + margin)
    y = margin + r * (size + margin)

    sheet[y:y+size, x:x+size] = tile

# ==========================================
# SAVE
# ==========================================
cv2.imwrite("aruco_sheet_numbers_bottom.png", sheet)

print("Saved: aruco_sheet_numbers_bottom.png")
