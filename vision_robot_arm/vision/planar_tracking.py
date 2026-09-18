"""Signed arm angles from camera XY only; no inferred depth enters control."""

import math


class PlanarArmTracker:
    """Keep signed angles continuous across +/-180 degrees while visible."""

    def __init__(self):
        self._previous: dict[str, float] = {}

    def reset(self):
        self._previous.clear()

    def measure(self, landmarks, hands, indices, aspect_ratio, min_visibility, mirrored):
        if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
            return {}
        result = {}

        def point(name):
            index = indices.get(name)
            if index is None or not 0 <= index < len(landmarks):
                return None
            p = landmarks[index]
            if p.visibility < min_visibility or not all(math.isfinite(v) for v in (p.x, p.y)):
                return None
            return p

        def vector(a, b):
            if a is None or b is None:
                return None
            dx = (b.x - a.x) * aspect_ratio * (-1 if mirrored else 1)
            dy = a.y - b.y
            if not all(math.isfinite(v) for v in (dx, dy)) or math.hypot(dx, dy) < 0.015:
                return None
            return dx, dy

        def turn(a, b):
            return math.degrees(math.atan2(a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1]))

        def save(name, angle):
            previous = self._previous.get(name, angle)
            angle = previous + (angle - previous + 180) % 360 - 180
            self._previous[name] = angle
            result[name] = angle

        for side in ("left", "right"):
            upper = side.upper()
            shoulder, elbow, wrist = (point(f"{upper}_{part}") for part in ("SHOULDER", "ELBOW", "WRIST"))
            arm = vector(shoulder, elbow)
            forearm = vector(elbow, wrist)
            hand = hands.get(side, ())
            palm = vector(hand[0], hand[9]) if len(hand) > 9 else None
            if arm is not None:
                save(f"{side}_shoulder_elevation", math.degrees(math.atan2(arm[1], arm[0])))
            if arm is not None and forearm is not None:
                save(f"{side}_elbow", turn(arm, forearm))
            if forearm is not None and palm is not None:
                save(f"{side}_wrist", turn(forearm, palm))
        return result
