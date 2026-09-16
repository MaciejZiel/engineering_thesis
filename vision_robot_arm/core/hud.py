from typing import Any

Point = tuple[int, int]
Color = tuple[int, int, int]

FONT_SCALE = 0.5
LINE_HEIGHT = 20
PADDING = 8
PANEL_COLOR: Color = (20, 20, 20)
PANEL_ALPHA = 0.6
TEXT_COLOR: Color = (235, 235, 235)
REFERENCE_HEIGHT = 720
MIN_UI_SCALE = 0.6
MAX_UI_SCALE = 2.5


def ui_scale(frame: Any) -> float:
    height = frame.shape[0]
    return max(MIN_UI_SCALE, min(MAX_UI_SCALE, height / REFERENCE_HEIGHT))


def scaled(value: float, frame: Any) -> int:
    return max(1, round(value * ui_scale(frame)))


def text_size(cv2: Any, text: str, scale: float = FONT_SCALE, thickness: int = 1) -> tuple[int, int]:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return width, height + baseline


def fill_translucent(
    cv2: Any,
    frame: Any,
    top_left: Point,
    bottom_right: Point,
    color: Color = PANEL_COLOR,
    alpha: float = PANEL_ALPHA,
) -> None:
    height, width = frame.shape[:2]
    x0, y0 = max(0, top_left[0]), max(0, top_left[1])
    x1, y1 = min(width, bottom_right[0]), min(height, bottom_right[1])
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    overlay = region.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, region, 1.0 - alpha, 0.0, region)


def put_text(
    cv2: Any,
    frame: Any,
    text: str,
    origin: Point,
    color: Color = TEXT_COLOR,
    scale: float = FONT_SCALE,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_text_panel(
    cv2: Any,
    frame: Any,
    lines: list[str],
    origin: Point,
    scale: float = FONT_SCALE,
    line_height: int = LINE_HEIGHT,
    padding: int = PADDING,
    color: Color = TEXT_COLOR,
) -> Point:
    if not lines:
        return origin
    factor = ui_scale(frame)
    scale *= factor
    line_height = max(1, round(line_height * factor))
    padding = max(1, round(padding * factor))
    thickness = 1 if factor < 1.4 else 2
    width = max(text_size(cv2, line, scale, thickness)[0] for line in lines)
    left, top = origin
    right = left + width + 2 * padding
    bottom = top + len(lines) * line_height + 2 * padding
    fill_translucent(cv2, frame, (left, top), (right, bottom))
    for row, line in enumerate(lines):
        baseline_y = top + padding + (row + 1) * line_height - round(5 * factor)
        put_text(cv2, frame, line, (left + padding, baseline_y), color, scale, thickness)
    return right, bottom


def panel_height(frame: Any, line_count: int, line_height: int = LINE_HEIGHT, padding: int = PADDING) -> int:
    factor = ui_scale(frame)
    return line_count * max(1, round(line_height * factor)) + 2 * max(1, round(padding * factor))


def draw_label(
    cv2: Any,
    frame: Any,
    text: str,
    anchor: Point,
    color: Color = TEXT_COLOR,
    scale: float = FONT_SCALE,
) -> None:
    factor = ui_scale(frame)
    scale *= factor
    thickness = 1 if factor < 1.4 else 2
    width, height = text_size(cv2, text, scale, thickness)
    margin = max(2, round(4 * factor))
    left, baseline_y = anchor
    fill_translucent(
        cv2,
        frame,
        (left - margin, baseline_y - height - margin // 2),
        (left + width + margin, baseline_y + margin),
    )
    put_text(cv2, frame, text, (left, baseline_y), color, scale, thickness)
