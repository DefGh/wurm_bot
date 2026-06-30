from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import SCREENS_DIR
from .models import OcrText
from .text import normalize, timestamp
from .vision import ocr_image


DIRECTIONS = ("north", "east", "south", "west")
LABEL_ALIASES = {
    "n": "north",
    "north": "north",
    "e": "east",
    "east": "east",
    "s": "south",
    "south": "south",
    "w": "west",
    "west": "west",
}


@dataclass(frozen=True)
class CompassReading:
    heading: str | None
    direction_degrees: int | None
    center: tuple[int, int] | None
    arrow: tuple[int, int] | None
    radius: int | None
    axis: tuple[int, int, int, int] | None
    labels: dict[str, OcrText]
    image: Image.Image
    region: tuple[int, int, int, int] | None
    status: str

    @property
    def available(self) -> bool:
        return self.arrow is not None and self.heading is not None and self.direction_degrees is not None

    @property
    def visible(self) -> bool:
        return self.center is not None and self.radius is not None and self.region is not None


def read_compass(
    image: Image.Image,
    region: tuple[int, int, int, int] | None = None,
    *,
    require_heading: bool = False,
) -> CompassReading:
    image = image.convert("RGB")
    explicit_region = region or env_compass_region()
    try:
        if explicit_region is None:
            center, radius, region = find_compass_circle_auto(image)
        else:
            region = clamp_box(explicit_region, image.size)
            center, radius = find_compass_circle(image, region)
    except RuntimeError:
        if require_heading:
            raise
        return CompassReading(
            heading=None,
            direction_degrees=None,
            center=None,
            arrow=None,
            radius=None,
            axis=None,
            labels={},
            image=image,
            region=None,
            status="hidden",
        )
    axis = None
    arrow = find_red_arrow_tip_by_radius(image, region, center, radius)
    labels: dict[str, OcrText] = {}
    if arrow is None:
        if require_heading:
            raise RuntimeError("Compass heading is not visible")
        heading = None
        direction_degrees = None
        status = "visible"
    else:
        heading = heading_from_points(center, arrow)
        direction_degrees = direction_degrees_from_points(center, arrow)
        status = "available"
    return CompassReading(
        heading=heading,
        direction_degrees=direction_degrees,
        center=center,
        arrow=arrow,
        radius=radius,
        axis=axis,
        labels=labels,
        image=image,
        region=region,
        status=status,
    )


def env_compass_region() -> tuple[int, int, int, int] | None:
    raw_value = os.environ.get("WURM_COMPASS_REGION")
    if not raw_value:
        return None
    return parse_box(raw_value, "WURM_COMPASS_REGION")


def parse_box(raw_value: str, name: str = "region") -> tuple[int, int, int, int]:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 4:
        raise RuntimeError(f"{name} must use x1,y1,x2,y2 format")
    try:
        x1, y1, x2, y2 = (int(part) for part in parts)
    except ValueError as error:
        raise RuntimeError(f"{name} must contain integer x1,y1,x2,y2 values") from error
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"{name} must use a positive-width box, got {raw_value!r}")
    return x1, y1, x2, y2


def clamp_box(box: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def shift_text(text: OcrText, dx: int, dy: int) -> OcrText:
    return OcrText(text.text, text.score, text.x1 + dx, text.y1 + dy, text.x2 + dx, text.y2 + dy)


def find_compass_labels(texts: Iterable[OcrText]) -> dict[str, OcrText]:
    by_direction: dict[str, list[OcrText]] = {direction: [] for direction in DIRECTIONS}
    for text in texts:
        cleaned = normalize(text.text).strip(".:;|[](){}<>")
        direction = LABEL_ALIASES.get(cleaned)
        if direction is not None:
            by_direction[direction].append(text)

    combos = compass_label_combinations(by_direction)
    if not combos:
        raise RuntimeError("Compass labels were not found by OCR")

    best = max(combos, key=compass_label_score)
    if compass_label_score(best) <= 0:
        raise RuntimeError("Compass labels were found, but their geometry does not look like a compass")
    return best


def compass_label_combinations(by_direction: dict[str, list[OcrText]]) -> list[dict[str, OcrText]]:
    combos: list[dict[str, OcrText]] = []
    norths = by_direction["north"] or [None]
    easts = by_direction["east"] or [None]
    souths = by_direction["south"] or [None]
    wests = by_direction["west"] or [None]
    for north in norths:
        for east in easts:
            for south in souths:
                for west in wests:
                    combo = {
                        direction: label
                        for direction, label in {
                            "north": north,
                            "east": east,
                            "south": south,
                            "west": west,
                        }.items()
                        if label is not None
                    }
                    if len(combo) >= 3:
                        combos.append(combo)
    return combos


def compass_label_score(labels: dict[str, OcrText]) -> float:
    center = compass_center(labels)
    score = float(len(labels)) * 100.0 + sum(label.score * 10 for label in labels.values())

    for direction, label in labels.items():
        dx = label.cx - center[0]
        dy = label.cy - center[1]
        distance = math.hypot(dx, dy)
        if distance < 12:
            score -= 100.0

        if direction == "north":
            score += -dy if dy < 0 else -80.0
            score -= abs(dx) * 0.25
        elif direction == "south":
            score += dy if dy > 0 else -80.0
            score -= abs(dx) * 0.25
        elif direction == "east":
            score += dx if dx > 0 else -80.0
            score -= abs(dy) * 0.25
        elif direction == "west":
            score += -dx if dx < 0 else -80.0
            score -= abs(dy) * 0.25

    return score


def compass_center(labels: dict[str, OcrText]) -> tuple[int, int]:
    if "east" in labels and "west" in labels:
        center_x = (labels["east"].cx + labels["west"].cx) / 2
    else:
        center_x = sum(label.cx for label in labels.values()) / len(labels)

    if "north" in labels and "south" in labels:
        center_y = (labels["north"].cy + labels["south"].cy) / 2
    else:
        center_y = sum(label.cy for label in labels.values()) / len(labels)

    return round(center_x), round(center_y)


def compass_radius(labels: dict[str, OcrText], center: tuple[int, int]) -> int:
    distances = [math.dist((label.cx, label.cy), center) for label in labels.values()]
    if not distances:
        return 80
    return max(30, min(220, round(max(distances) * 1.35)))


def find_compass_circle(
    image: Image.Image,
    region: tuple[int, int, int, int],
) -> tuple[tuple[int, int], int]:
    crop = np.array(image.crop(region).convert("RGB"))
    gray = cv2_gray(crop)
    blur = cv2_median_blur(gray, 5)
    min_radius = max(20, round(min(crop.shape[0], crop.shape[1]) * 0.28))
    max_radius = max(min_radius + 1, round(min(crop.shape[0], crop.shape[1]) * 0.50))
    circles = cv2_hough_circles(blur, min_radius, max_radius)
    if circles is None:
        center = ((region[0] + region[2]) // 2, (region[1] + region[3]) // 2)
        radius = round(min(region[2] - region[0], region[3] - region[1]) * 0.40)
        return center, radius

    candidates = sorted(circles, key=lambda item: circle_score(item, crop.shape), reverse=True)
    x, y, radius = candidates[0]
    return (region[0] + round(float(x)), region[1] + round(float(y))), round(float(radius))


def find_compass_circle_auto(image: Image.Image) -> tuple[tuple[int, int], int, tuple[int, int, int, int]]:
    rgb = np.array(image.convert("RGB"))
    gray = cv2_gray(rgb)
    blur = cv2_median_blur(gray, 5)
    min_radius = int(os.environ.get("WURM_COMPASS_AUTO_MIN_RADIUS", "45"))
    max_radius = int(os.environ.get("WURM_COMPASS_AUTO_MAX_RADIUS", "90"))
    circles = cv2_hough_circles(blur, min_radius, max_radius, param2=32)
    if circles is None:
        circles = cv2_hough_circles(blur, min_radius, max_radius, param2=26)
    if circles is None:
        raise RuntimeError("Compass circle was not found")

    scored = sorted(
        ((compass_candidate_score(rgb, circle), circle) for circle in circles),
        key=lambda item: item[0],
        reverse=True,
    )
    score, best = scored[0]
    if score < 250:
        raise RuntimeError(f"Compass circle candidates were weak; best score={score:.1f}")

    x, y, radius = best
    center = round(float(x)), round(float(y))
    radius_value = round(float(radius))
    margin = max(12, round(radius_value * 1.20))
    region = clamp_box(
        (
            center[0] - margin,
            center[1] - margin,
            center[0] + margin,
            center[1] + margin,
        ),
        image.size,
    )
    return center, radius_value, region


def compass_candidate_score(rgb: np.ndarray, circle: np.ndarray) -> float:
    x, y, radius = (float(value) for value in circle)
    height, width = rgb.shape[:2]
    x1 = max(0, int(x - radius * 1.10))
    y1 = max(0, int(y - radius * 1.10))
    x2 = min(width, int(x + radius * 1.10))
    y2 = min(height, int(y + radius * 1.10))
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return -1000.0

    red = crop[:, :, 0].astype(np.int16)
    green = crop[:, :, 1].astype(np.int16)
    blue = crop[:, :, 2].astype(np.int16)
    ys, xs = np.indices(crop.shape[:2])
    cx = x - x1
    cy = y - y1
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    brightness = (red + green + blue) / 3
    saturation = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])

    rim = (dist > radius * 0.75) & (dist < radius * 1.05)
    inner = dist < radius * 0.65
    warm_rim = (
        (red > 70)
        & (red > green + 8)
        & (green >= blue - 2)
        & (brightness > 55)
        & (saturation > 12)
        & (saturation < 100)
    )
    gold_inner = (
        (red > 60)
        & (green > 35)
        & (blue < 85)
        & (red >= green)
        & (green >= blue - 5)
        & inner
    )
    red_needle = (
        (red > 105)
        & (red > green + 35)
        & (red > blue + 45)
        & (dist > radius * 0.25)
        & (dist < radius * 0.82)
    )

    rim_score = float((warm_rim & rim).sum()) / max(1, int(rim.sum()))
    inner_score = float(gold_inner.sum()) / max(1, int(inner.sum()))
    red_score = min(120, int(red_needle.sum()))
    radius_penalty = abs(radius - 63.0) * 3.0
    return rim_score * 1000.0 + inner_score * 400.0 + red_score - radius_penalty


def circle_score(circle: np.ndarray, crop_shape: tuple[int, ...]) -> float:
    height, width = crop_shape[:2]
    x, y, radius = circle
    crop_center_x = width / 2
    crop_center_y = height / 2
    distance_from_region_center = math.hypot(float(x) - crop_center_x, float(y) - crop_center_y)
    expected_radius = min(width, height) * 0.40
    return 1000.0 - distance_from_region_center * 3.0 - abs(float(radius) - expected_radius)


def cv2_gray(rgb_array: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)


def cv2_median_blur(gray_array: np.ndarray, kernel_size: int) -> np.ndarray:
    import cv2

    return cv2.medianBlur(gray_array, kernel_size)


def cv2_hough_circles(
    gray_array: np.ndarray,
    min_radius: int,
    max_radius: int,
    param2: int = 20,
) -> np.ndarray | None:
    import cv2

    circles = cv2.HoughCircles(
        gray_array,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=80,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None
    return circles[0]


def find_compass_axis(
    image: Image.Image,
    region: tuple[int, int, int, int],
    center: tuple[int, int],
    radius: int,
) -> tuple[int, int, int, int] | None:
    import cv2

    crop = np.array(image.crop(region).convert("RGB"))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    red = crop[:, :, 0].astype(np.int16)
    green = crop[:, :, 1].astype(np.int16)
    blue = crop[:, :, 2].astype(np.int16)
    brightness = (red + green + blue) / 3
    saturation = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
    height, width = crop.shape[:2]
    ys, xs = np.indices((height, width))
    local_center = (center[0] - region[0], center[1] - region[1])
    dist_from_center = np.sqrt((xs - local_center[0]) ** 2 + (ys - local_center[1]) ** 2)
    grey_needle = (brightness > 45) & (brightness < 170) & (saturation < 35) & (dist_from_center < radius * 0.80)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=max(25, round(radius * 0.55)),
        maxLineGap=8,
    )
    if lines is None:
        return None

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = (float(value) for value in line)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < radius * 0.60:
            continue

        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
        midpoint_distance = math.dist(midpoint, local_center)
        line_distance = point_to_line_distance(local_center, (x1, y1), (x2, y2))
        endpoint_distance_1 = math.dist((x1, y1), local_center)
        endpoint_distance_2 = math.dist((x2, y2), local_center)
        if midpoint_distance > radius * 0.45:
            continue
        if line_distance > radius * 0.22:
            continue
        if endpoint_distance_1 > radius * 0.95 or endpoint_distance_2 > radius * 0.95:
            continue

        tube = pixel_line_distances(xs, ys, tuple(int(value) for value in line)) < 5.0
        grey_count = int((grey_needle & tube).sum())
        score = length - midpoint_distance * 0.5 - line_distance * 4.0 + grey_count * 0.12
        candidates.append((score, tuple(int(value) for value in line)))

    if not candidates:
        return None

    _score, line = max(candidates, key=lambda item: item[0])
    x1, y1, x2, y2 = line
    return region[0] + x1, region[1] + y1, region[0] + x2, region[1] + y2


def point_to_line_distance(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return math.dist(point, line_start)
    return abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1)) / length


def find_red_arrow_tip(
    image: Image.Image,
    region: tuple[int, int, int, int],
    center: tuple[int, int],
    radius: int,
    axis: tuple[int, int, int, int] | None,
) -> tuple[int, int]:
    crop = np.array(image.crop(region).convert("RGB"))
    red = crop[:, :, 0].astype(np.int16)
    green = crop[:, :, 1].astype(np.int16)
    blue = crop[:, :, 2].astype(np.int16)
    height, width = crop.shape[:2]
    ys, xs = np.indices((height, width))
    local_center = (center[0] - region[0], center[1] - region[1])
    dist_from_center = np.sqrt((xs - local_center[0]) ** 2 + (ys - local_center[1]) ** 2)
    mask = (red > 80) & (red > green + 20) & (red > blue + 25) & (dist_from_center < radius * 0.80)

    if axis is not None:
        local_axis = (
            axis[0] - region[0],
            axis[1] - region[1],
            axis[2] - region[0],
            axis[3] - region[1],
        )
        line_distance = pixel_line_distances(xs, ys, local_axis)
        mask &= line_distance < max(7.0, radius * 0.12)

    candidate_ys, candidate_xs = np.nonzero(mask)
    if len(candidate_xs) == 0:
        return find_red_arrow(image, center, radius)

    distances = np.sqrt((candidate_xs - local_center[0]) ** 2 + (candidate_ys - local_center[1]) ** 2)
    index = int(np.argmax(distances))
    return region[0] + int(candidate_xs[index]), region[1] + int(candidate_ys[index])


def find_red_arrow_tip_by_radius(
    image: Image.Image,
    region: tuple[int, int, int, int],
    center: tuple[int, int],
    radius: int,
) -> tuple[int, int] | None:
    crop = np.array(image.crop(region).convert("RGB"))
    red = crop[:, :, 0].astype(np.int16)
    green = crop[:, :, 1].astype(np.int16)
    blue = crop[:, :, 2].astype(np.int16)
    height, width = crop.shape[:2]
    ys, xs = np.indices((height, width))
    local_center = (center[0] - region[0], center[1] - region[1])
    dist_from_center = np.sqrt((xs - local_center[0]) ** 2 + (ys - local_center[1]) ** 2)

    red_nose = (
        (red > 105)
        & (red > green + 35)
        & (red > blue + 45)
        & (dist_from_center > radius * 0.25)
        & (dist_from_center < radius * 0.82)
    )
    red_nose = morphology_open(red_nose, size=2)

    component = largest_component(red_nose, min_area=5)
    if component is None:
        return None

    candidate_ys, candidate_xs = component
    distances = np.sqrt((candidate_xs - local_center[0]) ** 2 + (candidate_ys - local_center[1]) ** 2)
    index = int(np.argmax(distances))
    return region[0] + int(candidate_xs[index]), region[1] + int(candidate_ys[index])


def morphology_open(mask: np.ndarray, size: int) -> np.ndarray:
    import cv2

    kernel = np.ones((size, size), np.uint8)
    opened = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    return opened > 0


def largest_component(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, np.ndarray] | None:
    import cv2

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    best_index: int | None = None
    best_area = 0
    for index in range(1, component_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_index = index
            best_area = area

    if best_index is None:
        return None

    return np.nonzero(labels == best_index)


def pixel_line_distances(
    xs: np.ndarray,
    ys: np.ndarray,
    line: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = line
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return np.sqrt((xs - x1) ** 2 + (ys - y1) ** 2)
    return np.abs((x2 - x1) * (y1 - ys) - (x1 - xs) * (y2 - y1)) / length


def find_red_arrow(image: Image.Image, center: tuple[int, int], radius: int) -> tuple[int, int]:
    cx, cy = center
    box = clamp_box((cx - radius, cy - radius, cx + radius, cy + radius), image.size)
    arr = np.array(image.crop(box).convert("RGB"))
    if arr.size == 0:
        raise RuntimeError("Compass red-arrow search box is empty")

    red = arr[:, :, 0].astype(np.int16)
    green = arr[:, :, 1].astype(np.int16)
    blue = arr[:, :, 2].astype(np.int16)
    mask = (red > 110) & (red > green + 35) & (red > blue + 35)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("Compass red arrow was not found")

    weights = red[ys, xs].astype(np.float64)
    arrow_x = int(round(float(np.average(xs, weights=weights)))) + box[0]
    arrow_y = int(round(float(np.average(ys, weights=weights)))) + box[1]
    return arrow_x, arrow_y


def heading_from_points(center: tuple[int, int], arrow: tuple[int, int]) -> str:
    dx = arrow[0] - center[0]
    dy = arrow[1] - center[1]
    if abs(dx) > abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


def direction_degrees_from_points(center: tuple[int, int], arrow: tuple[int, int]) -> int:
    dx = arrow[0] - center[0]
    dy = arrow[1] - center[1]
    degrees = math.degrees(math.atan2(dx, -dy))
    return round((degrees + 360.0) % 360.0)


def save_compass_diagnostic(reading: CompassReading, output_path: Path | None = None) -> Path:
    SCREENS_DIR.mkdir(exist_ok=True)
    if output_path is None:
        output_path = SCREENS_DIR / f"compass_{timestamp()}.png"

    out = reading.image.copy()
    draw = ImageDraw.Draw(out)
    font = diagnostic_font()
    if not reading.visible:
        draw.text((20, 20), "compass=hidden", fill="white", font=font)
        draw.text((20, 38), "no reliable circle while moving", fill="white", font=font)
        out.save(output_path)
        out.save(SCREENS_DIR / "compass_latest.png")
        return output_path

    assert reading.center is not None
    assert reading.radius is not None
    assert reading.region is not None
    cx, cy = reading.center
    x1, y1, x2, y2 = reading.region

    draw.rectangle((x1, y1, x2, y2), outline="yellow", width=3)
    draw.ellipse((cx - reading.radius, cy - reading.radius, cx + reading.radius, cy + reading.radius), outline="cyan", width=2)
    draw.line((cx - 18, cy, cx + 18, cy), fill="cyan", width=3)
    draw.line((cx, cy - 18, cx, cy + 18), fill="cyan", width=3)
    if reading.axis is not None:
        draw.line(reading.axis, fill="lime", width=3)
    if reading.arrow is not None:
        ax, ay = reading.arrow
        draw_arrow(draw, (cx, cy), (ax, ay), fill="red", width=4)
        draw.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), outline="red", width=3)

    for direction, label in reading.labels.items():
        draw.rectangle((label.x1, label.y1, label.x2, label.y2), outline="lime", width=2)
        draw.text((label.x1, max(0, label.y1 - 16)), direction, fill="lime", font=font)

    if reading.available:
        assert reading.arrow is not None
        assert reading.direction_degrees is not None
        ax, ay = reading.arrow
        draw.text((cx + 12, cy + 12), f"heading={reading.heading}", fill="white", font=font)
        draw.text((ax + 8, ay + 8), f"dir={reading.direction_degrees:03d} deg", fill="white", font=font)
    else:
        draw.text((cx + 12, cy + 12), "heading=unavailable", fill="white", font=font)
        draw.text((cx + 12, cy + 30), "compass is hidden while moving", fill="white", font=font)
    out.save(output_path)
    out.save(SCREENS_DIR / "compass_latest.png")
    return output_path


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: str,
    width: int,
) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    head_length = 18
    head_angle = math.radians(28)
    points = [end]
    for sign in (-1, 1):
        theta = angle + math.pi + sign * head_angle
        points.append((round(ex + head_length * math.cos(theta)), round(ey + head_length * math.sin(theta))))
    draw.polygon(points, fill=fill)


def diagnostic_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()
