from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import SCREENS_DIR
from .models import OcrText
from .text import timestamp


WINDOW_GROUP_INVENTORY = "inventory"
WINDOW_GROUP_TOOLBELT = "toolbelt"
WINDOW_GROUP_OTHER = "other"
WINDOW_GROUP_LABELS = {
    WINDOW_GROUP_INVENTORY: "inventory",
    WINDOW_GROUP_TOOLBELT: "toolbelt",
    WINDOW_GROUP_OTHER: "other",
}
WINDOW_GROUP_COLORS = {
    WINDOW_GROUP_INVENTORY: "magenta",
    WINDOW_GROUP_TOOLBELT: "deepskyblue",
    WINDOW_GROUP_OTHER: "yellow",
}
TEMPLATE_DIR = Path(__file__).resolve().parent / "assets" / "inventory_templates"
MATCH_STRICTNESS = 0.99
TEMPLATE_SCALE = 0.5
TOP_LINE_TOLERANCE = 10
RIGHT_EDGE_TOLERANCE = 24
MIN_WINDOW_WIDTH = 120
MIN_WINDOW_HEIGHT = 90
VERTEX_STYLES = {
    "top_left": ("left_border.png", "lime"),
    "top_right": ("top_right.png", "red"),
    "bottom_right": ("bottom_right.png", "cyan"),
}


@dataclass(frozen=True)
class VertexMatch:
    kind: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


@dataclass(frozen=True)
class WindowRectangle:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    top_left: VertexMatch
    top_right: VertexMatch | None
    bottom_right: VertexMatch
    title: str = ""
    title_score: float = 0.0
    title_box: tuple[int, int, int, int] | None = None
    group: str = WINDOW_GROUP_OTHER


@dataclass(frozen=True)
class TemplateProfile:
    width: int
    height: int
    mask_x1: int
    mask_y1: int
    mask_x2: int
    mask_y2: int
    mask_area: int

    @property
    def mask_width(self) -> int:
        return self.mask_x2 - self.mask_x1

    @property
    def mask_height(self) -> int:
        return self.mask_y2 - self.mask_y1


def detect_inventory_vertices(image: Image.Image) -> list[VertexMatch]:
    image = image.convert("RGB")
    arr = np.array(image)
    matches: list[VertexMatch] = []

    for kind, (template_name, _draw_color) in VERTEX_STYLES.items():
        template_arr = np.array(Image.open(TEMPLATE_DIR / template_name).convert("RGB"))
        matches.extend(matches_from_template(kind, arr, template_arr))

    return sorted(non_max_suppression(matches, 0.25), key=lambda item: (item.y1, item.x1, item.kind))


def matches_from_template(kind: str, image_arr: np.ndarray, template_arr: np.ndarray) -> list[VertexMatch]:
    if kind == "top_left":
        return left_border_template_matches(kind, image_arr, template_arr)
    if kind == "top_right":
        return grayscale_template_matches(kind, image_arr, template_arr)
    if kind == "bottom_right":
        return masked_vertex_matches(kind, image_arr, template_arr)
    raise ValueError(f"Unknown vertex kind: {kind}")


def left_border_template_matches(
    kind: str,
    image_arr: np.ndarray,
    template_arr: np.ndarray,
) -> list[VertexMatch]:
    haystack = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    template = cv2.cvtColor(template_arr, cv2.COLOR_RGB2GRAY)
    template = scaled_image(template, TEMPLATE_SCALE, cv2.INTER_AREA)
    result = cv2.matchTemplate(haystack, template, cv2.TM_CCORR_NORMED)
    return matches_from_result(kind, result, template.shape[1], template.shape[0], 0, 0)


def grayscale_template_matches(
    kind: str,
    image_arr: np.ndarray,
    template_arr: np.ndarray,
) -> list[VertexMatch]:
    haystack = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    template = cv2.cvtColor(template_arr, cv2.COLOR_RGB2GRAY)
    template = scaled_image(template, TEMPLATE_SCALE, cv2.INTER_AREA)
    result = cv2.matchTemplate(haystack, template, cv2.TM_CCOEFF_NORMED)
    return matches_from_result(kind, result, template.shape[1], template.shape[0], 0, 0)


def masked_vertex_matches(
    kind: str,
    image_arr: np.ndarray,
    template_arr: np.ndarray,
) -> list[VertexMatch]:
    image_mask = vertex_mask(kind, image_arr)
    template_mask = vertex_mask(kind, template_arr)
    profile = template_profile(template_arr, template_mask)
    template_mask = template_mask[profile.mask_y1 : profile.mask_y2, profile.mask_x1 : profile.mask_x2]
    template_mask = scaled_image(template_mask, TEMPLATE_SCALE, cv2.INTER_NEAREST)

    result = cv2.matchTemplate(image_mask, template_mask, cv2.TM_CCORR_NORMED)
    scaled_width = int(profile.width * TEMPLATE_SCALE)
    scaled_height = int(profile.height * TEMPLATE_SCALE)
    x_offset = int(profile.mask_x1 * TEMPLATE_SCALE)
    y_offset = int(profile.mask_y1 * TEMPLATE_SCALE)
    return matches_from_result(kind, result, scaled_width, scaled_height, x_offset, y_offset)


def scaled_image(image: np.ndarray, scale: float, interpolation: int) -> np.ndarray:
    height, width = image.shape[:2]
    scaled_width = max(1, int(width * scale))
    scaled_height = max(1, int(height * scale))
    return cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)


def matches_from_result(
    kind: str,
    result: np.ndarray,
    width: int,
    height: int,
    x_offset: int,
    y_offset: int,
) -> list[VertexMatch]:
    ys, xs = np.where(np.isfinite(result) & (result >= MATCH_STRICTNESS))
    return [
        VertexMatch(
            kind=kind,
            x1=int(x - x_offset),
            y1=int(y - y_offset),
            x2=int(x - x_offset + width),
            y2=int(y - y_offset + height),
            score=float(result[y, x]),
        )
        for y, x in zip(ys, xs)
    ]


def assemble_window_rectangles(matches: list[VertexMatch]) -> list[WindowRectangle]:
    top_lefts = [match for match in matches if match.kind == "top_left"]
    top_rights = [match for match in matches if match.kind == "top_right"]
    bottom_rights = [match for match in matches if match.kind == "bottom_right"]

    rectangles: list[WindowRectangle] = []
    for top_left in top_lefts:
        found_rectangle = False
        for top_right in sorted(top_rights, key=lambda item: item.x1):
            if not top_right_matches(top_left, top_right):
                continue

            bottom_right = best_bottom_right(top_left, top_right, bottom_rights)
            if bottom_right is None:
                continue

            rectangles.append(make_rectangle(top_left, top_right, bottom_right))
            found_rectangle = True
            break

        if found_rectangle:
            continue

        bottom_right = best_bottom_right_without_top_right(top_left, bottom_rights)
        if bottom_right is not None:
            rectangles.append(make_rectangle(top_left, None, bottom_right))

    return sorted(non_max_suppression_rectangles(rectangles, 0.25), key=lambda item: (item.y1, item.x1))


def top_right_matches(top_left: VertexMatch, top_right: VertexMatch) -> bool:
    if abs(top_left.y1 - top_right.y1) > TOP_LINE_TOLERANCE:
        return False
    return top_right.x2 - top_left.x1 >= MIN_WINDOW_WIDTH


def best_bottom_right(
    top_left: VertexMatch,
    top_right: VertexMatch,
    bottom_rights: list[VertexMatch],
) -> VertexMatch | None:
    top_y = min(top_left.y1, top_right.y1)
    candidates = [
        bottom_right
        for bottom_right in bottom_rights
        if abs(bottom_right.x2 - top_right.x2) <= RIGHT_EDGE_TOLERANCE
        and bottom_right.y2 - top_y >= MIN_WINDOW_HEIGHT
    ]
    if not candidates:
        return None

    return min(candidates, key=lambda item: (item.y2, abs(item.x2 - top_right.x2)))


def best_bottom_right_without_top_right(
    top_left: VertexMatch,
    bottom_rights: list[VertexMatch],
) -> VertexMatch | None:
    candidates = [
        bottom_right
        for bottom_right in bottom_rights
        if bottom_right.x2 - top_left.x1 >= MIN_WINDOW_WIDTH
        and bottom_right.y2 - top_left.y1 >= MIN_WINDOW_HEIGHT
    ]
    if not candidates:
        return None

    return min(candidates, key=lambda item: (item.x2, item.y2))


def make_rectangle(
    top_left: VertexMatch,
    top_right: VertexMatch | None,
    bottom_right: VertexMatch,
) -> WindowRectangle:
    x1 = top_left.x1
    y1 = min(top_left.y1, top_right.y1) if top_right is not None else top_left.y1
    x2 = max(top_right.x2, bottom_right.x2) if top_right is not None else bottom_right.x2
    y2 = bottom_right.y2
    scores = [top_left.score, bottom_right.score]
    if top_right is not None:
        scores.append(top_right.score)
    return WindowRectangle(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=min(scores),
        top_left=top_left,
        top_right=top_right,
        bottom_right=bottom_right,
    )


def recognize_window_titles(image: Image.Image, rectangles: list[WindowRectangle]) -> list[WindowRectangle]:
    from .vision import ocr_image

    return assign_window_titles(rectangles, ocr_image(image.convert("RGB")))


def assign_window_titles(
    rectangles: list[WindowRectangle],
    texts: list[OcrText],
) -> list[WindowRectangle]:
    return [rectangle_with_title(rectangle, texts) for rectangle in rectangles]


def rectangle_with_title(rectangle: WindowRectangle, texts: list[OcrText]) -> WindowRectangle:
    candidates = title_candidates(rectangle, texts)
    if not candidates:
        return rectangle

    title = best_title_text(rectangle, candidates)
    return replace(
        rectangle,
        title=title.text.strip(),
        title_score=title.score,
        title_box=(title.x1, title.y1, title.x2, title.y2),
        group=classify_window_title(title.text),
    )


def title_candidates(rectangle: WindowRectangle, texts: list[OcrText]) -> list[OcrText]:
    title_x1 = rectangle.x1 + 6
    title_x2 = min(rectangle.x2 - 48, rectangle.x1 + 260)
    title_y1 = rectangle.y1 - 6
    title_y2 = rectangle.y1 + 46
    return [
        text
        for text in texts
        if title_x1 <= text.cx <= title_x2
        and title_y1 <= text.cy <= title_y2
        and title_text_like(text.text)
    ]


def best_title_text(rectangle: WindowRectangle, candidates: list[OcrText]) -> OcrText:
    return min(candidates, key=lambda item: (abs(item.x1 - (rectangle.x1 + 14)), item.y1, -item.score))


def title_text_like(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text.strip())
    normalized = cleaned.lower()
    if not normalized:
        return False
    if normalized in {"name", "ql", "dmg", "weight", "filter", "activated:"}:
        return False
    if normalized.replace(" ", "") in {"dmgweight", "ql", "name", "weight", "dmg"}:
        return False
    if normalized.startswith(("filter", "activated:", "ctrl+")):
        return False
    if re.fullmatch(r"[0-9.,: ]+", normalized):
        return False
    return bool(re.search(r"[a-z]", normalized))


def classify_window_title(title: str) -> str:
    normalized = re.sub(r"\s+", " ", title.strip()).lower()
    if normalized == "inventory":
        return WINDOW_GROUP_INVENTORY
    if normalized == "toolbelt":
        return WINDOW_GROUP_TOOLBELT
    return WINDOW_GROUP_OTHER


def group_window_rectangles(rectangles: list[WindowRectangle]) -> dict[str, list[WindowRectangle]]:
    return {
        group: [rectangle for rectangle in rectangles if rectangle.group == group]
        for group in (WINDOW_GROUP_INVENTORY, WINDOW_GROUP_TOOLBELT, WINDOW_GROUP_OTHER)
    }


def vertex_mask(kind: str, arr: np.ndarray) -> np.ndarray:
    red = arr[:, :, 0].astype(np.float32)
    green = arr[:, :, 1].astype(np.float32)
    blue = arr[:, :, 2].astype(np.float32)

    if kind == "top_right":
        mask = (red > 120) & (red > green * 1.45) & (red > blue * 1.45)
    elif kind == "bottom_right":
        mask = (blue > 70) & (blue > red * 1.05) & (blue > green * 1.05)
    else:
        raise ValueError(f"Unknown vertex kind: {kind}")

    return mask.astype(np.uint8) * 255


def template_profile(template_arr: np.ndarray, mask: np.ndarray) -> TemplateProfile:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Template does not contain detectable vertex-colored pixels")

    return TemplateProfile(
        width=template_arr.shape[1],
        height=template_arr.shape[0],
        mask_x1=int(xs.min()),
        mask_y1=int(ys.min()),
        mask_x2=int(xs.max()) + 1,
        mask_y2=int(ys.max()) + 1,
        mask_area=int(len(xs)),
    )


def non_max_suppression(matches: list[VertexMatch], overlap_threshold: float) -> list[VertexMatch]:
    kept: list[VertexMatch] = []
    for match in sorted(matches, key=lambda item: item.score, reverse=True):
        if any(match.kind == old.kind and iou(match, old) >= overlap_threshold for old in kept):
            continue
        kept.append(match)
    return kept


def iou(a: VertexMatch, b: VertexMatch) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    a_area = (a.x2 - a.x1) * (a.y2 - a.y1)
    b_area = (b.x2 - b.x1) * (b.y2 - b.y1)
    return intersection / float(a_area + b_area - intersection)


def non_max_suppression_rectangles(
    rectangles: list[WindowRectangle],
    overlap_threshold: float,
) -> list[WindowRectangle]:
    kept: list[WindowRectangle] = []
    for rectangle in sorted(rectangles, key=lambda item: item.score, reverse=True):
        if any(rectangle_iou(rectangle, old) >= overlap_threshold for old in kept):
            continue
        kept.append(rectangle)
    return kept


def rectangle_iou(a: WindowRectangle, b: WindowRectangle) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    a_area = (a.x2 - a.x1) * (a.y2 - a.y1)
    b_area = (b.x2 - b.x1) * (b.y2 - b.y1)
    return intersection / float(a_area + b_area - intersection)


def save_vertex_overlay(
    image: Image.Image,
    matches: list[VertexMatch],
    output_path: Path | None = None,
    rectangles: list[WindowRectangle] | None = None,
) -> Path:
    SCREENS_DIR.mkdir(exist_ok=True)
    if output_path is None:
        output_path = SCREENS_DIR / f"inventory_vertices_{timestamp()}.png"

    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font()

    for index, rectangle in enumerate(rectangles or [], start=1):
        title = rectangle.title.strip() or "unknown"
        group = WINDOW_GROUP_LABELS.get(rectangle.group, WINDOW_GROUP_OTHER)
        color = WINDOW_GROUP_COLORS.get(rectangle.group, "yellow")
        label = f"{index}: {group}: {title} {rectangle.score:.2f}"
        draw.rectangle((rectangle.x1, rectangle.y1, rectangle.x2, rectangle.y2), outline=color, width=4)
        if rectangle.title_box is not None:
            draw.rectangle(rectangle.title_box, outline="white", width=2)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        label_box = (
            rectangle.x1,
            rectangle.y1,
            rectangle.x1 + 8 + text_bbox[2] - text_bbox[0],
            rectangle.y1 + 18,
        )
        draw.rectangle(label_box, fill=(0, 0, 0))
        draw.text((label_box[0] + 4, label_box[1] + 2), label, fill=color, font=font)

    for index, match in enumerate(matches, start=1):
        _template_name, color = VERTEX_STYLES[match.kind]
        label = f"{index}: {match.kind} {match.score:.2f}"
        draw.rectangle((match.x1, match.y1, match.x2, match.y2), outline=color, width=3)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        label_box = (
            match.x1,
            max(0, match.y1 - 18),
            match.x1 + 8 + text_bbox[2] - text_bbox[0],
            max(16, match.y1 - 2),
        )
        draw.rectangle(label_box, fill=(0, 0, 0))
        draw.text((label_box[0] + 4, label_box[1] + 2), label, fill=color, font=font)

    out.save(output_path)
    out.save(SCREENS_DIR / "inventory_vertices_latest.png")
    if rectangles:
        out.save(SCREENS_DIR / "inventory_rectangles_latest.png")
    return output_path


def _font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        return ImageFont.load_default()
