from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wurm_bot.config import SCREENS_DIR
    from wurm_bot.models import OcrText, Table
    from wurm_bot.text import normalize, timestamp
    from wurm_bot.vision import ocr_image
else:
    from .config import SCREENS_DIR
    from .models import OcrText, Table
    from .text import normalize, timestamp
    from .vision import ocr_image


TEMPLATE_DIR = Path(__file__).resolve().parent / "assets" / "inventory_templates"


@dataclass(frozen=True)
class InventoryWindow:
    title: str
    x1: int
    y1: int
    x2: int
    y2: int
    table: Table


@dataclass(frozen=True)
class InventoryDetection:
    image: Image.Image
    texts: list[OcrText]
    inventories: list[InventoryWindow]


@dataclass(frozen=True)
class TemplateMatch:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


def detect_inventories(image: Image.Image) -> InventoryDetection:
    image = image.convert("RGB")
    texts = ocr_image(image)
    inventories = [
        refine_window_with_templates(expand_table_to_window(table, texts, image.size), image)
        for table in find_inventory_tables(texts, image.size)
    ]
    return InventoryDetection(image=image, texts=texts, inventories=inventories)


def find_inventory_tables(texts: list[OcrText], image_size: tuple[int, int]) -> list[Table]:
    width, height = image_size
    names = sorted([item for item in texts if normalize(item.text) == "name"], key=lambda item: (item.cy, item.x1))
    tables: list[Table] = []

    for name in names:
        right_limit = table_right_limit(name, names, width)
        nearby = [
            item
            for item in texts
            if abs(item.cy - name.cy) <= 10 and name.x2 < item.x1 < right_limit
        ]
        ql_items = [item for item in nearby if normalize(item.text) == "ql"]
        dmg_items = [item for item in nearby if "dmg" in normalize(item.text)]
        weight_items = [item for item in nearby if "weight" in normalize(item.text)]
        if not ql_items or not weight_items or not dmg_items:
            continue

        ql = min(ql_items, key=lambda item: item.x1)
        weight = max(weight_items, key=lambda item: item.x2)
        title = find_inventory_table_title(texts, name, right_limit)

        x1 = max(0, name.x1 - 36)
        x2 = min(width - 1, right_limit, max(weight.x2 + 40, name.x2 + 220))
        y1 = max(0, name.y1 - 50)
        y2 = find_inventory_table_bottom(texts, name, x1, x2, height)

        tables.append(
            Table(
                title=title,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                header_y=name.cy,
                name_x1=name.x1,
                ql_x1=ql.x1,
                weight_x2=weight.x2,
            )
        )

    return dedupe_inventory_tables(tables)


def table_right_limit(name: OcrText, names: list[OcrText], image_width: int) -> int:
    next_names = [
        item.x1 - 12
        for item in names
        if item.x1 > name.x1 and abs(item.cy - name.cy) <= 10
    ]
    return min(next_names) if next_names else image_width - 1


def find_inventory_table_title(texts: list[OcrText], header: OcrText, right_limit: int) -> str:
    candidates = [
        item
        for item in texts
        if header.x1 - 180 <= item.x1 <= min(right_limit, header.x2 + 140)
        and header.y1 - 90 <= item.cy <= header.y1 - 8
        and title_text_like(item.text)
    ]
    if not candidates:
        return ""

    return max(candidates, key=lambda item: (item.y1, item.score)).text


def title_text_like(text: str) -> bool:
    cleaned = normalize(text)
    if not cleaned:
        return False
    if cleaned in {"name", "ql", "dmg", "weight", "filter"}:
        return False
    if re.fullmatch(r"[0-9., ]+", cleaned):
        return False
    return bool(re.search(r"[a-z]", cleaned))


def find_inventory_table_bottom(texts: list[OcrText], header: OcrText, x1: int, x2: int, image_height: int) -> int:
    row_texts = sorted(
        [
            item
            for item in texts
            if x1 <= item.cx <= x2 and item.cy > header.cy + 8 and not normalize(item.text).startswith("filter")
        ],
        key=lambda item: item.cy,
    )
    if not row_texts:
        return min(image_height - 1, header.cy + 240)

    kept = []
    last_y = header.cy
    for item in row_texts:
        if not kept and item.cy - header.cy > 80:
            break
        if kept and item.cy - last_y > 55:
            break
        kept.append(item)
        last_y = item.cy

    if not kept:
        return min(image_height - 1, header.cy + 300)

    return min(image_height - 1, max(item.y2 for item in kept) + 42)


def dedupe_inventory_tables(tables: list[Table]) -> list[Table]:
    unique: list[Table] = []
    for table in sorted(tables, key=lambda item: (item.y1, item.x1)):
        if any(abs(table.x1 - old.x1) < 20 and abs(table.header_y - old.header_y) < 10 for old in unique):
            continue
        unique.append(table)
    return unique


def expand_table_to_window(
    table: Table,
    texts: list[OcrText],
    image_size: tuple[int, int],
) -> InventoryWindow:
    width, height = image_size
    title = find_table_title_text(table, texts)

    x1 = table.x1
    y1 = table.y1
    x2 = table.x2
    y2 = table.y2

    if title is not None:
        x1 = min(x1, title.x1 - 12)
        y1 = min(y1, title.y1 - 16)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2 + 12)
    y2 = min(height - 1, inventory_window_bottom(table, texts, x1, x2, height))
    return InventoryWindow(table.title, x1, y1, x2, y2, table)


def refine_window_with_templates(window: InventoryWindow, image: Image.Image) -> InventoryWindow:
    width, height = image.size
    top_right = find_template_in_box(
        image,
        "top_right.png",
        (
            max(0, window.x2 - 180),
            max(0, window.y1 - 35),
            min(width, window.x2 + 90),
            min(height, window.y1 + 110),
        ),
    )
    bottom_right = find_template_in_box(
        image,
        "bottom_right.png",
        (
            max(0, window.x2 - 140),
            max(0, window.y2 - 140),
            min(width, window.x2 + 90),
            min(height, window.y2 + 150),
        ),
    )
    left_border = find_template_in_box(
        image,
        "left_border.png",
        (
            max(0, window.x1 - 70),
            max(0, window.y1 - 20),
            min(width, window.x1 + 70),
            min(height, window.y2 + 40),
        ),
    )

    x1, y1, x2, y2 = window.x1, window.y1, window.x2, window.y2
    if left_border is not None:
        x1 = left_border.x1

    if top_right is not None:
        y1 = min(y1, top_right.y1)
        x2 = max(x2, top_right.x2)

    if bottom_right is not None:
        x2 = max(x2, bottom_right.x2)
        y2 = bottom_right.y2

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width - 1))
    y2 = max(y1 + 1, min(y2, height - 1))
    return InventoryWindow(window.title, x1, y1, x2, y2, window.table)


def find_template_in_box(
    image: Image.Image,
    template_name: str,
    search_box: tuple[int, int, int, int],
) -> TemplateMatch | None:
    template_path = TEMPLATE_DIR / template_name
    if not template_path.exists():
        return None

    if search_box[2] <= search_box[0] or search_box[3] <= search_box[1]:
        return None

    haystack = template_image_array(image.crop(search_box))
    template = template_image_array(Image.open(template_path))
    best = best_scaled_template_match(haystack, template)
    if best is None:
        return None

    max_score, max_loc, template_width, template_height = best
    if max_score < template_threshold(template_name):
        return None

    x1 = search_box[0] + max_loc[0]
    y1 = search_box[1] + max_loc[1]
    return TemplateMatch(
        x1=x1,
        y1=y1,
        x2=x1 + template_width,
        y2=y1 + template_height,
        score=float(max_score),
    )


def find_template_near_window(
    image: Image.Image,
    window: InventoryWindow,
    template_name: str,
    search_margin: tuple[int, int, int, int],
) -> TemplateMatch | None:
    left_margin, top_margin, right_margin, bottom_margin = search_margin
    width, height = image.size
    return find_template_in_box(
        image,
        template_name,
        (
            max(0, window.x1 - left_margin),
            max(0, window.y1 - top_margin),
            min(width, window.x2 + right_margin),
            min(height, window.y2 + bottom_margin),
        ),
    )


def template_image_array(image: Image.Image) -> np.ndarray:
    gray = np.array(image.convert("L"))
    return cv2.Canny(gray, 40, 120)


def best_scaled_template_match(
    haystack: np.ndarray,
    template: np.ndarray,
) -> tuple[float, tuple[int, int], int, int] | None:
    best: tuple[float, tuple[int, int], int, int] | None = None
    for scale in (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40):
        scaled_width = max(8, int(template.shape[1] * scale))
        scaled_height = max(8, int(template.shape[0] * scale))
        if scaled_width > haystack.shape[1] or scaled_height > haystack.shape[0]:
            continue

        scaled = cv2.resize(template, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(haystack, scaled, cv2.TM_CCOEFF_NORMED)
        _min_score, max_score, _min_loc, max_loc = cv2.minMaxLoc(result)
        candidate = (float(max_score), max_loc, scaled_width, scaled_height)
        if best is None or candidate[0] > best[0]:
            best = candidate

    return best


def template_threshold(template_name: str) -> float:
    if template_name == "left_border.png":
        return 0.28
    return 0.32



def find_table_title_text(table: Table, texts: list[OcrText]) -> OcrText | None:
    title = normalize(table.title)
    if not title:
        return None

    candidates = [
        item
        for item in texts
        if normalize(item.text) == title
        and table.x1 - 260 <= item.cx <= table.x2 + 40
        and table.y1 - 80 <= item.cy <= table.header_y + 8
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda item: (item.score, item.y1))


def inventory_window_bottom(table: Table, texts: list[OcrText], x1: int, x2: int, image_height: int) -> int:
    lower_panel_texts = [
        item
        for item in texts
        if x1 <= item.cx <= x2
        and table.y2 - 8 <= item.cy <= table.y2 + 120
        and normalize(item.text).startswith(("filter", "activated:"))
    ]
    if lower_panel_texts:
        return min(image_height - 1, max(item.y2 for item in lower_panel_texts) + 22)

    if not table_has_rows(table, texts):
        return min(image_height - 1, table.y2 + 20)

    return min(image_height - 1, table.y2 + 82)


def table_has_rows(table: Table, texts: list[OcrText]) -> bool:
    return any(
        table.x1 <= item.cx <= table.x2
        and table.header_y + 8 <= item.cy <= table.y2
        and not normalize(item.text).startswith("filter")
        for item in texts
    )


def save_inventory_overlay(
    detection: InventoryDetection,
    output_path: Path | None = None,
) -> Path:
    SCREENS_DIR.mkdir(exist_ok=True)
    if output_path is None:
        output_path = SCREENS_DIR / f"inventories_{timestamp()}.png"

    out = detection.image.copy()
    draw = ImageDraw.Draw(out)
    font = _font()

    for index, inventory in enumerate(detection.inventories, start=1):
        label = inventory.title.strip() or "inventory"
        text = f"{index}: {label}"
        draw.rectangle((inventory.x1, inventory.y1, inventory.x2, inventory.y2), outline="magenta", width=4)
        draw.line((inventory.table.x1, inventory.table.header_y, inventory.table.x2, inventory.table.header_y), fill="yellow", width=2)

        text_bbox = draw.textbbox((0, 0), text, font=font)
        label_box = (
            inventory.x1 + 4,
            max(0, inventory.y1 - 24),
            inventory.x1 + 12 + text_bbox[2] - text_bbox[0],
            max(18, inventory.y1 - 4),
        )
        draw.rectangle(label_box, fill=(0, 0, 0))
        draw.text((label_box[0] + 4, label_box[1] + 3), text, fill="magenta", font=font)

    out.save(output_path)
    out.save(SCREENS_DIR / "inventories_latest.png")
    return output_path


def _font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    import sxtemp1

    image = sxtemp1.screenshot()
    detection = detect_inventories(image)
    output = save_inventory_overlay(detection)
    print(f"Inventories found: {len(detection.inventories)}")
    for index, inventory in enumerate(detection.inventories, start=1):
        title = inventory.title.strip() or "inventory"
        print(f"{index}. {title}: ({inventory.x1}, {inventory.y1})-({inventory.x2}, {inventory.y2})")
    print(f"Overlay saved: {output}")


if __name__ == "__main__":
    main()
