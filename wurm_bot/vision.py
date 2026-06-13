from __future__ import annotations

from collections.abc import Callable
import re

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

import sxtemp1

from .config import MIN_ACTION_PIXELS, OCR_MIN_SCORE, ROW_HEIGHT
from .models import Candidate, OcrText, Table
from .text import normalize


def ocr_image(image: Image.Image) -> list[OcrText]:
    ocr = RapidOCR()
    result, _ = ocr(np.array(image))
    texts: list[OcrText] = []
    for box, text, score in result or []:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        if score_value < OCR_MIN_SCORE:
            continue

        xs = [int(point[0]) for point in box]
        ys = [int(point[1]) for point in box]
        texts.append(OcrText(str(text), score_value, min(xs), min(ys), max(xs), max(ys)))
    return texts


def scan() -> tuple[Image.Image, list[OcrText], list[Table], list[Candidate]]:
    image = sxtemp1.screenshot()
    from .upgrade_targets import detect_upgrade_targets

    detection = detect_upgrade_targets(image)
    return detection.image, detection.texts, detection.tables, detection.targets


def find_tables(texts: list[OcrText], image_size: tuple[int, int]) -> list[Table]:
    width, height = image_size
    names = [item for item in texts if normalize(item.text) == "name"]
    tables: list[Table] = []

    for name in names:
        nearby = [item for item in texts if abs(item.cy - name.cy) <= 10 and item.x1 > name.x2]
        ql_items = [item for item in nearby if normalize(item.text) == "ql"]
        weight_items = [item for item in nearby if "weight" in normalize(item.text)]
        dmg_or_merged = [item for item in nearby if "dmg" in normalize(item.text)]
        if not ql_items or not weight_items or not dmg_or_merged:
            continue

        ql = min(ql_items, key=lambda item: item.x1)
        weight = max(weight_items, key=lambda item: item.x2)
        title = find_table_title(texts, name)

        x1 = max(0, name.x1 - 36)
        x2 = min(width - 1, max(weight.x2 + 40, name.x2 + 260))
        y1 = max(0, name.y1 - 50)
        y2 = find_table_bottom(texts, name, x1, x2, height)

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

    return dedupe_tables(tables)


def find_table_title(texts: list[OcrText], header: OcrText) -> str:
    candidates = [
        item
        for item in texts
        if header.x1 - 160 <= item.x1 <= header.x2 + 120 and header.y1 - 90 <= item.cy <= header.y1 - 8
    ]
    if not candidates:
        return ""
    return max(candidates, key=lambda item: (item.y1, item.score)).text


def find_table_bottom(texts: list[OcrText], header: OcrText, x1: int, x2: int, image_height: int) -> int:
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
        if kept and item.cy - last_y > 55:
            break
        kept.append(item)
        last_y = item.cy

    return min(image_height - 1, max(item.y2 for item in kept) + 42)


def dedupe_tables(tables: list[Table]) -> list[Table]:
    unique: list[Table] = []
    for table in sorted(tables, key=lambda item: (item.y1, item.x1)):
        if any(abs(table.x1 - old.x1) < 20 and abs(table.header_y - old.header_y) < 10 for old in unique):
            continue
        unique.append(table)
    return unique


def table_is_target_container(table: Table) -> bool:
    title = normalize(table.title)
    if not title:
        return False
    blocked = ("inventory", "toolbelt", "backpack")
    return not any(word in title for word in blocked)


def table_is_inventory(table: Table) -> bool:
    return "inventory" in normalize(table.title)


def row_name_like(text: str) -> bool:
    cleaned = normalize(text)
    if not cleaned:
        return False
    if cleaned in {"name", "ql", "dmg", "weight"}:
        return False
    if cleaned.startswith(("filter", "activated:", "ctrl+")):
        return False
    if re.fullmatch(r"[0-9., ]+", cleaned):
        return False
    return bool(re.search(r"[a-z]", cleaned))


def action_pixel_count(image: Image.Image, table: Table, y: int) -> int:
    arr = np.array(image.convert("RGB"))
    patch_x1 = max(table.x1, table.x2 - 26)
    patch_x2 = min(table.x2, table.x2 - 18)
    patch_y1 = max(table.y1, y - 8)
    patch_y2 = min(table.y2, y + 9)
    patch = arr[patch_y1:patch_y2, patch_x1:patch_x2]
    if patch.size == 0:
        return 0

    sample_y = patch.shape[0] // 2
    sample = patch[max(0, sample_y - 1) : min(patch.shape[0], sample_y + 2), 0:3]
    background = np.median(sample.reshape(-1, 3), axis=0)
    diff = np.abs(patch.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    return int(np.count_nonzero(diff > 18))


def text_rows_for_table(texts: list[OcrText], table: Table) -> list[OcrText]:
    rows = []
    for item in texts:
        if item.cy <= table.header_y + 14:
            continue
        if not (table.x1 <= item.cx <= table.x2 and table.y1 <= item.cy <= table.y2):
            continue
        if item.x1 >= table.ql_x1 - 22:
            continue
        if not row_name_like(item.text):
            continue
        rows.append(item)
    return sorted(rows, key=lambda item: item.cy)


def find_candidates(image: Image.Image, texts: list[OcrText], tables: list[Table]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for table in tables:
        if not table_is_target_container(table):
            continue

        for row in text_rows_for_table(texts, table):
            pixels = action_pixel_count(image, table, row.cy)
            if pixels < MIN_ACTION_PIXELS:
                continue

            y1 = max(table.y1, row.cy - ROW_HEIGHT // 2)
            y2 = min(table.y2, row.cy + ROW_HEIGHT // 2)
            candidates.append(
                Candidate(
                    table=table,
                    name=row.text,
                    x1=table.x1 + 4,
                    y1=y1,
                    x2=table.x2 - 4,
                    y2=y2,
                    click_x=max(table.x1 + 10, row.cx),
                    click_y=row.cy,
                    action_pixels=pixels,
                )
            )

    return dedupe_candidates(candidates)


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    unique: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.y1, item.x1)):
        if any(abs(candidate.click_y - old.click_y) < 8 and abs(candidate.x1 - old.x1) < 20 for old in unique):
            continue
        unique.append(candidate)
    return unique


def find_inventory_rows(
    texts: list[OcrText],
    tables: list[Table],
    predicate: Callable[[str], bool],
) -> list[OcrText]:
    inventory_tables = [table for table in tables if table_is_inventory(table)]
    rows: list[OcrText] = []
    for table in inventory_tables:
        for row in texts:
            if row.cy <= table.header_y + 14:
                continue
            if not (table.x1 <= row.cx <= table.x2 and table.y1 <= row.cy <= table.y2):
                continue
            if predicate(row.text):
                rows.append(row)
    return sorted(rows, key=lambda item: item.cy)


def find_log_rows(texts: list[OcrText], tables: list[Table]) -> list[OcrText]:
    return find_inventory_rows(texts, tables, text_is_log_row)


def is_inventory_item_active(texts: list[OcrText], predicate: Callable[[str], bool]) -> bool:
    return any("activated:" in normalize(item.text) and predicate(item.text) for item in texts)


def is_log_active(texts: list[OcrText]) -> bool:
    return is_inventory_item_active(texts, text_is_log_row)


def text_is_log_row(text: str) -> bool:
    cleaned = normalize(text)
    compact = cleaned.replace(" ", "").replace(",", "")
    compact = compact.replace("0", "o").replace("1", "l").replace("|", "l")
    return "log" in compact
