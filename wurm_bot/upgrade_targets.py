from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wurm_bot.config import MIN_ACTION_PIXELS, ROW_HEIGHT, SCREENS_DIR
    from wurm_bot.inventory_vertices import (
        WINDOW_GROUP_OTHER,
        WindowRectangle,
        assemble_window_rectangles,
        assign_window_titles,
        detect_inventory_vertices,
    )
    from wurm_bot.models import Candidate, OcrText, Table, candidate_action_point
    from wurm_bot.text import normalize, timestamp
    from wurm_bot.vision import action_pixel_count, ocr_image, row_name_like, text_rows_for_table
else:
    from .config import MIN_ACTION_PIXELS, ROW_HEIGHT, SCREENS_DIR
    from .inventory_vertices import (
        WINDOW_GROUP_OTHER,
        WindowRectangle,
        assemble_window_rectangles,
        assign_window_titles,
        detect_inventory_vertices,
    )
    from .models import Candidate, OcrText, Table, candidate_action_point
    from .text import normalize, timestamp
    from .vision import action_pixel_count, ocr_image, row_name_like, text_rows_for_table


@dataclass(frozen=True)
class UpgradeTargetDetection:
    image: Image.Image
    texts: list[OcrText]
    windows: list[WindowRectangle]
    tables: list[Table]
    target_tables: list[Table]
    targets: list[Candidate]


def detect_upgrade_targets(image: Image.Image, debug: bool = False) -> UpgradeTargetDetection:
    image = image.convert("RGB")
    texts = ocr_image(image)
    vertices = detect_inventory_vertices(image)
    rectangles = assemble_window_rectangles(vertices)
    windows = assign_window_titles(rectangles, texts)
    tables = window_tables(windows, texts)
    target_tables = other_container_tables(windows, texts)
    targets = find_upgrade_targets(image, texts, target_tables)
    detection = UpgradeTargetDetection(
        image=image,
        texts=texts,
        windows=windows,
        tables=tables,
        target_tables=target_tables,
        targets=targets,
    )
    if debug:
        print_detection_debug(detection, vertices)
    return detection


def print_detection_debug(detection: UpgradeTargetDetection, vertices: list[object]) -> None:
    print("== upgrade target detection ==")
    print(f"image: {detection.image.size[0]}x{detection.image.size[1]}")
    print(f"ocr texts: {len(detection.texts)}")
    print(f"vertices: {len(vertices)}")

    vertex_counts: dict[str, int] = {}
    for vertex in vertices:
        kind = getattr(vertex, "kind", "unknown")
        vertex_counts[kind] = vertex_counts.get(kind, 0) + 1
    for kind, count in sorted(vertex_counts.items()):
        print(f"  vertex {kind}: {count}")
    for index, vertex in enumerate(vertices, start=1):
        kind = getattr(vertex, "kind", "unknown")
        score = getattr(vertex, "score", 0.0)
        x1 = getattr(vertex, "x1", 0)
        y1 = getattr(vertex, "y1", 0)
        x2 = getattr(vertex, "x2", 0)
        y2 = getattr(vertex, "y2", 0)
        print(f"  vertex {index}: kind={kind} box=({x1},{y1})-({x2},{y2}) score={score:.3f}")

    headers = [text for text in detection.texts if normalize(text.text) == "name"]
    print(f"name headers: {len(headers)}")
    for index, header in enumerate(sorted(headers, key=lambda item: (item.cy, item.x1)), start=1):
        print(
            f"  name header {index}: box=({header.x1},{header.y1})-({header.x2},{header.y2}) "
            f"center=({header.cx},{header.cy}) score={header.score:.2f}"
        )

    print(f"windows: {len(detection.windows)}")
    if not detection.windows and vertices:
        print("  no windows assembled from vertices")
    for index, window in enumerate(detection.windows, start=1):
        title = window.title.strip() or "<no title>"
        print(
            f"  window {index}: group={window.group} title={title!r} "
            f"box=({window.x1},{window.y1})-({window.x2},{window.y2}) score={window.score:.3f}"
        )

    print(f"tables: {len(detection.tables)}")
    for index, table in enumerate(detection.tables, start=1):
        rows = text_rows_for_table(detection.texts, table)
        print(
            f"  table {index}: title={table.title!r} box=({table.x1},{table.y1})-({table.x2},{table.y2}) "
            f"header_y={table.header_y} rows={len(rows)}"
        )
        for row in rows:
            pixels = action_pixel_count(detection.image, table, row.cy)
            keep = table in detection.target_tables and row_name_like(row.text) and pixels >= MIN_ACTION_PIXELS
            print(
                f"    row y={row.cy:<4} x=({row.x1},{row.x2}) pixels={pixels:<4} "
                f"keep={str(keep):<5} text={row.text!r}"
            )

    print(f"target tables: {len(detection.target_tables)}")
    for index, table in enumerate(detection.target_tables, start=1):
        print(f"  target table {index}: {table.title!r}")

    print(f"targets: {len(detection.targets)}")
    for index, target in enumerate(detection.targets, start=1):
        print(
            f"  target {index}: table={target.table.title!r} name={target.name!r} "
            f"click=({target.click_x},{target.click_y}) pixels={target.action_pixels}"
        )


def window_tables(windows: list[WindowRectangle], texts: list[OcrText]) -> list[Table]:
    tables: list[Table] = []
    for window in windows:
        table = table_for_window(window, texts)
        if table is not None:
            tables.append(table)
    return tables


def other_container_tables(windows: list[WindowRectangle], texts: list[OcrText]) -> list[Table]:
    tables: list[Table] = []
    for window in windows:
        if window.group != WINDOW_GROUP_OTHER:
            continue

        table = table_for_window(window, texts)
        if table is not None:
            tables.append(table)

    return tables


def table_for_window(window: WindowRectangle, texts: list[OcrText]) -> Table | None:
    headers = [
        text
        for text in texts
        if normalize(text.text) == "name"
        and window.x1 <= text.cx <= window.x2
        and window.y1 <= text.cy <= window.y2
    ]
    if not headers:
        return None

    for name in sorted(headers, key=lambda item: (item.y1, item.x1)):
        nearby = [
            text
            for text in texts
            if window.x1 <= text.cx <= window.x2
            and abs(text.cy - name.cy) <= 10
            and text.x1 > name.x2
        ]
        ql_items = [text for text in nearby if normalize(text.text) == "ql"]
        dmg_items = [text for text in nearby if "dmg" in normalize(text.text)]
        weight_items = [text for text in nearby if "weight" in normalize(text.text)]
        if not ql_items or not dmg_items or not weight_items:
            continue

        ql = min(ql_items, key=lambda item: item.x1)
        weight = max(weight_items, key=lambda item: item.x2)
        return Table(
            title=window.title.strip() or "other",
            x1=window.x1,
            y1=window.y1,
            x2=window.x2,
            y2=window.y2,
            header_y=name.cy,
            name_x1=name.x1,
            ql_x1=ql.x1,
            weight_x2=weight.x2,
        )

    return None


def find_upgrade_targets(
    image: Image.Image,
    texts: list[OcrText],
    tables: list[Table],
) -> list[Candidate]:
    targets: list[Candidate] = []
    for table in tables:
        for row in text_rows_for_table(texts, table):
            if not row_name_like(row.text):
                continue

            pixels = action_pixel_count(image, table, row.cy)
            if pixels < MIN_ACTION_PIXELS:
                continue

            y1 = max(table.y1, row.cy - ROW_HEIGHT // 2)
            y2 = min(table.y2, row.cy + ROW_HEIGHT // 2)
            targets.append(
                Candidate(
                    table=table,
                    name=clean_target_name(row.text),
                    x1=table.x1 + 4,
                    y1=y1,
                    x2=table.x2 - 4,
                    y2=y2,
                    click_x=max(table.x1 + 10, row.cx),
                    click_y=row.cy,
                    action_pixels=pixels,
                )
            )

    return dedupe_targets(targets)


def clean_target_name(text: str) -> str:
    return text.strip().lstrip("?:+*->\"' \u56fd\u65e5\u81ea\u7530\u62ff\u8eab").strip()


def dedupe_targets(targets: list[Candidate]) -> list[Candidate]:
    unique: list[Candidate] = []
    for target in sorted(targets, key=lambda item: (item.table.title, item.y1, item.x1)):
        if any(abs(target.click_y - old.click_y) < 8 and abs(target.x1 - old.x1) < 20 for old in unique):
            continue
        unique.append(target)
    return unique


def save_upgrade_target_overlay(
    detection: UpgradeTargetDetection,
    output_path: Path | None = None,
) -> Path:
    SCREENS_DIR.mkdir(exist_ok=True)
    if output_path is None:
        output_path = SCREENS_DIR / f"upgrade_targets_{timestamp()}.png"

    out = detection.image.copy()
    draw = ImageDraw.Draw(out)
    font = _font()

    for window in detection.windows:
        if window.group != WINDOW_GROUP_OTHER:
            continue
        label = window.title.strip() or "other"
        draw.rectangle((window.x1, window.y1, window.x2, window.y2), outline="yellow", width=3)
        draw.text((window.x1 + 6, window.y1 + 4), label, fill="yellow", font=font)

    for index, target in enumerate(detection.targets, start=1):
        x, y = candidate_action_point(target)
        label = f"{index}: {target.table.title}: {target.name[:34]} ({target.action_pixels})"
        draw.rectangle((target.x1, target.y1, target.x2, target.y2), outline="lime", width=3)
        draw.line((x - 14, y, x + 14, y), fill="red", width=2)
        draw.line((x, y - 14, x, y + 14), fill="red", width=2)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        label_box = (
            target.x1 + 4,
            target.y1 + 2,
            target.x1 + 10 + text_bbox[2] - text_bbox[0],
            target.y1 + 6 + text_bbox[3] - text_bbox[1],
        )
        draw.rectangle(label_box, fill=(0, 0, 0))
        draw.text((target.x1 + 6, target.y1 + 3), label, fill="lime", font=font)

    out.save(output_path)
    out.save(SCREENS_DIR / "upgrade_targets_latest.png")
    return output_path


def _font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    import sxtemp1

    image = sxtemp1.screenshot()
    detection = detect_upgrade_targets(image, debug=True)
    output = save_upgrade_target_overlay(detection)
    print(f"Windows found: {len(detection.windows)}")
    print(f"Tables found: {len(detection.tables)}")
    print(f"Target tables found: {len(detection.target_tables)}")
    print(f"Upgrade targets found: {len(detection.targets)}")
    for index, target in enumerate(detection.targets, start=1):
        print(f"{index}. {target.table.title}: {target.name} at ({target.click_x}, {target.click_y}) with {target.action_pixels} action pixels")
    print(f"Overlay saved: {output}")


if __name__ == "__main__":
    main()
