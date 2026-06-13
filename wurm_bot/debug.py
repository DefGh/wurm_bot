from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import SCREENS_DIR
from .models import Candidate, OcrText, candidate_action_point
from .text import timestamp
from .windows import screen_to_wurm_local


def clean_screenshot_history() -> None:
    if not SCREENS_DIR.exists():
        return

    for path in SCREENS_DIR.glob("*.png"):
        if path.stem.endswith("_latest"):
            continue
        path.unlink()


def _font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def save_debug_image(
    image: Image.Image,
    candidates: list[Candidate],
    prefix: str,
    log_rows: list[OcrText] | None = None,
    row_label: str = "input",
) -> Path:
    SCREENS_DIR.mkdir(exist_ok=True)
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = _font()

    for index, candidate in enumerate(candidates, start=1):
        draw.rectangle((candidate.x1, candidate.y1, candidate.x2, candidate.y2), outline="lime", width=3)
        label = f"{index}: {candidate.name[:28]}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        label_box = (
            candidate.x1 + 4,
            candidate.y1 + 2,
            candidate.x1 + 8 + text_bbox[2] - text_bbox[0],
            candidate.y1 + 6 + text_bbox[3] - text_bbox[1],
        )
        draw.rectangle(label_box, fill=(0, 0, 0))
        draw.text((candidate.x1 + 6, candidate.y1 + 3), label, fill="lime", font=font)

    for index, row in enumerate(log_rows or [], start=1):
        x1 = max(0, row.x1 - 8)
        y1 = max(0, row.y1 - 4)
        x2 = row.x2 + 8
        y2 = row.y2 + 4
        draw.rectangle((x1, y1, x2, y2), outline="cyan", width=3)
        label = f"{row_label} {index}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        draw.rectangle(
            (x1 + 3, y1 + 2, x1 + 9 + text_bbox[2] - text_bbox[0], y1 + 8 + text_bbox[3] - text_bbox[1]),
            fill=(0, 0, 0),
        )
        draw.text((x1 + 5, y1 + 3), label, fill="cyan", font=font)

    path = SCREENS_DIR / f"{prefix}_{timestamp()}.png"
    out.save(path)
    if prefix == "upgrade_carpenter_candidates":
        out.save(SCREENS_DIR / "upgrade_carpenter_candidates_latest.png")
    return path


def save_mouse_diagnostic(image: Image.Image, candidate: Candidate, mouse_x: int, mouse_y: int) -> Path:
    SCREENS_DIR.mkdir(exist_ok=True)
    out = image.copy()
    draw = ImageDraw.Draw(out)
    center_x, center_y = candidate_action_point(candidate)
    draw.rectangle((candidate.x1, candidate.y1, candidate.x2, candidate.y2), outline="lime", width=3)
    draw.line((center_x - 18, center_y, center_x + 18, center_y), fill="red", width=3)
    draw.line((center_x, center_y - 18, center_x, center_y + 18), fill="red", width=3)
    local_mouse_x, local_mouse_y = screen_to_wurm_local(mouse_x, mouse_y)
    draw.line((local_mouse_x - 18, local_mouse_y, local_mouse_x + 18, local_mouse_y), fill="cyan", width=3)
    draw.line((local_mouse_x, local_mouse_y - 18, local_mouse_x, local_mouse_y + 18), fill="cyan", width=3)
    draw.text((candidate.x1 + 6, candidate.y1 + 3), "red=target cyan=mouse", fill="white")
    path = SCREENS_DIR / "upgrade_carpenter_mouse_diagnostic_latest.png"
    out.save(path)
    return path


def save_log_select_diagnostic(image: Image.Image, row: OcrText, mouse_x: int, mouse_y: int) -> Path:
    SCREENS_DIR.mkdir(exist_ok=True)
    out = image.copy()
    draw = ImageDraw.Draw(out)
    local_mouse_x, local_mouse_y = screen_to_wurm_local(mouse_x, mouse_y)
    draw.rectangle((row.x1 - 8, row.y1 - 4, row.x2 + 8, row.y2 + 4), outline="cyan", width=3)
    draw.line((local_mouse_x - 18, local_mouse_y, local_mouse_x + 18, local_mouse_y), fill="red", width=3)
    draw.line((local_mouse_x, local_mouse_y - 18, local_mouse_x, local_mouse_y + 18), fill="red", width=3)
    path = SCREENS_DIR / "upgrade_carpenter_log_select_latest.png"
    out.save(path)
    return path
