from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    SCREENS_DIR,
    VITALS_COLOR_TOLERANCE,
    VITALS_FOOD_EMPTY_RGB,
    VITALS_FOOD_LINE,
    VITALS_FOOD_MIN_FILLED,
    VITALS_SAMPLE_COUNT,
    VITALS_STAMINA_LINE,
    VITALS_STAMINA_MIN_FILLED,
    VITALS_STAMINA_READY_RGB,
    VITALS_WATER_LINE,
    VITALS_WATER_MIN_FILLED,
    VITALS_WATER_MIN_RGB,
)


@dataclass(frozen=True)
class VitalSample:
    point: tuple[int, int]
    rgb: tuple[int, int, int]
    matched: bool


@dataclass(frozen=True)
class VitalCheck:
    name: str
    line: tuple[int, int, int]
    target_rgb: tuple[int, int, int]
    samples: list[VitalSample]
    filled_percent: int
    ok: bool
    ok_label: str
    low_label: str
    mode: str

    @property
    def status(self) -> str:
        return self.ok_label if self.ok else self.low_label

    @property
    def matched_count(self) -> int:
        return sum(1 for sample in self.samples if sample.matched)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def empty_percent(self) -> int:
        return 100 - self.filled_percent


@dataclass(frozen=True)
class Vitals:
    stamina: VitalCheck
    water: VitalCheck
    food: VitalCheck

    @property
    def blocking_checks(self) -> list[VitalCheck]:
        return [check for check in (self.water, self.food) if not check.ok]


def read_vitals(image: Image.Image) -> Vitals:
    image = image.convert("RGB")
    return Vitals(
        stamina=_check_line(
            image,
            "stamina",
            VITALS_STAMINA_LINE,
            VITALS_STAMINA_READY_RGB,
            VITALS_STAMINA_MIN_FILLED,
            mode="match",
            ok_label="ready",
            low_label="not ready",
        ),
        water=_check_line(
            image,
            "water",
            VITALS_WATER_LINE,
            VITALS_WATER_MIN_RGB,
            VITALS_WATER_MIN_FILLED,
            mode="match",
            ok_label="ok",
            low_label="below threshold",
        ),
        food=_check_line(
            image,
            "food",
            VITALS_FOOD_LINE,
            VITALS_FOOD_EMPTY_RGB,
            VITALS_FOOD_MIN_FILLED,
            mode="not-match",
            ok_label="ok",
            low_label="below threshold",
        ),
    )


def save_vitals_diagnostic(image: Image.Image, vitals: Vitals, output_path: Path | None = None) -> Path:
    SCREENS_DIR.mkdir(exist_ok=True)
    if output_path is None:
        output_path = SCREENS_DIR / "vitals_latest.png"

    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font()
    labels = []
    for check in (vitals.stamina, vitals.water, vitals.food):
        color = "lime" if check.ok else "red"
        x1, x2, y = check.line
        draw.line((x1, y, x2, y), fill=color, width=1)
        for sample in check.samples:
            x, sample_y = sample.point
            sample_color = "lime" if sample.matched else "red"
            draw.rectangle((x - 2, sample_y - 2, x + 2, sample_y + 2), outline=sample_color, width=1)
        labels.append(
            (
                f"{check.name}: approx={check.filled_percent}% {check.status} "
                f"{sample_summary(check)} target={check.target_rgb} mode={check.mode}",
                color,
            )
        )

    _draw_label_panel(draw, font, labels, out.size)

    out.save(output_path)
    return output_path


def format_vitals(vitals: Vitals) -> str:
    return "; ".join(
        f"{check.name}=~{check.filled_percent}% {check.status} "
        f"{sample_summary(check)} target={check.target_rgb} mode={check.mode}"
        for check in (vitals.stamina, vitals.water, vitals.food)
    )


def sample_summary(check: VitalCheck) -> str:
    label = "empty" if check.mode == "not-match" else "matches"
    return f"{label}={check.matched_count}/{check.sample_count}"


def _draw_label_panel(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    labels: list[tuple[str, str]],
    image_size: tuple[int, int],
) -> None:
    image_width, image_height = image_size
    line_boxes = [draw.textbbox((0, 0), label, font=font) for label, _color in labels]
    line_height = max((box[3] - box[1] for box in line_boxes), default=12)
    panel_width = max((box[2] - box[0] for box in line_boxes), default=0) + 14
    panel_height = line_height * len(labels) + 10
    panel_x = min(260, max(0, image_width - panel_width - 8))
    panel_y = min(58, max(0, image_height - panel_height - 8))

    draw.rectangle((panel_x, panel_y, panel_x + panel_width, panel_y + panel_height), fill=(0, 0, 0))
    for index, (label, color) in enumerate(labels):
        draw.text((panel_x + 7, panel_y + 5 + index * line_height), label, fill=color, font=font)


def median_rgb(image: Image.Image, point: tuple[int, int], radius: int = 1) -> tuple[int, int, int]:
    x, y = point
    if not (0 <= x < image.width and 0 <= y < image.height):
        raise RuntimeError(f"Vitals pixel {point} is outside screenshot size {image.size}")

    left = max(0, x - radius)
    top = max(0, y - radius)
    right = min(image.width, x + radius + 1)
    bottom = min(image.height, y + radius + 1)
    patch = np.array(image.crop((left, top, right, bottom)).convert("RGB"))
    median = np.median(patch.reshape(-1, 3), axis=0)
    return tuple(int(round(value)) for value in median)


def _check_line(
    image: Image.Image,
    name: str,
    line: tuple[int, int, int],
    target_rgb: tuple[int, int, int],
    min_filled_percent: float,
    mode: str,
    ok_label: str,
    low_label: str,
) -> VitalCheck:
    samples = line_samples(image, line, target_rgb)
    matched_count = sum(1 for sample in samples if sample.matched)
    if mode == "match":
        filled_count = matched_count
    elif mode == "not-match":
        filled_count = len(samples) - matched_count
    else:
        raise RuntimeError(f"Unknown vitals check mode: {mode}")
    filled_percent = round(filled_count * 100 / len(samples)) if samples else 0
    return VitalCheck(
        name=name,
        line=line,
        target_rgb=target_rgb,
        samples=samples,
        filled_percent=filled_percent,
        ok=filled_percent >= min_filled_percent,
        ok_label=ok_label,
        low_label=low_label,
        mode=mode,
    )


def line_samples(image: Image.Image, line: tuple[int, int, int], target_rgb: tuple[int, int, int]) -> list[VitalSample]:
    x1, x2, y = line
    if VITALS_SAMPLE_COUNT < 1:
        raise RuntimeError("WURM_VITALS_SAMPLE_COUNT must be at least 1")
    if x1 > x2:
        raise RuntimeError(f"Vitals line has invalid x range: {line}")
    if not (0 <= x1 < image.width and 0 <= x2 < image.width and 0 <= y < image.height):
        raise RuntimeError(f"Vitals line {line} is outside screenshot size {image.size}")

    if VITALS_SAMPLE_COUNT == 1:
        points = [((x1 + x2) // 2, y)]
    else:
        step = (x2 - x1) / (VITALS_SAMPLE_COUNT - 1)
        points = [(round(x1 + index * step), y) for index in range(VITALS_SAMPLE_COUNT)]

    samples = []
    for point in points:
        rgb = median_rgb(image, point)
        samples.append(VitalSample(point=point, rgb=rgb, matched=rgb_close(rgb, target_rgb)))
    return samples


def rgb_close(rgb: tuple[int, int, int], target_rgb: tuple[int, int, int]) -> bool:
    return max(abs(value - target) for value, target in zip(rgb, target_rgb, strict=True)) <= VITALS_COLOR_TOLERANCE


def _font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        return ImageFont.load_default()
