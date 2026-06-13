from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    SCREENS_DIR,
    VITALS_COLOR_TOLERANCE,
    VITALS_FOOD_EMPTY_RGB,
    VITALS_FOOD_LINE,
    VITALS_FOOD_MIN_FILLED,
    VITALS_OVERLAY_HEIGHT,
    VITALS_OVERLAY_OFFSET_X,
    VITALS_OVERLAY_OFFSET_Y,
    VITALS_OVERLAY_TOPMOST,
    VITALS_OVERLAY_WIDTH,
    VITALS_POLL_SECONDS,
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


def render_vitals_overlay(vitals: Vitals | None, message: str | None = None) -> Image.Image:
    width = max(240, VITALS_OVERLAY_WIDTH)
    height = max(90, VITALS_OVERLAY_HEIGHT)
    out = Image.new("RGB", (width, height), (16, 18, 20))
    draw = ImageDraw.Draw(out)
    font = _font()
    title_font = _font(14)

    draw.rectangle((0, 0, width - 1, height - 1), outline=(62, 68, 75), width=1)
    draw.text((12, 8), "Wurm vitals", fill=(235, 238, 242), font=title_font)

    if message:
        draw.text((12, 38), message[:60], fill=(255, 190, 90), font=font)
        return out

    if vitals is None:
        draw.text((12, 38), "waiting for screenshot...", fill=(180, 186, 194), font=font)
        return out

    bar_x = 86
    bar_w = width - bar_x - 16
    y = 34
    for check in (vitals.stamina, vitals.water, vitals.food):
        _draw_overlay_bar(draw, font, check, bar_x, y, bar_w)
        y += 26

    return out


def run_vitals_overlay() -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Vitals overlay requires opencv-python.") from error

    import sxtemp1

    window_name = "Wurm vitals"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, VITALS_OVERLAY_WIDTH, VITALS_OVERLAY_HEIGHT)
    _place_overlay_window(cv2, window_name, sxtemp1)
    if VITALS_OVERLAY_TOPMOST and hasattr(cv2, "WND_PROP_TOPMOST"):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print("Vitals overlay is running. Press q or Esc in the overlay window to close.")
    last_place = 0.0
    while True:
        now = time.monotonic()
        if now - last_place >= 5.0:
            _place_overlay_window(cv2, window_name, sxtemp1)
            last_place = now

        try:
            image = sxtemp1.screenshot()
            frame = render_vitals_overlay(read_vitals(image))
        except Exception as error:
            frame = render_vitals_overlay(None, str(error))

        cv2.imshow(window_name, _pil_to_bgr(frame))
        key = cv2.waitKey(max(50, int(VITALS_POLL_SECONDS * 1000))) & 0xFF
        if key in {27, ord("q")}:
            break

    cv2.destroyWindow(window_name)


def _draw_overlay_bar(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    check: VitalCheck,
    x: int,
    y: int,
    width: int,
) -> None:
    label = check.name.upper()
    percent = max(0, min(100, check.filled_percent))
    fill_width = round(width * percent / 100)
    color = (92, 205, 120) if check.ok else (235, 82, 82)
    label_color = (215, 220, 226)
    text = f"{percent}% {check.status}"

    draw.text((12, y + 2), label, fill=label_color, font=font)
    draw.rectangle((x, y, x + width, y + 15), fill=(39, 43, 48), outline=(82, 88, 95), width=1)
    if fill_width > 0:
        draw.rectangle((x + 1, y + 1, x + fill_width - 1, y + 14), fill=color)
    draw.text((x + 6, y + 1), text, fill=(245, 247, 250), font=font)


def _place_overlay_window(cv2, window_name: str, sxtemp1_module) -> None:
    try:
        x, y, _width, _height = sxtemp1_module.find_wurm_region()
    except Exception:
        return
    cv2.moveWindow(window_name, x + VITALS_OVERLAY_OFFSET_X, y + VITALS_OVERLAY_OFFSET_Y)


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))[:, :, ::-1]


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


def _font(size: int = 13):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()
