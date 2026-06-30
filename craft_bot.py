from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import time

from PIL import ImageDraw
import sxtemp1

from wurm_bot.config import ACTION_TIMEOUT, LOGS_DIR, SCREENS_DIR, VITALS_POLL_SECONDS
from wurm_bot.events import EventLogTail, SkillGain, SkillLogTail
from wurm_bot.models import OcrText
from wurm_bot.text import normalize, timestamp
from wurm_bot.vitals import Vitals, VitalsPixelOverlay, format_vitals, read_vitals, sample_summary, save_vitals_diagnostic
from wurm_bot.vision import ocr_image
from wurm_bot.windows import left_click_wurm_local, press


CRAFT_START_MARKERS = (
    "you start to work",
    "you start creating",
    "you start to create",
    "you start continuing",
    "you start to continue",
)

CRAFT_QUEUED_MARKERS = (
    "after you finish creating you will start creating again",
    "after you finish continuing you will start continuing again",
)

CRAFT_DONE_MARKERS = (
    "you create ",
    "you attach ",
    "you continue ",
    "you finish ",
    "you complete ",
)

CRAFT_FAIL_MARKERS = (
    "could very well work next time",
    "you almost made it",
    "problems solved in the wrong way",
    "you fail miserably",
)

CRAFT_BUSY_MARKERS = (
    "you're too busy",
    "you are too busy",
)

CRAFT_TERMINAL_ERROR_MARKERS = (
    "isn't enough weight",
    "is not enough weight",
    "not enough",
    "please select a larger part",
    "doesn't fit",
    "does not fit",
    "don't fit",
    "do not fit",
    "inventory contains too many items",
    "would be too large to handle",
    "too far away",
    "cannot",
    "can't",
    "you need",
    "you must",
    "missing",
    "lacks",
    "no space",
    "no room",
)

CREATE_BUTTON_TEXTS = ("create", "continue")


@dataclass(frozen=True)
class CraftResult:
    status: str
    lines: list[str]

    @property
    def ok_to_continue(self) -> bool:
        return self.status in {"success", "failed_attempt"}

    @property
    def last_line(self) -> str:
        return self.lines[-1] if self.lines else ""


class VitalsOverlay:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.overlay: VitalsPixelOverlay | None = None
        if not enabled:
            return

        try:
            self.overlay = VitalsPixelOverlay("Wurm craft vitals frames")
        except RuntimeError as error:
            print(f"Vitals overlay disabled: {error}")
            self.enabled = False
            return

        print(f"Vitals overlay enabled: {self.overlay.backend_name}")
        self.update(None, "waiting for vitals...")

    def before_screenshot(self) -> None:
        # The frame overlay avoids the sampled pixels, so it can stay visible
        # during screenshots without polluting vitals reads.
        return

    def update(self, vitals: Vitals | None, message: str | None = None) -> None:
        if not self.enabled or self.overlay is None:
            return

        self.overlay.update_for_wurm(vitals, message)

    def close(self) -> None:
        if self.enabled and self.overlay is not None:
            self.overlay.close()


def click_or_press_create(
    key: str | None,
    click_local: tuple[int, int] | None,
    auto_button: bool,
    cached_button_local: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    if click_local is not None:
        left_click_wurm_local(*click_local)
        return cached_button_local

    if cached_button_local is not None:
        left_click_wurm_local(*cached_button_local)
        return cached_button_local

    if auto_button:
        button = find_create_button()
        if button is not None:
            cached_button_local = (button.cx, button.cy)
            print(f"Found {button.text!r} button at local {cached_button_local}; reusing this coordinate.")
            left_click_wurm_local(*cached_button_local)
            return cached_button_local

        if key is None:
            raise RuntimeError("Create/Continue button was not found. Use --click-local x,y or --key KEY.")

        print(f"Create/Continue button was not found; falling back to key {key!r}.")

    if key is not None:
        press(key)
        return cached_button_local

    raise RuntimeError("Either auto button detection, --key, or --click-local must be available")


def find_create_button() -> OcrText | None:
    image = sxtemp1.screenshot()
    texts = ocr_image(image)
    return find_create_button_from_texts(texts, image.size)


def diagnose_create_button() -> None:
    image = sxtemp1.screenshot()
    texts = ocr_image(image)
    button = find_create_button_from_texts(texts, image.size)
    SCREENS_DIR.mkdir(exist_ok=True)
    out = image.copy()
    draw = ImageDraw.Draw(out)

    for text in texts:
        cleaned = normalize(text.text).replace(" ", "")
        if any(label in cleaned for label in CREATE_BUTTON_TEXTS):
            color = "lime" if button is not None and text == button else "yellow"
            draw.rectangle((text.x1 - 4, text.y1 - 4, text.x2 + 4, text.y2 + 4), outline=color, width=3)
            draw.text((text.x1, max(0, text.y1 - 18)), f"{text.text} {text.score:.2f}", fill=color)

    path = SCREENS_DIR / f"craft_create_button_{timestamp()}.png"
    out.save(path)
    if button is None:
        print(f"Create/Continue button was not found. Saved diagnostic: {path}")
        return

    print(f"Found button {button.text!r} at local ({button.cx}, {button.cy}), score={button.score:.2f}")
    print(f"Saved diagnostic: {path}")


def diagnose_click_create_button() -> None:
    image = sxtemp1.screenshot()
    texts = ocr_image(image)
    button = find_create_button_from_texts(texts, image.size)
    if button is None:
        diagnose_create_button()
        return

    print(f"Left-clicking button {button.text!r} at local ({button.cx}, {button.cy}), score={button.score:.2f}")
    diagnose_create_button()
    left_click_wurm_local(button.cx, button.cy)


def find_create_button_from_texts(texts: list[OcrText], image_size: tuple[int, int]) -> OcrText | None:
    _width, height = image_size
    candidates = []
    for text in texts:
        cleaned = normalize(text.text)
        compact = cleaned.replace(" ", "")
        if not button_text_like(compact):
            continue
        if text.y1 < height * 0.15:
            continue
        candidates.append(text)

    if not candidates:
        return None

    return max(candidates, key=lambda item: (item.score, item.y1))


def button_text_like(compact_text: str) -> bool:
    if "[" in compact_text or "]" in compact_text:
        return False
    if len(compact_text) > 16:
        return False
    return bool(re.fullmatch(r"(create|continue)[0-9.,:]*", compact_text))


def wait_for_craft_result(log_tail: EventLogTail, timeout: int = ACTION_TIMEOUT) -> CraftResult:
    deadline = time.monotonic() + timeout
    collected: list[str] = []
    saw_start = False
    saw_busy = False

    while time.monotonic() < deadline:
        new_lines = log_tail.read_new()
        if new_lines:
            collected.extend(new_lines)
            for line in new_lines:
                text = normalize(line)
                if _has_marker(text, CRAFT_TERMINAL_ERROR_MARKERS):
                    return CraftResult("terminal_error", collected)
                if _has_marker(text, CRAFT_DONE_MARKERS):
                    return CraftResult("success", collected)
                if _has_marker(text, CRAFT_FAIL_MARKERS):
                    return CraftResult("failed_attempt", collected)
                if _has_marker(text, CRAFT_START_MARKERS) or _has_marker(text, CRAFT_QUEUED_MARKERS):
                    saw_start = True
                if _has_marker(text, CRAFT_BUSY_MARKERS):
                    saw_busy = True

        time.sleep(0.20)

    if saw_busy and not saw_start:
        return CraftResult("busy", collected)
    if saw_start:
        return CraftResult("timeout_after_start", collected)
    return CraftResult("timeout", collected)


def ensure_vitals_ready(overlay: VitalsOverlay) -> None:
    last_wait_print = 0.0
    while True:
        overlay.before_screenshot()
        image = sxtemp1.screenshot()
        vitals = read_vitals(image)
        overlay.update(vitals)
        blocked_checks = vitals.blocking_checks
        if blocked_checks:
            path = save_vitals_diagnostic(image, vitals)
            problem = ", ".join(
                f"{check.name} ~{check.filled_percent}% {check.status} "
                f"{sample_summary(check)} line={check.line} target={check.target_rgb}"
                for check in blocked_checks
            )
            raise RuntimeError(f"Vitals guard stopped: {problem}. Saved diagnostic: {path}")

        if vitals.stamina.ok:
            return

        now = time.monotonic()
        if now - last_wait_print >= 5.0:
            print(f"Waiting for stamina: {format_vitals(vitals)}")
            last_wait_print = now
        time.sleep(max(0.1, VITALS_POLL_SECONDS))


def run(
    count: int | None,
    count_success: int | None,
    key: str | None,
    click_local: tuple[int, int] | None,
    auto_button: bool,
    timeout: int,
    check_vitals: bool,
    overlay: VitalsOverlay,
    stop_on_failed_attempt: bool,
) -> None:
    event_tail = EventLogTail(LOGS_DIR)
    skill_tail = SkillLogTail(LOGS_DIR)
    completed = 0
    failed_attempts = 0
    iteration = 0
    cached_button_local: tuple[int, int] | None = None

    try:
        while (count is None or iteration < count) and (count_success is None or completed < count_success):
            iteration += 1
            if check_vitals:
                ensure_vitals_ready(overlay)

            event_tail.mark()
            cached_button_local = click_or_press_create(key, click_local, auto_button, cached_button_local)
            result = wait_for_craft_result(event_tail, timeout)

            for line in result.lines:
                print(line)

            if result.status == "success":
                completed += 1
                print(f"Craft #{iteration}: success ({completed} completed)")
            elif result.status == "failed_attempt":
                failed_attempts += 1
                print(f"Craft #{iteration}: failed attempt ({failed_attempts} failed attempts)")
                if stop_on_failed_attempt:
                    break
            elif result.status == "busy":
                print("Craft action was not accepted: too busy. Retrying after vitals wait.")
                iteration -= 1
                time.sleep(0.8)
                continue
            else:
                reason = result.last_line or result.status
                print(f"Stopping craft loop: {result.status}: {reason}")
                break

            if not result.ok_to_continue:
                break

    finally:
        print(f"Craft loop summary: iterations={iteration}, completed={completed}, failed_attempts={failed_attempts}")
        print_skill_gains(skill_tail.read_gains())
        overlay.close()


def print_skill_gains(gains: list[SkillGain]) -> None:
    if not gains:
        print("Skill gains: none")
        return

    print("Skill gains:")
    for gain in gains:
        count = f", ticks={gain.count}" if gain.count > 1 else ""
        print(f"- {gain.name}: +{gain.amount:.6f} -> {gain.value:.6f}{count}")


def parse_click_local(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None

    parts = [part.strip() for part in value.split(",", 1)]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--click-local must use x,y format")

    try:
        x, y = int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("--click-local must contain integer x,y values") from error

    return x, y


def _has_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat Wurm Create/Continue actions using event-log completion.")
    parser.add_argument("--count", type=int, default=None, help="Number of craft attempts. Omit to run until terminal error.")
    parser.add_argument(
        "--count-success",
        "--count-sucsess",
        dest="count_success",
        type=int,
        default=None,
        help="Number of successful crafts to complete. Failed attempts do not count toward this limit.",
    )
    parser.add_argument("--key", default=None, help="Fallback key that activates Create/Continue if OCR button detection fails.")
    parser.add_argument("--click-local", type=parse_click_local, default=None, help="Click Wurm-local Create/Continue button coordinate: x,y.")
    parser.add_argument("--timeout", type=int, default=ACTION_TIMEOUT, help="Seconds to wait for each craft result.")
    parser.add_argument("--no-vitals", action="store_true", help="Do not wait for stamina/water/food between attempts.")
    parser.add_argument("--no-overlay", action="store_true", help="Do not show the live vitals overlay window.")
    parser.add_argument("--no-auto-button", action="store_true", help="Do not OCR-detect Create/Continue; use --key or --click-local.")
    parser.add_argument("--diagnose-create-button", action="store_true", help="Find Create/Continue by OCR and save a diagnostic screenshot.")
    parser.add_argument("--diagnose-click-create", action="store_true", help="Find Create/Continue by OCR, save diagnostic, and perform one left click.")
    parser.add_argument("--stop-on-failed-attempt", action="store_true", help="Stop on normal Wurm RNG craft failure.")
    args = parser.parse_args()

    if args.diagnose_click_create:
        diagnose_click_create_button()
        return

    if args.diagnose_create_button:
        diagnose_create_button()
        return

    if args.count is not None and args.count < 1:
        raise RuntimeError("--count must be at least 1")
    if args.count_success is not None and args.count_success < 1:
        raise RuntimeError("--count-success must be at least 1")

    auto_button = args.click_local is None and not args.no_auto_button
    key = None if args.click_local is not None else args.key
    overlay = VitalsOverlay(enabled=not args.no_overlay and not args.no_vitals)
    run(
        count=args.count,
        count_success=args.count_success,
        key=key,
        click_local=args.click_local,
        auto_button=auto_button,
        timeout=args.timeout,
        check_vitals=not args.no_vitals,
        overlay=overlay,
        stop_on_failed_attempt=args.stop_on_failed_attempt,
    )


if __name__ == "__main__":
    main()
