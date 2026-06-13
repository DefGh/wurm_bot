from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
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
    VITALS_OVERLAY_MARKER_SIZE,
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
class VitalOverlaySample:
    point: tuple[int, int]
    ok: bool


@dataclass(frozen=True)
class VitalOverlayFrame:
    line: tuple[int, int, int]
    samples: list[VitalOverlaySample]


@dataclass(frozen=True)
class VitalOverlaySegment:
    x1: int
    y1: int
    x2: int
    y2: int
    ok: bool


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


class VitalsPixelOverlay:
    def __init__(self, title: str = "Wurm vitals pixels", marker_size: int | None = None):
        try:
            self._backend = _MacOSVitalsPixelOverlay(title, marker_size) if sys.platform == "darwin" else None
        except RuntimeError:
            self._backend = None

        if self._backend is None:
            self._backend = _TkVitalsPixelOverlay(title, marker_size)

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    @property
    def closed(self) -> bool:
        return self._backend.closed

    def update_for_wurm(self, vitals: Vitals | None, message: str | None = None) -> None:
        self._backend.update_for_wurm(vitals, message)

    def update(
        self,
        vitals: Vitals | None,
        region: tuple[int, int, int, int] | None,
        message: str | None = None,
    ) -> None:
        self._backend.update(vitals, region, message)

    def hide(self) -> None:
        self._backend.hide()

    def close(self) -> None:
        self._backend.close()


class _MacOSVitalsPixelOverlay:
    backend_name = "macos-native"

    def __init__(self, title: str = "Wurm vitals pixels", marker_size: int | None = None):
        try:
            import AppKit
            from Foundation import NSDate, NSRunLoop, NSString
            import objc
        except Exception as error:
            raise RuntimeError("macOS native vitals overlay requires PyObjC AppKit.") from error

        self.AppKit = AppKit
        self.NSDate = NSDate
        self.NSRunLoop = NSRunLoop
        self.NSString = NSString
        self.objc = objc
        self.title = title
        self.marker_size = marker_size if marker_size is not None else VITALS_OVERLAY_MARKER_SIZE
        if self.marker_size < 1:
            raise RuntimeError("WURM_VITALS_OVERLAY_MARKER_SIZE must be at least 1")

        self.closed = False
        self.window = None
        self.view = None
        self._view_class = self._make_view_class()

        self.app = AppKit.NSApplication.sharedApplication()
        activation_policy = getattr(AppKit, "NSApplicationActivationPolicyAccessory", None)
        if activation_policy is not None:
            self.app.setActivationPolicy_(self._constant(activation_policy))

    def update_for_wurm(self, vitals: Vitals | None, message: str | None = None) -> None:
        try:
            import sxtemp1

            region = sxtemp1.find_wurm_region()
        except Exception as error:
            self.update(None, None, message or str(error))
            return

        self.update(vitals, region, message)

    def update(
        self,
        vitals: Vitals | None,
        region: tuple[int, int, int, int] | None,
        message: str | None = None,
    ) -> None:
        if self.closed:
            return
        if region is None:
            self.hide()
            self._pump()
            return

        if message or vitals is None:
            frames: list[VitalOverlayFrame] = []
            status = message or "waiting for vitals..."
        else:
            frames = overlay_frames(vitals)
            status = None

        x, y, width, height = region
        self._ensure_window(width, height)
        self._move_window(x, y, width, height)
        self.view.setOverlayState_markerSize_message_(frames, self.marker_size, status)
        self.view.setNeedsDisplay_(True)
        self.view.displayIfNeeded()
        self.window.orderFrontRegardless()
        self._pump()

    def hide(self) -> None:
        if self.window is not None:
            self.window.orderOut_(None)
        self._pump()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.window is not None:
            self.window.orderOut_(None)
            self.window.close()
            self.window = None
            self.view = None
        self._pump()

    def _ensure_window(self, width: int, height: int) -> None:
        AppKit = self.AppKit
        if self.window is None:
            frame = AppKit.NSMakeRect(0, 0, width, height)
            style = getattr(AppKit, "NSWindowStyleMaskBorderless", None)
            if style is None:
                style = getattr(AppKit, "NSBorderlessWindowMask")
            backing = getattr(AppKit, "NSBackingStoreBuffered")
            self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                frame,
                self._constant(style),
                self._constant(backing),
                False,
            )
            self.window.setTitle_(self.title)
            self.window.setOpaque_(False)
            self.window.setBackgroundColor_(AppKit.NSColor.clearColor())
            self.window.setIgnoresMouseEvents_(True)
            self.window.setHasShadow_(False)
            self._configure_window_level()
            self.view = self._view_class.alloc().initWithFrame_(frame)
            self.window.setContentView_(self.view)
            return

        content_size = self.window.contentView().frame().size
        if int(content_size.width) == width and int(content_size.height) == height:
            return

        frame = AppKit.NSMakeRect(0, 0, width, height)
        self.view.setFrame_(frame)

    def _move_window(self, x: int, y: int, width: int, height: int) -> None:
        AppKit = self.AppKit
        appkit_y = self._top_left_to_appkit_y(y, height)
        self.window.setFrame_display_(AppKit.NSMakeRect(x, appkit_y, width, height), True)

    def _top_left_to_appkit_y(self, y: int, height: int) -> int:
        screens = self.AppKit.NSScreen.screens()
        screen_top = max(screen.frame().origin.y + screen.frame().size.height for screen in screens)
        return round(screen_top - y - height)

    def _configure_window_level(self) -> None:
        AppKit = self.AppKit
        level = getattr(AppKit, "NSScreenSaverWindowLevel", None)
        if level is None:
            level = getattr(AppKit, "NSStatusWindowLevel", 25)
        if VITALS_OVERLAY_TOPMOST:
            self.window.setLevel_(self._constant(level))

        behavior = 0
        for name in (
            "NSWindowCollectionBehaviorCanJoinAllSpaces",
            "NSWindowCollectionBehaviorFullScreenAuxiliary",
            "NSWindowCollectionBehaviorStationary",
            "NSWindowCollectionBehaviorIgnoresCycle",
        ):
            behavior |= self._constant(getattr(AppKit, name, 0))
        if behavior:
            self.window.setCollectionBehavior_(behavior)

    def _make_view_class(self):
        AppKit = self.AppKit
        NSString = self.NSString
        objc = self.objc

        class VitalsOverlayView(AppKit.NSView):
            def initWithFrame_(self, frame):
                self = objc.super(VitalsOverlayView, self).initWithFrame_(frame)
                if self is None:
                    return None
                self.frames = []
                self.marker_size = 7
                self.message = None
                return self

            def isOpaque(self):
                return False

            def setOverlayState_markerSize_message_(self, frames, marker_size, message):
                self.frames = list(frames)
                self.marker_size = marker_size
                self.message = message

            def drawRect_(self, rect):
                bounds = self.bounds()
                height = bounds.size.height
                if self.message:
                    AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.06, 0.07, 0.08, 0.82).set()
                    AppKit.NSBezierPath.fillRect_(AppKit.NSMakeRect(12, height - 38, min(720, bounds.size.width - 24), 26))
                    attributes = {
                        AppKit.NSForegroundColorAttributeName: AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                            1.0, 0.75, 0.35, 1.0
                        ),
                        AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_(13),
                    }
                    NSString.stringWithString_(str(self.message[:120])).drawAtPoint_withAttributes_(
                        AppKit.NSMakePoint(20, height - 31),
                        attributes,
                    )
                    return

                for frame in self.frames:
                    for segment in overlay_frame_segments(frame, self.marker_size):
                        if segment.ok:
                            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 1.0, 0.0, 0.95).set()
                        else:
                            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.0, 0.0, 0.95).set()
                        AppKit.NSBezierPath.fillRect_(
                            AppKit.NSMakeRect(
                                segment.x1,
                                height - segment.y2,
                                segment.x2 - segment.x1,
                                segment.y2 - segment.y1,
                            )
                        )

        return VitalsOverlayView

    def _constant(self, value):
        return value() if callable(value) else value

    def _pump(self) -> None:
        self.NSRunLoop.currentRunLoop().runUntilDate_(self.NSDate.dateWithTimeIntervalSinceNow_(0.001))


class _TkVitalsPixelOverlay:
    backend_name = "tk"

    def __init__(self, title: str = "Wurm vitals pixels", marker_size: int | None = None):
        try:
            import tkinter as tk
        except ImportError as error:
            raise RuntimeError("Vitals pixel overlay requires tkinter.") from error

        self.tk = tk
        self.title = title
        self.marker_size = marker_size if marker_size is not None else VITALS_OVERLAY_MARKER_SIZE
        if self.marker_size < 1:
            raise RuntimeError("WURM_VITALS_OVERLAY_MARKER_SIZE must be at least 1")

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("q", lambda _event: self.close())

        self.closed = False
        self._segments: list[object] = []
        self._message_window = None
        self._message_label = None

    def update_for_wurm(self, vitals: Vitals | None, message: str | None = None) -> None:
        try:
            import sxtemp1

            region = sxtemp1.find_wurm_region()
        except Exception as error:
            self.update(None, None, message or str(error))
            return

        self.update(vitals, region, message)

    def update(
        self,
        vitals: Vitals | None,
        region: tuple[int, int, int, int] | None,
        message: str | None = None,
    ) -> None:
        if self.closed:
            return

        if message or vitals is None or region is None:
            self.hide_segments()
            self._show_message(message or "waiting for vitals...", region)
            self._pump()
            return

        self._hide_message()
        segments = [
            segment
            for frame in overlay_frames(vitals)
            for segment in overlay_frame_segments(frame, self.marker_size)
        ]
        self._ensure_segment_count(len(segments))

        x0, y0, _width, _height = region
        for window, segment in zip(self._segments[: len(segments)], segments, strict=True):
            color = "#00ff00" if segment.ok else "#ff3030"
            window.configure(bg=color)
            width = max(1, segment.x2 - segment.x1)
            height = max(1, segment.y2 - segment.y1)
            window.geometry(f"{width}x{height}+{x0 + segment.x1}+{y0 + segment.y1}")
            window.deiconify()
            self._raise_window(window)
            self._configure_native_overlay_window(window)

        self._pump()

    def hide_segments(self) -> None:
        for segment in self._segments:
            try:
                segment.withdraw()
            except self.tk.TclError:
                pass

    def hide(self) -> None:
        if self.closed:
            return
        self.hide_segments()
        self._hide_message()
        self._pump()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    def _ensure_segment_count(self, count: int) -> None:
        while len(self._segments) < count:
            segment = self.tk.Toplevel(self.root)
            segment.withdraw()
            segment.title(f"{self.title} frame {len(self._segments) + 1}")
            segment.overrideredirect(True)
            segment.resizable(False, False)
            segment.configure(bg="#ff3030")
            self._raise_window(segment)
            self._configure_native_overlay_window(segment)
            self._segments.append(segment)

        for segment in self._segments[count:]:
            try:
                segment.withdraw()
            except self.tk.TclError:
                pass

    def _show_message(self, message: str, region: tuple[int, int, int, int] | None) -> None:
        if self._message_window is None:
            window = self.tk.Toplevel(self.root)
            window.withdraw()
            window.title(f"{self.title} status")
            window.overrideredirect(True)
            window.resizable(False, False)
            label = self.tk.Label(
                window,
                bg="#101214",
                fg="#ffbd5f",
                padx=8,
                pady=5,
                text="",
            )
            label.pack()
            self._message_window = window
            self._message_label = label
            self._configure_native_overlay_window(window)

        self._message_label.configure(text=message[:120])
        self._message_window.update_idletasks()
        if region is None:
            x, y = 40, 40
        else:
            x, y, _width, _height = region
            x += VITALS_OVERLAY_OFFSET_X
            y += VITALS_OVERLAY_OFFSET_Y
        self._message_window.geometry(f"+{x}+{y}")
        self._message_window.deiconify()
        self._raise_window(self._message_window)
        self._configure_native_overlay_window(self._message_window)

    def _hide_message(self) -> None:
        if self._message_window is None:
            return
        try:
            self._message_window.withdraw()
        except self.tk.TclError:
            pass

    def _raise_window(self, window) -> None:
        try:
            if VITALS_OVERLAY_TOPMOST:
                window.attributes("-topmost", True)
            window.lift()
        except self.tk.TclError:
            pass

    def _pump(self) -> None:
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.closed = True

    def _configure_native_overlay_window(self, window) -> None:
        if sys.platform != "darwin":
            return

        try:
            from AppKit import NSApplication
            import AppKit
        except Exception:
            return

        try:
            self.root.update_idletasks()
            app = NSApplication.sharedApplication()
            ns_window = next(
                (candidate for candidate in app.windows() if str(candidate.title()) == window.title()),
                None,
            )
            if ns_window is None:
                return

            level = getattr(AppKit, "NSScreenSaverWindowLevel", None)
            if level is None:
                level = getattr(AppKit, "NSStatusWindowLevel", 25)
            ns_window.setLevel_(level() if callable(level) else level)
            ns_window.setIgnoresMouseEvents_(True)
            ns_window.setHasShadow_(False)

            behavior = 0
            for name in (
                "NSWindowCollectionBehaviorCanJoinAllSpaces",
                "NSWindowCollectionBehaviorFullScreenAuxiliary",
                "NSWindowCollectionBehaviorStationary",
            ):
                value = getattr(AppKit, name, 0)
                behavior |= value() if callable(value) else value
            if behavior:
                ns_window.setCollectionBehavior_(ns_window.collectionBehavior() | behavior)
        except Exception:
            return


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


def overlay_samples(vitals: Vitals) -> list[VitalOverlaySample]:
    samples = []
    for check in (vitals.stamina, vitals.water, vitals.food):
        invert = check.mode == "not-match"
        for sample in check.samples:
            samples.append(VitalOverlaySample(point=sample.point, ok=not sample.matched if invert else sample.matched))
    return samples


def overlay_frames(vitals: Vitals) -> list[VitalOverlayFrame]:
    frames = []
    for check in (vitals.stamina, vitals.water, vitals.food):
        invert = check.mode == "not-match"
        samples = [
            VitalOverlaySample(point=sample.point, ok=not sample.matched if invert else sample.matched)
            for sample in check.samples
        ]
        frames.append(VitalOverlayFrame(line=check.line, samples=samples))
    return frames


def overlay_frame_segments(frame: VitalOverlayFrame, marker_size: int) -> list[VitalOverlaySegment]:
    if not frame.samples:
        return []

    line_x1, line_x2, line_y = frame.line
    border = max(2, marker_size // 3)
    gap = max(2, marker_size // 2)
    top_y1 = line_y - gap - border
    top_y2 = line_y - gap
    bottom_y1 = line_y + gap + 1
    bottom_y2 = bottom_y1 + border
    side_y1 = top_y1
    side_y2 = bottom_y2

    samples = sorted(frame.samples, key=lambda sample: sample.point[0])
    segments: list[VitalOverlaySegment] = []
    for index, sample in enumerate(samples):
        sample_x, _sample_y = sample.point
        if index == 0:
            x1 = line_x1
        else:
            previous_x = samples[index - 1].point[0]
            x1 = round((previous_x + sample_x) / 2)

        if index == len(samples) - 1:
            x2 = line_x2 + 1
        else:
            next_x = samples[index + 1].point[0]
            x2 = round((sample_x + next_x) / 2)

        if x2 <= x1:
            x2 = x1 + 1

        segments.append(VitalOverlaySegment(x1=x1, y1=top_y1, x2=x2, y2=top_y2, ok=sample.ok))
        segments.append(VitalOverlaySegment(x1=x1, y1=bottom_y1, x2=x2, y2=bottom_y2, ok=sample.ok))

    segments.append(
        VitalOverlaySegment(
            x1=line_x1 - border - 1,
            y1=side_y1,
            x2=line_x1 - 1,
            y2=side_y2,
            ok=samples[0].ok,
        )
    )
    segments.append(
        VitalOverlaySegment(
            x1=line_x2 + 2,
            y1=side_y1,
            x2=line_x2 + border + 2,
            y2=side_y2,
            ok=samples[-1].ok,
        )
    )
    return segments


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
    import sxtemp1

    overlay = VitalsPixelOverlay()
    print(f"Vitals frame overlay is running with {overlay.backend_name} backend. Press Ctrl+C in the terminal to close.")
    try:
        while not overlay.closed:
            try:
                image = sxtemp1.screenshot()
                overlay.update_for_wurm(read_vitals(image))
            except Exception as error:
                overlay.update_for_wurm(None, str(error))

            time.sleep(max(0.05, VITALS_POLL_SECONDS))
    except KeyboardInterrupt:
        pass
    finally:
        overlay.close()


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
