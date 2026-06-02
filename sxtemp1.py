from __future__ import annotations

import os
import sys
import time
import re
import subprocess
import tempfile
import ctypes
from ctypes import byref, c_bool, c_char_p, c_double, c_long, c_longlong, c_uint32, c_void_p, create_string_buffer
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace


sys.modules.setdefault("mouseinfo", SimpleNamespace(MouseInfoWindow=lambda: None))

import pyautogui
import pyscreeze
from PIL import Image

try:
    from mss import MSS
except ImportError:
    MSS = None

try:
    from pipewire_capture import CaptureStream, PortalCapture, is_available as pipewire_is_available
except ImportError:
    CaptureStream = None
    PortalCapture = None
    pipewire_is_available = lambda: False


BASE_DIR = Path(__file__).resolve().parent
SCREENS_DIR = BASE_DIR / "screens"
HAS_OPENCV = find_spec("cv2") is not None
PIPEWIRE_CAPTURE_INTERVAL = 0.1
_pipewire_session = None
_pipewire_stream = None
_macos_window_cache = None
pyautogui.PAUSE = 0.05
_pyautogui_screenshot = pyautogui.screenshot


@dataclass(frozen=True)
class MacWindow:
    window_id: int
    x: int
    y: int
    width: int
    height: int

    @property
    def region(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True)
class MacWindowInfo:
    owner: str
    title: str
    layer: int
    window: MacWindow


def save_screenshot(image: Image.Image) -> None:
    if os.environ.get("WURM_SAVE_RAW_SCREENSHOTS") != "1":
        return

    SCREENS_DIR.mkdir(exist_ok=True)
    filename = f"screen_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}.png"
    image.save(SCREENS_DIR / filename)


def configured_wurm_region() -> tuple[int, int, int, int] | None:
    raw_region = os.environ.get("WURM_WINDOW_REGION")
    if not raw_region:
        return None

    parts = [part for part in re.split(r"[,\sx]+", raw_region.strip()) if part]
    if len(parts) != 4:
        raise RuntimeError("WURM_WINDOW_REGION must be four numbers: x,y,width,height")

    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise RuntimeError("WURM_WINDOW_REGION must contain only integers") from exc

    if width <= 0 or height <= 0:
        raise RuntimeError("WURM_WINDOW_REGION width and height must be positive")

    return x, y, width, height


def resize_to_region_size(image: Image.Image, region: tuple[int, int, int, int] | None) -> Image.Image:
    if region is None:
        return image.convert("RGB")

    _x, _y, width, height = region
    if image.size == (width, height):
        return image.convert("RGB")

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((width, height), resampling).convert("RGB")


def pyautogui_screenshot(region=None):
    image = _pyautogui_screenshot(region=region) if region else _pyautogui_screenshot()
    return resize_to_region_size(image, region)


def macos_window_screenshot(window: MacWindow) -> Image.Image:
    fd, path = tempfile.mkstemp(".png")
    os.close(fd)
    try:
        result = subprocess.run(
            ["screencapture", "-l", str(window.window_id), "-o", "-x", path],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(message or f"screencapture failed with code {result.returncode}")

        image = Image.open(path)
        image.load()
        return resize_to_region_size(image, window.region)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def pipewire_screenshot():
    global _pipewire_session, _pipewire_stream

    if not pipewire_is_available():
        return None

    if _pipewire_session is None or _pipewire_stream is None:
        print("Select the Wurm window in the screen sharing dialog.")
        portal = PortalCapture()
        session = portal.select_window()
        if session is None:
            return None

        stream = CaptureStream(
            session.fd,
            session.node_id,
            session.width,
            session.height,
            capture_interval=PIPEWIRE_CAPTURE_INTERVAL,
        )
        stream.start()
        _pipewire_session = session
        _pipewire_stream = stream

    for _ in range(30):
        frame = _pipewire_stream.get_frame()
        if frame is not None:
            if getattr(_pipewire_stream, "window_invalid", False):
                _pipewire_stream.stop()
                _pipewire_session.close()
                _pipewire_session = None
                _pipewire_stream = None
                return None

            rgb_frame = frame[:, :, [2, 1, 0]]
            return Image.fromarray(rgb_frame, "RGB")

        time.sleep(0.1)

    return None


def crop_to_wurm(image: Image.Image) -> Image.Image:
    x, y, width, height = find_wurm_region()
    if image.size == (width, height):
        return image

    if image.width >= x + width and image.height >= y + height:
        return image.crop((x, y, x + width, y + height))

    return image


def mss_screenshot(region=None):
    if region is None:
        region = find_wurm_region()

    if MSS is None:
        return pyautogui_screenshot(region)

    with MSS() as screen:
        if region:
            left, top, width, height = region
            monitor = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
        else:
            monitor = screen.monitors[0]

        shot = screen.grab(monitor)
        return resize_to_region_size(Image.frombytes("RGB", shot.size, shot.rgb), region)


def screenshot(region=None):
    if sys.platform == "darwin":
        image = macos_screenshot(region)
    else:
        image = pipewire_screenshot()
        if image is None:
            image = mss_screenshot(region)
        else:
            image = crop_to_wurm(image)

    save_screenshot(image)
    return image


def macos_screenshot(region=None):
    backend = os.environ.get("WURM_MAC_SCREENSHOT_BACKEND", "window").lower()
    configured_region = configured_wurm_region()

    if backend == "window" and region is None and configured_region is None:
        window = find_wurm_window_macos()
        if window is not None:
            return macos_window_screenshot(window)

    region = region or configured_region or find_wurm_region()

    if backend == "window":
        backend = "mss"

    if backend == "pyautogui":
        return pyautogui_screenshot(region)

    if backend != "mss":
        raise RuntimeError("WURM_MAC_SCREENSHOT_BACKEND must be 'window', 'mss', or 'pyautogui'")

    try:
        return mss_screenshot(region)
    except Exception as exc:
        if os.environ.get("WURM_MAC_SCREENSHOT_FALLBACK") == "pyautogui":
            return pyautogui_screenshot(region)
        raise RuntimeError(
            "Could not capture the Wurm window with mss. Set WURM_MAC_SCREENSHOT_BACKEND=pyautogui "
            "to try the pyautogui backend, or set WURM_WINDOW_REGION=x,y,width,height in .env."
        ) from exc


pyautogui.screenshot = screenshot
pyscreeze.screenshot = screenshot


def find_wurm_region() -> tuple[int, int, int, int]:
    configured_region = configured_wurm_region()
    if configured_region is not None:
        return configured_region

    if sys.platform == "darwin":
        return find_wurm_region_macos()

    return find_wurm_region_x11()


def full_screen_region() -> tuple[int, int, int, int]:
    width, height = pyautogui.size()
    if width <= 0 or height <= 0:
        raise RuntimeError(
            "pyautogui could not read the screen size. On macOS, grant your Terminal "
            "Accessibility and Screen Recording permissions, or set WURM_WINDOW_REGION=x,y,width,height in .env."
        )

    return 0, 0, int(width), int(height)


def find_wurm_region_pyautogui() -> tuple[int, int, int, int] | None:
    match_text = window_match_text()
    try:
        windows = pyautogui.getWindowsWithTitle(match_text)
    except (AttributeError, NotImplementedError):
        return None

    candidates = []
    for window in windows:
        width = int(getattr(window, "width", 0) or 0)
        height = int(getattr(window, "height", 0) or 0)
        if width < 300 or height < 300:
            continue

        x = int(getattr(window, "left", 0) or 0)
        y = int(getattr(window, "top", 0) or 0)
        candidates.append((width * height, x, y, width, height))

    if not candidates:
        return None

    _area, x, y, width, height = max(candidates)
    return x, y, width, height


def find_wurm_window_macos() -> MacWindow | None:
    global _macos_window_cache

    now = time.monotonic()
    cache_ttl = float(os.environ.get("WURM_WINDOW_CACHE_TTL", "1.0"))
    if _macos_window_cache is not None:
        cached_at, cached_window = _macos_window_cache
        if now - cached_at <= cache_ttl:
            return cached_window

    window = find_wurm_window_macos_coregraphics()
    _macos_window_cache = (now, window)
    return window


def find_wurm_window_macos_coregraphics() -> MacWindow | None:
    match_text = window_match_text().lower()
    exclude_terms = window_exclude_terms()

    candidates: list[tuple[int, MacWindow]] = []
    for info in list_macos_windows_coregraphics():
        haystack = f"{info.owner} {info.title}".lower()
        if match_text not in haystack:
            continue
        if any(term in haystack for term in exclude_terms):
            continue
        if info.layer != 0 or info.window.width < 300 or info.window.height < 300:
            continue

        candidates.append((info.window.width * info.window.height, info.window))

    if not candidates:
        return None

    _area, window = max(candidates, key=lambda item: item[0])
    return window


def list_macos_windows_coregraphics() -> list[MacWindowInfo]:
    try:
        cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    except OSError:
        return []

    configure_corefoundation_functions(cf)
    cg.CGWindowListCopyWindowInfo.argtypes = [c_uint32, c_uint32]
    cg.CGWindowListCopyWindowInfo.restype = c_void_p

    window_list_option_on_screen_only = 1
    window_list_exclude_desktop_elements = 16
    options = window_list_option_on_screen_only | window_list_exclude_desktop_elements
    window_infos = cg.CGWindowListCopyWindowInfo(options, 0)
    if not window_infos:
        return []

    keys = {
        "number": cf_string(cf, "kCGWindowNumber"),
        "owner": cf_string(cf, "kCGWindowOwnerName"),
        "name": cf_string(cf, "kCGWindowName"),
        "bounds": cf_string(cf, "kCGWindowBounds"),
        "layer": cf_string(cf, "kCGWindowLayer"),
    }

    windows: list[MacWindowInfo] = []
    try:
        count = cf.CFArrayGetCount(window_infos)
        for index in range(count):
            info = cf.CFArrayGetValueAtIndex(window_infos, index)
            if not info:
                continue

            owner = cf_string_value(cf, cf_dictionary_value(cf, info, keys["owner"]))
            title = cf_string_value(cf, cf_dictionary_value(cf, info, keys["name"]))
            layer = cf_number_int(cf, cf_dictionary_value(cf, info, keys["layer"]))
            window_id = cf_number_int(cf, cf_dictionary_value(cf, info, keys["number"]))
            bounds = cf_dictionary_value(cf, info, keys["bounds"])
            x = cf_dictionary_number_int(cf, bounds, "X")
            y = cf_dictionary_number_int(cf, bounds, "Y")
            width = cf_dictionary_number_int(cf, bounds, "Width")
            height = cf_dictionary_number_int(cf, bounds, "Height")
            if window_id <= 0 or width <= 0 or height <= 0:
                continue

            windows.append(
                MacWindowInfo(
                    owner=owner,
                    title=title,
                    layer=layer,
                    window=MacWindow(window_id, x, y, width, height),
                )
            )
    finally:
        for key in keys.values():
            if key:
                cf.CFRelease(key)
        cf.CFRelease(window_infos)

    return windows


def window_match_text() -> str:
    return os.environ.get("WURM_WINDOW_MATCH", "Wurm Online")


def window_exclude_terms() -> tuple[str, ...]:
    raw_terms = os.environ.get("WURM_WINDOW_EXCLUDE", "wurm_bot,visual studio code,code,codex")
    return tuple(term.strip().lower() for term in raw_terms.split(",") if term.strip())


def configure_corefoundation_functions(cf) -> None:
    cf.CFArrayGetCount.argtypes = [c_void_p]
    cf.CFArrayGetCount.restype = c_long
    cf.CFArrayGetValueAtIndex.argtypes = [c_void_p, c_long]
    cf.CFArrayGetValueAtIndex.restype = c_void_p
    cf.CFDictionaryGetValue.argtypes = [c_void_p, c_void_p]
    cf.CFDictionaryGetValue.restype = c_void_p
    cf.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
    cf.CFStringCreateWithCString.restype = c_void_p
    cf.CFStringGetCString.argtypes = [c_void_p, c_char_p, c_long, c_uint32]
    cf.CFStringGetCString.restype = c_bool
    cf.CFNumberGetValue.argtypes = [c_void_p, c_long, c_void_p]
    cf.CFNumberGetValue.restype = c_bool
    cf.CFRelease.argtypes = [c_void_p]
    cf.CFRelease.restype = None


def cf_string(cf, text: str):
    return cf.CFStringCreateWithCString(None, text.encode("utf-8"), 0x08000100)


def cf_dictionary_value(cf, dictionary, key):
    if not dictionary or not key:
        return None
    return cf.CFDictionaryGetValue(dictionary, key)


def cf_string_value(cf, value) -> str:
    if not value:
        return ""

    buffer = create_string_buffer(4096)
    if cf.CFStringGetCString(value, buffer, len(buffer), 0x08000100):
        return buffer.value.decode("utf-8", errors="replace")
    return ""


def cf_number_int(cf, value) -> int:
    if not value:
        return 0

    number = c_longlong()
    if cf.CFNumberGetValue(value, 4, byref(number)):
        return int(number.value)

    double_number = c_double()
    if cf.CFNumberGetValue(value, 13, byref(double_number)):
        return int(double_number.value)

    return 0


def cf_dictionary_number_int(cf, dictionary, key_text: str) -> int:
    key = cf_string(cf, key_text)
    try:
        return cf_number_int(cf, cf_dictionary_value(cf, dictionary, key))
    finally:
        if key:
            cf.CFRelease(key)


def find_wurm_region_macos() -> tuple[int, int, int, int]:
    window = find_wurm_window_macos()
    if window is not None:
        return window.region

    pyautogui_region = find_wurm_region_pyautogui()
    if pyautogui_region is not None:
        return pyautogui_region

    osascript_region = find_wurm_region_macos_osascript()
    if osascript_region is not None:
        return osascript_region

    if os.environ.get("WURM_ALLOW_FULL_SCREEN_FALLBACK") == "1":
        return full_screen_region()

    raise RuntimeError(
        "Wurm window was not found. Make sure Wurm Online is visible, grant Terminal "
        "Accessibility permission, or set WURM_WINDOW_REGION=x,y,width,height in .env."
    )


def find_wurm_region_macos_osascript() -> tuple[int, int, int, int] | None:
    match_text = window_match_text()
    script = macos_window_bounds_script(match_text)
    timeout = float(os.environ.get("WURM_WINDOW_DETECT_TIMEOUT", "3"))

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "Could not read Wurm window bounds with osascript. Grant Terminal Accessibility "
            f"permission or set WURM_WINDOW_REGION=x,y,width,height in .env. {message}"
        )

    output = result.stdout.strip()
    if not output:
        return None

    parts = [part.strip() for part in output.split(",")]
    if len(parts) != 4:
        return None

    try:
        x, y, width, height = (int(float(part)) for part in parts)
    except ValueError:
        return None

    if width < 300 or height < 300:
        return None

    return x, y, width, height


def macos_window_bounds_script(match_text: str) -> str:
    escaped_match_text = match_text.replace("\\", "\\\\").replace('"', '\\"')
    return f'''
on boundsText(theWindow)
    tell application "System Events"
        set winPosition to position of theWindow
        set winSize to size of theWindow
    end tell
    return (item 1 of winPosition as integer as text) & "," & (item 2 of winPosition as integer as text) & "," & (item 1 of winSize as integer as text) & "," & (item 2 of winSize as integer as text)
end boundsText

set matchText to "{escaped_match_text}"
tell application "System Events"
    ignoring case
        repeat with proc in application processes
            try
                set procName to name of proc as text
                set procWindows to windows of proc
                if procName contains matchText then
                    repeat with win in procWindows
                        return my boundsText(win)
                    end repeat
                end if
                repeat with win in procWindows
                    try
                        set winName to name of win as text
                        if winName contains matchText then
                            return my boundsText(win)
                        end if
                    end try
                end repeat
            end try
        end repeat
    end ignoring
end tell
return ""
'''


def find_wurm_region_x11() -> tuple[int, int, int, int]:
    try:
        result = subprocess.run(
            ["xwininfo", "-root", "-tree"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Wurm window was not found with xwininfo. Set WURM_WINDOW_REGION=x,y,width,height."
        ) from exc

    candidates = []
    pattern = re.compile(
        r"^\s+(0x[0-9a-f]+).*?(\d+)x(\d+)\+(-?\d+)\+(-?\d+)\s+\+(-?\d+)\+(-?\d+)",
        re.IGNORECASE,
    )

    for line in result.stdout.splitlines():
        if "wurm" not in line.lower():
            continue

        match = pattern.search(line)
        if not match:
            continue

        window_id, width, height, _x, _y, abs_x, abs_y = match.groups()
        width = int(width)
        height = int(height)
        abs_x = int(abs_x)
        abs_y = int(abs_y)

        if width < 300 or height < 300:
            continue

        candidates.append((width * height, window_id, abs_x, abs_y, width, height))

    if not candidates:
        raise RuntimeError("Wurm window was not found")

    _area, _window_id, x, y, width, height = max(candidates)
    return x, y, width, height


def locate_on_wurm(pic: str | Pattern):
    x, y, width, height = find_wurm_region()
    haystack = screenshot((x, y, width, height))

    locate_kwargs = {}
    if HAS_OPENCV:
        locate_kwargs["confidence"] = image_confidence(pic)

    try:
        location = pyscreeze.locate(image_path(pic), haystack, **locate_kwargs)
    except pyscreeze.ImageNotFoundException:
        return None

    if not location:
        return None

    return pyscreeze.Box(location.left + x, location.top + y, location.width, location.height)


@dataclass(frozen=True)
class Pattern:
    filename: str
    confidence: float = 0.8

    def similar(self, confidence: float) -> "Pattern":
        return Pattern(self.filename, confidence)

    @property
    def path(self) -> str:
        return str(BASE_DIR / self.filename)


def image_path(pic: str | Pattern) -> str:
    if isinstance(pic, Pattern):
        return pic.path
    return str(BASE_DIR / pic)


def image_confidence(pic: str | Pattern) -> float:
    if isinstance(pic, Pattern):
        return pic.confidence
    return 0.8


def wait(pic_or_seconds: str | Pattern | int | float, timeout: int | float | None = None):
    if isinstance(pic_or_seconds, (int, float)) and timeout is None:
        time.sleep(pic_or_seconds)
        return None

    if timeout is None:
        timeout = 30

    pic = pic_or_seconds
    deadline = time.monotonic() + timeout
    while True:
        location = locate_on_wurm(pic)

        if location:
            return location

        if time.monotonic() >= deadline:
            raise TimeoutError(f"Image was not found on screen: {pic}")

        time.sleep(0.2)


def click(pic: str | Pattern) -> None:
    location = wait(pic)
    pyautogui.click(pyautogui.center(location))


def hover(pic: str | Pattern) -> None:
    location = wait(pic)
    pyautogui.moveTo(pyautogui.center(location))


def press(key: str) -> None:
    pyautogui.press(key)


def ended() -> bool:
    try:
        wait("a.png", 1)
        return True
    except TimeoutError:
        return False


def highligh_wait(pic: str | Pattern, dur: int | float):
    wait(pic, dur)


def finish_action():
    highligh_wait(Pattern("1779123792217.png").similar(0.90), 30)
    highligh_wait(Pattern("1779968820468.png").similar(0.97), 30)


def upgrade(count: int):
    for _ in range(1, count):
        wait(1)
        finish_action()
        press("i")
        wait(1)
        finish_action()
        press("o")


def spam(key: str, count: int):
    for _ in range(1, count):
        wait(3)
        finish_action()
        press(key)


def cont():
    while not ended():
        wait(1)
        finish_action()
        click("1779123892085.png")
        hover("1779128182230.png")


def cont_c(count: int):
    for _ in range(count):
        wait(1)
        finish_action()
        click("1779123892085.png")
        hover("1779128182230.png")


if __name__ == "__main__":
    # cont_c(50)
    # cont()
    upgrade(130)
    # spam("2", 150)
