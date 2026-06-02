from __future__ import annotations

import time
import re
import subprocess
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
import sys


sys.modules.setdefault("mouseinfo", SimpleNamespace(MouseInfoWindow=lambda: None))

import pyautogui
import pyscreeze
from PIL import Image
from mss import MSS

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
pyautogui.PAUSE = 0.05


def save_screenshot(image: Image.Image) -> None:
    SCREENS_DIR.mkdir(exist_ok=True)
    filename = f"screen_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}.png"
    image.save(SCREENS_DIR / filename)


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
    with MSS() as screen:
        if region is None:
            region = find_wurm_region()

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
        return Image.frombytes("RGB", shot.size, shot.rgb)


def screenshot(region=None):
    image = pipewire_screenshot()
    if image is None:
        image = mss_screenshot(region)
    else:
        image = crop_to_wurm(image)

    save_screenshot(image)
    return image


pyautogui.screenshot = screenshot
pyscreeze.screenshot = screenshot


def find_wurm_region() -> tuple[int, int, int, int]:
    result = subprocess.run(
        ["xwininfo", "-root", "-tree"],
        check=True,
        capture_output=True,
        text=True,
    )
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
