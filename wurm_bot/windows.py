from __future__ import annotations

import re
import subprocess
import sys

import sxtemp1

from .config import USE_CLIENT_WINDOW_OFFSET


def find_wurm_click_region() -> tuple[int, int, int, int]:
    if sys.platform == "darwin" or not USE_CLIENT_WINDOW_OFFSET:
        return sxtemp1.find_wurm_region()

    try:
        result = subprocess.run(
            ["xwininfo", "-root", "-tree"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return sxtemp1.find_wurm_region()

    candidates = []
    pattern = re.compile(
        r"^\s+(0x[0-9a-f]+).*?(\d+)x(\d+)\+(-?\d+)\+(-?\d+)\s+\+(-?\d+)\+(-?\d+)",
        re.IGNORECASE,
    )

    for line in result.stdout.splitlines():
        if "wurm online" not in line.lower():
            continue
        if "mutter-x11-frames" in line.lower():
            continue

        match = pattern.search(line)
        if not match:
            continue

        _window_id, width, height, _x, _y, abs_x, abs_y = match.groups()
        width = int(width)
        height = int(height)
        abs_x = int(abs_x)
        abs_y = int(abs_y)
        if width < 300 or height < 300:
            continue

        candidates.append((width * height, abs_x, abs_y, width, height))

    if candidates:
        _area, x, y, width, height = max(candidates)
        return x, y, width, height

    return sxtemp1.find_wurm_region()


def click_wurm_local(x: int, y: int, clicks: int = 1, button: str = "left") -> None:
    offset_x, offset_y, _width, _height = find_wurm_click_region()
    sxtemp1.pyautogui.click(offset_x + x, offset_y + y, clicks=clicks, interval=0.08, button=button)


def left_click_wurm_local(x: int, y: int, hold: float = 0.06) -> None:
    offset_x, offset_y, _width, _height = find_wurm_click_region()
    screen_x = offset_x + x
    screen_y = offset_y + y
    sxtemp1.pyautogui.moveTo(screen_x, screen_y, duration=0.05)
    sxtemp1.pyautogui.sleep(0.05)
    sxtemp1.pyautogui.mouseDown(screen_x, screen_y, button="left")
    sxtemp1.pyautogui.sleep(hold)
    sxtemp1.pyautogui.mouseUp(screen_x, screen_y, button="left")


def double_click_wurm_local(x: int, y: int, interval: float = 0.10) -> None:
    offset_x, offset_y, _width, _height = find_wurm_click_region()
    screen_x = offset_x + x
    screen_y = offset_y + y
    sxtemp1.pyautogui.click(screen_x, screen_y)
    sxtemp1.pyautogui.sleep(interval)
    sxtemp1.pyautogui.click(screen_x, screen_y)


def move_wurm_local(x: int, y: int) -> None:
    offset_x, offset_y, _width, _height = find_wurm_click_region()
    sxtemp1.pyautogui.moveTo(offset_x + x, offset_y + y)


def screen_to_wurm_local(x: int, y: int) -> tuple[int, int]:
    offset_x, offset_y, _width, _height = find_wurm_click_region()
    return x - offset_x, y - offset_y


def press(key: str) -> None:
    sxtemp1.pyautogui.press(key)
