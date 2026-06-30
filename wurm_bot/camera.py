from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time

import sxtemp1

from .compass import CompassReading, read_compass
from .windows import click_wurm_local, find_wurm_click_region


CARDINAL_DEGREES = {
    "north": 0,
    "east": 90,
    "south": 180,
    "west": 270,
}


@dataclass(frozen=True)
class CameraTurnConfig:
    tolerance_degrees: float = 5.0
    max_steps: int = 12
    max_drag_pixels: int = 260
    min_drag_pixels: int = 8
    drag_duration: float = 0.16
    settle_seconds: float = 0.35
    read_timeout: float = 5.0
    read_interval: float = 0.15
    kp: float = 4.2
    ki: float = 0.0
    kd: float = 0.8
    drag_sign: int | None = None
    probe_pixels: int = 90
    focus_click: bool = False
    center_y_offset: int = 0

    @classmethod
    def from_env(cls) -> "CameraTurnConfig":
        return cls(
            tolerance_degrees=float(os.environ.get("WURM_CAMERA_TURN_TOLERANCE_DEGREES", "5")),
            max_steps=int(os.environ.get("WURM_CAMERA_TURN_MAX_STEPS", "12")),
            max_drag_pixels=int(os.environ.get("WURM_CAMERA_TURN_MAX_DRAG_PIXELS", "260")),
            min_drag_pixels=int(os.environ.get("WURM_CAMERA_TURN_MIN_DRAG_PIXELS", "8")),
            drag_duration=float(os.environ.get("WURM_CAMERA_TURN_DRAG_DURATION", "0.16")),
            settle_seconds=float(os.environ.get("WURM_CAMERA_TURN_SETTLE_SECONDS", "0.35")),
            read_timeout=float(os.environ.get("WURM_CAMERA_TURN_READ_TIMEOUT", "5.0")),
            read_interval=float(os.environ.get("WURM_CAMERA_TURN_READ_INTERVAL", "0.15")),
            kp=float(os.environ.get("WURM_CAMERA_TURN_KP", "4.2")),
            ki=float(os.environ.get("WURM_CAMERA_TURN_KI", "0.0")),
            kd=float(os.environ.get("WURM_CAMERA_TURN_KD", "0.8")),
            drag_sign=env_drag_sign(),
            probe_pixels=int(os.environ.get("WURM_CAMERA_TURN_PROBE_PIXELS", "90")),
            focus_click=os.environ.get("WURM_CAMERA_TURN_FOCUS_CLICK", "0") != "0",
            center_y_offset=int(os.environ.get("WURM_CAMERA_TURN_CENTER_Y_OFFSET", "0")),
        )


@dataclass(frozen=True)
class TurnStep:
    step: int
    before_degrees: int
    target_degrees: int
    error_degrees: float
    drag_pixels: int
    after_degrees: int | None


@dataclass(frozen=True)
class TurnResult:
    target_name: str
    target_degrees: int
    final_reading: CompassReading
    steps: list[TurnStep]


class PID:
    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.previous_error: float | None = None

    def update(self, error: float) -> float:
        self.integral += error
        derivative = 0.0 if self.previous_error is None else error - self.previous_error
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def env_drag_sign() -> int | None:
    raw_value = os.environ.get("WURM_CAMERA_TURN_DRAG_SIGN")
    if raw_value is None or raw_value == "":
        return None

    value = int(raw_value)
    if value not in (-1, 1):
        raise RuntimeError("WURM_CAMERA_TURN_DRAG_SIGN must be -1 or 1")
    return value


def parse_heading_target(value: str) -> tuple[str, int]:
    cleaned = value.strip().lower()
    if cleaned in CARDINAL_DEGREES:
        return cleaned, CARDINAL_DEGREES[cleaned]

    try:
        degrees = int(cleaned)
    except ValueError as error:
        raise RuntimeError(f"Unknown heading target: {value!r}") from error
    return f"{degrees % 360:03d}", degrees % 360


def shortest_turn_degrees(target_degrees: float, current_degrees: float) -> float:
    return (target_degrees - current_degrees + 540.0) % 360.0 - 180.0


def wait_compass_available(config: CameraTurnConfig) -> CompassReading:
    deadline = time.monotonic() + config.read_timeout
    last_reading: CompassReading | None = None
    while True:
        last_reading = read_compass(sxtemp1.screenshot())
        if last_reading.available:
            return last_reading

        if time.monotonic() >= deadline:
            status = last_reading.status if last_reading is not None else "none"
            raise RuntimeError(f"Compass heading is unavailable after {config.read_timeout:.1f}s; status={status}")

        time.sleep(config.read_interval)


def focus_wurm_center(config: CameraTurnConfig) -> None:
    if not config.focus_click:
        return

    _x, _y, width, height = find_wurm_click_region()
    click_wurm_local(width // 2, height // 2)
    time.sleep(config.settle_seconds)


def drag_camera_pixels(pixels: int, config: CameraTurnConfig) -> None:
    x, y, width, height = find_wurm_click_region()
    center_x = x + width // 2
    center_y = y + height // 2 + config.center_y_offset
    sxtemp1.pyautogui.moveTo(center_x, center_y, duration=0.04)
    sxtemp1.pyautogui.mouseDown(center_x, center_y, button="left")
    sxtemp1.pyautogui.moveRel(pixels, 0, duration=config.drag_duration)
    sxtemp1.pyautogui.mouseUp(button="left")
    time.sleep(config.settle_seconds)


def calibrate_drag_sign(config: CameraTurnConfig) -> int:
    if config.drag_sign is not None:
        return config.drag_sign

    before = wait_compass_available(config)
    assert before.direction_degrees is not None
    drag_camera_pixels(config.probe_pixels, config)
    after = wait_compass_available(config)
    assert after.direction_degrees is not None
    delta = shortest_turn_degrees(after.direction_degrees, before.direction_degrees)
    if abs(delta) < 2:
        raise RuntimeError(f"Camera drag probe moved the compass too little: delta={delta:.1f} deg")

    return 1 if delta > 0 else -1


def turn_to_heading(
    target_name: str,
    target_degrees: int,
    config: CameraTurnConfig,
    *,
    drag_sign: int,
) -> TurnResult:
    pid = PID(config.kp, config.ki, config.kd)
    steps: list[TurnStep] = []

    for step_index in range(1, config.max_steps + 1):
        reading = wait_compass_available(config)
        assert reading.direction_degrees is not None
        error = shortest_turn_degrees(target_degrees, reading.direction_degrees)
        if abs(error) <= config.tolerance_degrees:
            return TurnResult(target_name, target_degrees, reading, steps)

        raw_pixels = pid.update(error)
        drag_pixels = round(drag_sign * raw_pixels)
        drag_pixels = clamp_signed(drag_pixels, config.min_drag_pixels, config.max_drag_pixels)

        drag_camera_pixels(drag_pixels, config)
        after = wait_compass_available(config)
        steps.append(
            TurnStep(
                step=step_index,
                before_degrees=reading.direction_degrees,
                target_degrees=target_degrees,
                error_degrees=error,
                drag_pixels=drag_pixels,
                after_degrees=after.direction_degrees,
            )
        )

    final_reading = wait_compass_available(config)
    assert final_reading.direction_degrees is not None
    final_error = shortest_turn_degrees(target_degrees, final_reading.direction_degrees)
    if abs(final_error) <= config.tolerance_degrees:
        return TurnResult(target_name, target_degrees, final_reading, steps)

    raise RuntimeError(
        f"Could not turn to {target_name} ({target_degrees} deg): "
        f"final={final_reading.direction_degrees} deg, error={final_error:.1f} deg"
    )


def clamp_signed(value: int, min_abs: int, max_abs: int) -> int:
    if value == 0:
        return 0

    sign = 1 if value > 0 else -1
    magnitude = min(max_abs, max(min_abs, abs(value)))
    return sign * magnitude
