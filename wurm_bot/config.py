from __future__ import annotations

import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"
SCREENS_DIR = BASE_DIR / "screens"


def _load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()

        os.environ.setdefault(key, value)


_load_env_file()


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number, got {raw_value!r}") from error


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {raw_value!r}") from error


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {raw_value!r}")


def _env_line(name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"{name} must use x1,x2,y format, got {raw_value!r}")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as error:
        raise RuntimeError(f"{name} must use integer x1,x2,y format, got {raw_value!r}") from error


def _env_rgb(name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"{name} must use r,g,b format, got {raw_value!r}")
    try:
        rgb = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as error:
        raise RuntimeError(f"{name} must use integer r,g,b format, got {raw_value!r}") from error
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise RuntimeError(f"{name} must use RGB values in 0..255, got {raw_value!r}")
    return rgb


IMPROVE_KEY = "i"
REPAIR_KEY = "o"
ACTION_TIMEOUT = 45
DEBUG_SCREENSHOTS = True
USE_CLIENT_WINDOW_OFFSET = True

ROW_HEIGHT = 22
MIN_ACTION_PIXELS = 5
OCR_MIN_SCORE = 0.50

VITALS_STAMINA_LINE = _env_line("WURM_STAMINA_LINE", (32, 229, 82))
VITALS_WATER_LINE = _env_line("WURM_WATER_LINE", (32, 106, 92))
VITALS_FOOD_LINE = _env_line("WURM_FOOD_LINE", (138, 239, 92))
VITALS_STAMINA_READY_RGB = _env_rgb("WURM_STAMINA_READY_RGB", (95, 136, 34))
VITALS_WATER_MIN_RGB = _env_rgb("WURM_WATER_MIN_RGB", (70, 114, 165))
VITALS_FOOD_EMPTY_RGB = _env_rgb("WURM_FOOD_EMPTY_RGB", (53, 46, 37))
VITALS_COLOR_TOLERANCE = _env_float("WURM_VITALS_COLOR_TOLERANCE", 35.0)
VITALS_SAMPLE_COUNT = _env_int("WURM_VITALS_SAMPLE_COUNT", 20)
VITALS_STAMINA_MIN_FILLED = _env_float("WURM_STAMINA_MIN_FILLED", 100.0)
VITALS_WATER_MIN_FILLED = _env_float("WURM_WATER_MIN_FILLED", 70.0)
VITALS_FOOD_MIN_FILLED = _env_float("WURM_FOOD_MIN_FILLED", 70.0)
VITALS_POLL_SECONDS = _env_float("WURM_VITALS_POLL_SECONDS", 1.0)
VITALS_OVERLAY_WIDTH = _env_int("WURM_VITALS_OVERLAY_WIDTH", 320)
VITALS_OVERLAY_HEIGHT = _env_int("WURM_VITALS_OVERLAY_HEIGHT", 118)
VITALS_OVERLAY_OFFSET_X = _env_int("WURM_VITALS_OVERLAY_OFFSET_X", 16)
VITALS_OVERLAY_OFFSET_Y = _env_int("WURM_VITALS_OVERLAY_OFFSET_Y", 16)
VITALS_OVERLAY_TOPMOST = _env_bool("WURM_VITALS_OVERLAY_TOPMOST", True)


def _default_logs_dir() -> Path:
    env_path = os.environ.get("WURM_LOGS_DIR")
    if env_path:
        return Path(env_path).expanduser()

    player = os.environ.get("WURM_PLAYER", "Defgh")
    if sys.platform == "darwin":
        candidates = [
            Path.home()
            / "Library/Application Support/Steam/steamapps/common/Wurm Online/gamedata/players"
            / player
            / "logs",
            Path.home() / "Library/Application Support/Wurm Online/gamedata/players" / player / "logs",
            Path.home() / "wurm/players" / player / "logs",
        ]
    else:
        candidates = [
            Path("/mnt/data/steam/steamapps/common/Wurm Online/gamedata/players") / player / "logs",
            Path.home() / ".steam/steam/steamapps/common/Wurm Online/gamedata/players" / player / "logs",
            Path.home() / ".local/share/Steam/steamapps/common/Wurm Online/gamedata/players" / player / "logs",
            Path.home() / "wurm/players" / player / "logs",
        ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


LOGS_DIR = _default_logs_dir()
