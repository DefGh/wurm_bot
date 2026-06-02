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

IMPROVE_KEY = "i"
REPAIR_KEY = "o"
ACTION_TIMEOUT = 45
DEBUG_SCREENSHOTS = True
USE_CLIENT_WINDOW_OFFSET = True

ROW_HEIGHT = 22
MIN_ACTION_PIXELS = 20
OCR_MIN_SCORE = 0.50


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
