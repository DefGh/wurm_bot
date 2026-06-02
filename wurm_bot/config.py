from __future__ import annotations

from pathlib import Path


IMPROVE_KEY = "i"
REPAIR_KEY = "o"
LOGS_DIR = Path("/mnt/data/steam/steamapps/common/Wurm Online/gamedata/players/Defgh/logs")
ACTION_TIMEOUT = 45
DEBUG_SCREENSHOTS = True
USE_CLIENT_WINDOW_OFFSET = True

BASE_DIR = Path(__file__).resolve().parents[1]
SCREENS_DIR = BASE_DIR / "screens"
ROW_HEIGHT = 22
MIN_ACTION_PIXELS = 20
OCR_MIN_SCORE = 0.50
