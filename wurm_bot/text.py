from __future__ import annotations

import re
from datetime import datetime


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
