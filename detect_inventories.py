from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from wurm_bot.config import SCREENS_DIR
from wurm_bot.inventories import detect_inventories, save_inventory_overlay


def default_input_path() -> Path:
    preferred = SCREENS_DIR / "upgrade_carpenter_candidates_latest.png"
    if preferred.exists():
        return preferred

    screenshots = sorted(SCREENS_DIR.glob("*.png"), key=lambda path: path.stat().st_mtime)
    if not screenshots:
        raise RuntimeError(f"No screenshots found in {SCREENS_DIR}")
    return screenshots[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", type=Path, default=None, help="Screenshot file to analyze.")
    parser.add_argument("--output", type=Path, default=None, help="Output overlay image path.")
    args = parser.parse_args()

    input_path = args.image or default_input_path()
    image = Image.open(input_path)
    detection = detect_inventories(image)
    output_path = save_inventory_overlay(detection, args.output)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Inventories: {len(detection.inventories)}")
    for index, inventory in enumerate(detection.inventories, start=1):
        print(
            f"{index}. {inventory.title!r} region=({inventory.x1}, {inventory.y1}, {inventory.x2}, {inventory.y2}) "
            f"table=({inventory.table.x1}, {inventory.table.y1}, {inventory.table.x2}, {inventory.table.y2})"
        )


if __name__ == "__main__":
    main()
