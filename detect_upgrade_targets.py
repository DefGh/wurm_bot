from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from wurm_bot.config import SCREENS_DIR
from wurm_bot.upgrade_targets import detect_upgrade_targets, save_upgrade_target_overlay


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
    detection = detect_upgrade_targets(image)
    output_path = save_upgrade_target_overlay(detection, args.output)

    other_windows = [window for window in detection.windows if window.group == "other"]
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Other containers: {len(other_windows)} {[window.title for window in other_windows]}")
    print(f"Target tables: {len(detection.target_tables)} {[table.title for table in detection.target_tables]}")
    print(f"Upgrade targets: {len(detection.targets)}")
    for index, target in enumerate(detection.targets, start=1):
        print(
            f"{index}. table={target.table.title!r} name={target.name!r} "
            f"row=({target.x1}, {target.y1}, {target.x2}, {target.y2}) "
            f"click=({target.click_x}, {target.click_y}) action_pixels={target.action_pixels}"
        )


if __name__ == "__main__":
    main()
