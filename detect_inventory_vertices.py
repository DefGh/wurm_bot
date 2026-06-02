from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from wurm_bot.config import SCREENS_DIR
from wurm_bot.inventory_vertices import (
    assemble_window_rectangles,
    detect_inventory_vertices,
    group_window_rectangles,
    recognize_window_titles,
    save_vertex_overlay,
)


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
    matches = detect_inventory_vertices(image)
    rectangles = assemble_window_rectangles(matches)
    rectangles = recognize_window_titles(image, rectangles)
    grouped = group_window_rectangles(rectangles)
    output_path = save_vertex_overlay(image, matches, args.output, rectangles)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Vertices: {len(matches)}")
    for index, match in enumerate(matches, start=1):
        print(
            f"{index}. {match.kind} region=({match.x1}, {match.y1}, {match.x2}, {match.y2}) "
            f"score={match.score:.3f}"
        )
    print(f"Rectangles: {len(rectangles)}")
    for index, rectangle in enumerate(rectangles, start=1):
        title = rectangle.title.strip() or "<unknown>"
        print(
            f"{index}. group={rectangle.group} title={title!r} "
            f"region=({rectangle.x1}, {rectangle.y1}, {rectangle.x2}, {rectangle.y2}) "
            f"score={rectangle.score:.3f}"
        )
    print("Groups:")
    for group, group_rectangles in grouped.items():
        titles = [rectangle.title.strip() or "<unknown>" for rectangle in group_rectangles]
        print(f"{group}: {len(group_rectangles)} {titles}")


if __name__ == "__main__":
    main()
