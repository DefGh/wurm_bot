from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

import sxtemp1
from wurm_bot.compass import parse_box, read_compass, save_compass_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Wurm compass OCR and red-arrow heading.")
    parser.add_argument("image", nargs="?", type=Path, default=None, help="Optional screenshot to analyze instead of live Wurm.")
    parser.add_argument("--region", default=None, help="Optional compass region in local x1,y1,x2,y2 coordinates.")
    parser.add_argument("--output", type=Path, default=None, help="Output diagnostic image path.")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB") if args.image is not None else sxtemp1.screenshot()
    region = parse_box(args.region, "--region") if args.region else None
    reading = read_compass(image, region)
    path = save_compass_diagnostic(reading, args.output)

    if reading.available:
        print(f"Heading: {reading.heading}")
        print(f"Direction: {reading.direction_degrees:03d} deg")
        print(f"Arrow: {reading.arrow}")
    elif not reading.visible:
        print("Compass: hidden")
        print("Heading: unavailable")
        print("Direction: unavailable")
        print("Arrow: unavailable")
    else:
        print("Compass: visible")
        print("Heading: unavailable")
        print("Direction: unavailable")
        print("Arrow: unavailable")
    print(f"Center: {reading.center}")
    print(f"Radius: {reading.radius}")
    print(f"Region: {reading.region}")
    print(f"Labels: {', '.join(f'{name}=({text.cx},{text.cy}) {text.text!r}' for name, text in sorted(reading.labels.items()))}")
    print(f"Saved diagnostic: {path}")


if __name__ == "__main__":
    main()
