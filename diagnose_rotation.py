from __future__ import annotations

import argparse
from dataclasses import replace
import time

from wurm_bot.camera import (
    CameraTurnConfig,
    calibrate_drag_sign,
    focus_wurm_center,
    parse_heading_target,
    turn_to_heading,
    wait_compass_available,
)
from wurm_bot.compass import save_compass_diagnostic
from wurm_bot.config import SCREENS_DIR
from wurm_bot.text import timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn Wurm camera to requested compass headings.")
    parser.add_argument(
        "--sequence",
        default="north,west,south,east",
        help="Comma-separated headings: north,west,south,east or numeric degrees.",
    )
    parser.add_argument("--tolerance", type=float, default=None, help="Acceptable final heading error in degrees.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum PID drag iterations per target.")
    parser.add_argument("--drag-sign", type=int, choices=(-1, 1), default=None, help="Skip calibration and use this drag sign.")
    parser.add_argument("--kp", type=float, default=None, help="PID proportional gain, pixels per degree.")
    parser.add_argument("--ki", type=float, default=None, help="PID integral gain.")
    parser.add_argument("--kd", type=float, default=None, help="PID derivative gain.")
    parser.add_argument("--focus-click", action="store_true", help="Click Wurm center once before rotation.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait before starting.")
    args = parser.parse_args()

    config = CameraTurnConfig.from_env()
    if args.tolerance is not None:
        config = replace(config, tolerance_degrees=args.tolerance)
    if args.max_steps is not None:
        config = replace(config, max_steps=args.max_steps)
    if args.drag_sign is not None:
        config = replace(config, drag_sign=args.drag_sign)
    if args.kp is not None:
        config = replace(config, kp=args.kp)
    if args.ki is not None:
        config = replace(config, ki=args.ki)
    if args.kd is not None:
        config = replace(config, kd=args.kd)
    if args.focus_click:
        config = replace(config, focus_click=True)

    if args.sleep > 0:
        print(f"Starting in {args.sleep:.1f}s...")
        time.sleep(args.sleep)

    focus_wurm_center(config)

    initial = wait_compass_available(config)
    print(f"Initial heading: {initial.heading} {initial.direction_degrees:03d} deg")

    drag_sign = calibrate_drag_sign(config)
    print(f"Drag sign: {drag_sign}")

    for raw_target in split_sequence(args.sequence):
        target_name, target_degrees = parse_heading_target(raw_target)
        print(f"\nTurning to {target_name} ({target_degrees:03d} deg)")
        result = turn_to_heading(target_name, target_degrees, config, drag_sign=drag_sign)
        assert result.final_reading.direction_degrees is not None

        for step in result.steps:
            after = "none" if step.after_degrees is None else f"{step.after_degrees:03d}"
            print(
                f"  step {step.step}: before={step.before_degrees:03d} "
                f"error={step.error_degrees:+.1f} drag={step.drag_pixels:+d}px after={after}"
            )

        safe_name = "".join(char for char in target_name if char.isalnum() or char in ("-", "_"))
        output = SCREENS_DIR / f"rotation_{timestamp()}_{safe_name}.png"
        save_compass_diagnostic(result.final_reading, output)
        print(f"Final: {result.final_reading.heading} {result.final_reading.direction_degrees:03d} deg")
        print(f"Saved diagnostic: {output}")


def split_sequence(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise RuntimeError("--sequence must contain at least one heading")
    return values


if __name__ == "__main__":
    main()
