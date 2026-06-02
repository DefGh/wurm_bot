from __future__ import annotations

import argparse
import os
import time

import sxtemp1

from wurm_bot.config import DEBUG_SCREENSHOTS, IMPROVE_KEY, LOGS_DIR, REPAIR_KEY
from wurm_bot.debug import clean_screenshot_history, save_debug_image, save_log_select_diagnostic, save_mouse_diagnostic
from wurm_bot.events import (
    EventLogTail,
    SkillGain,
    SkillLogTail,
    event_action_started_or_done,
    event_damaged,
    event_log_too_low_quality,
    event_needs_log,
    event_needs_repair,
    event_too_far_away,
)
from wurm_bot.models import Candidate, candidate_action_point
from wurm_bot.vision import find_log_rows, is_log_active, scan
from wurm_bot.windows import click_wurm_local, double_click_wurm_local, move_wurm_local, press, screen_to_wurm_local


def select_log(image, texts, tables) -> None:
    if is_log_active(texts):
        return

    logs = find_log_rows(texts, tables)
    if not logs:
        save_debug_image(image, [], "upgrade_carpenter_error_no_log", [])
        raise RuntimeError("No visible log rows found in Inventory")

    row = logs[0]
    log_x, log_y = log_click_point(row, tables)
    double_click_wurm_local(log_x, log_y)
    time.sleep(0.7)


def log_click_point(row, tables) -> tuple[int, int]:
    for table in tables:
        if table.x1 <= row.cx <= table.x2 and table.y1 <= row.cy <= table.y2:
            return (table.x1 + table.x2) // 2, row.cy
    return row.cx, row.cy


def repair_candidate(candidate: Candidate, log_tail: EventLogTail) -> None:
    log_tail.mark()
    press(REPAIR_KEY)
    repair_lines = log_tail.wait_for_relevant()
    for line in repair_lines:
        print(line)
    if event_too_far_away(repair_lines):
        raise RuntimeError(f"Too far away while repairing {candidate.name}")


def improve_candidate(candidate: Candidate, log_tail: EventLogTail, max_improves: int | None = None) -> tuple[str, int]:
    x, y = candidate_action_point(candidate)
    click_wurm_local(x, y)
    time.sleep(0.15)

    improve_presses = 0
    for attempt in range(4):
        if max_improves is not None and improve_presses >= max_improves:
            return "limit", improve_presses

        log_tail.mark()
        move_wurm_local(x, y)
        time.sleep(0.05)
        press(IMPROVE_KEY)
        improve_presses += 1

        lines = log_tail.wait_for_relevant()
        if not lines:
            raise RuntimeError(f"No relevant event log lines after Improve for {candidate.name}")

        for line in lines:
            print(line)

        if event_too_far_away(lines):
            raise RuntimeError(f"Too far away while improving {candidate.name}")

        if event_log_too_low_quality(lines):
            print(f"Skipping {candidate.name}: active log quality is too low")
            return "skip", improve_presses

        if event_needs_repair(lines) or event_damaged(lines):
            repair_candidate(candidate, log_tail)
            time.sleep(0.2)
            return "continue", improve_presses

        if event_needs_log(lines):
            image, texts, tables, _candidates = scan()
            select_log(image, texts, tables)
            time.sleep(0.2)
            return "continue", improve_presses

        if event_action_started_or_done(lines):
            return "continue", improve_presses

        if any("too busy" in line.lower() for line in lines):
            time.sleep(0.8)
            continue

        if attempt == 3:
            raise RuntimeError(f"Unable to improve {candidate.name}; last lines: {lines[-5:]}")

    return "continue", improve_presses


def print_scan_summary(tables, log_rows, candidates) -> None:
    print(f"Tables: {[table.title for table in tables]}")
    print(f"Logs: {[(row.text, row.cx, row.cy) for row in log_rows]}")
    print(f"Candidates: {len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"{index}. {candidate.name} @ ({candidate.click_x}, {candidate.click_y}), "
            f"table={candidate.table.title!r}, action_pixels={candidate.action_pixels}"
        )


def print_skill_gains(gains: list[SkillGain]) -> None:
    if not gains:
        print("Skill gains: none")
        return

    print("Skill gains:")
    for gain in gains:
        count = f", ticks={gain.count}" if gain.count > 1 else ""
        print(f"- {gain.name}: +{gain.amount:.6f} -> {gain.value:.6f}{count}")


def run(dry_run: bool = False, limit: int | None = None, max_improves: int | None = None) -> None:
    image, texts, tables, candidates = scan()
    log_rows = find_log_rows(texts, tables)
    if DEBUG_SCREENSHOTS and dry_run:
        path = save_debug_image(image, candidates, "upgrade_carpenter_candidates", log_rows)
        print(f"Saved candidates screenshot: {path}")

    print_scan_summary(tables, log_rows, candidates)

    if dry_run:
        return
    if not candidates:
        save_debug_image(image, [], "upgrade_carpenter_error_no_candidates")
        raise RuntimeError("No improvable item candidates found")

    skill_tail = SkillLogTail(LOGS_DIR)
    first_container = candidates[0].table
    try:
        click_wurm_local((first_container.x1 + first_container.x2) // 2, (first_container.y1 + first_container.y2) // 2)
        time.sleep(0.2)
        select_log(image, texts, tables)

        log_tail = EventLogTail(LOGS_DIR)
        remaining_candidates = candidates[: min(limit, len(candidates))] if limit else list(candidates)
        improve_presses = 0
        candidate_index = 0

        while candidate_index < len(remaining_candidates):
            if max_improves is not None and improve_presses >= max_improves:
                print(f"Reached improve press limit: {max_improves}")
                break

            candidate = remaining_candidates[candidate_index]
            x, y = candidate_action_point(candidate)
            print(f"Improving: {candidate.name} at action point ({x}, {y})")
            remaining = None if max_improves is None else max_improves - improve_presses
            status, presses = improve_candidate(candidate, log_tail, max_improves=remaining)
            improve_presses += presses

            if status == "skip":
                remaining_candidates.pop(candidate_index)
            elif status == "limit":
                break
            else:
                # Stay on the same candidate until it is explicitly skipped or the
                # global improve limit is reached.
                pass

            time.sleep(0.3)

        print(f"Improve key presses: {improve_presses}")
    finally:
        print_skill_gains(skill_tail.read_gains())


def diagnose_mouse(candidate_index: int) -> None:
    image, texts, tables, candidates = scan()
    log_rows = find_log_rows(texts, tables)
    save_debug_image(image, candidates, "upgrade_carpenter_candidates", log_rows)
    if not candidates:
        raise RuntimeError("No candidates found for mouse diagnosis")
    if candidate_index < 1 or candidate_index > len(candidates):
        raise RuntimeError(f"Candidate index {candidate_index} is out of range 1..{len(candidates)}")

    candidate = candidates[candidate_index - 1]
    x, y = candidate_action_point(candidate)
    move_wurm_local(x, y)
    time.sleep(0.5)
    mouse_x, mouse_y = sxtemp1.pyautogui.position()
    after = sxtemp1.screenshot()
    path = save_mouse_diagnostic(after, candidate, mouse_x, mouse_y)
    print(f"Candidate: {candidate.name}")
    print(f"Target local action point: ({x}, {y})")
    print(f"Mouse screen position: ({mouse_x}, {mouse_y})")
    print(f"Mouse local position: {screen_to_wurm_local(mouse_x, mouse_y)}")
    print(f"Saved mouse diagnostic: {path}")


def diagnose_log_select() -> None:
    image, texts, tables, candidates = scan()
    logs = find_log_rows(texts, tables)
    path = save_debug_image(image, candidates, "upgrade_carpenter_candidates", logs)
    print(f"Saved candidates screenshot: {path}")
    print(f"Logs: {[(row.text, row.cx, row.cy) for row in logs]}")
    if not logs:
        raise RuntimeError("No log rows found")

    row = logs[0]
    log_x, log_y = log_click_point(row, tables)
    move_wurm_local(log_x, log_y)
    time.sleep(0.4)
    double_click_wurm_local(log_x, log_y)
    time.sleep(1.0)

    after, after_texts, _tables, _candidates = scan()
    mouse_x, mouse_y = sxtemp1.pyautogui.position()
    out_path = save_log_select_diagnostic(after, row, mouse_x, mouse_y)
    print(f"Selected log row: {row.text} at ({log_x}, {log_y})")
    print(f"Mouse local position: {screen_to_wurm_local(mouse_x, mouse_y)}")
    print(f"Active log detected: {is_log_active(after_texts)}")
    print(f"Saved log select diagnostic: {out_path}")


def diagnose_window() -> None:
    region = sxtemp1.find_wurm_region()
    window = sxtemp1.find_wurm_window_macos() if sxtemp1.sys.platform == "darwin" else None
    image = sxtemp1.screenshot()
    path = save_debug_image(image, [], "upgrade_carpenter_window")
    if window is not None:
        print(f"Window id: {window.window_id}")
    elif sxtemp1.sys.platform == "darwin":
        print("Window id: not found; using rectangle fallback")
    print(f"Window region: {region}")
    print(f"Screenshot size: {image.size}")
    print(f"Screenshot backend: {os.environ.get('WURM_MAC_SCREENSHOT_BACKEND', 'window')}")
    print(f"Saved window diagnostic: {path}")


def diagnose_windows() -> None:
    if sxtemp1.sys.platform != "darwin":
        raise RuntimeError("--diagnose-windows is only supported on macOS")

    match = os.environ.get("WURM_WINDOW_MATCH", "Wurm Online").lower()
    exclude = tuple(term.strip().lower() for term in os.environ.get("WURM_WINDOW_EXCLUDE", "").split(",") if term.strip())
    windows = sorted(
        sxtemp1.list_macos_windows_coregraphics(),
        key=lambda item: item.window.width * item.window.height,
        reverse=True,
    )
    for info in windows[:40]:
        haystack = f"{info.owner} {info.title}".lower()
        marker = "*" if match in haystack and not any(term in haystack for term in exclude) else " "
        window = info.window
        print(
            f"{marker} id={window.window_id} layer={info.layer} "
            f"region=({window.x},{window.y},{window.width},{window.height}) "
            f"owner={info.owner!r} title={info.title!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only scan and save candidate rectangles.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidates to process.")
    parser.add_argument("--max-improves", type=int, default=None, help="Limit total Improve key presses.")
    parser.add_argument("--diagnose-mouse", type=int, default=None, help="Move to candidate index and save a diagnostic screenshot.")
    parser.add_argument("--diagnose-log-select", action="store_true", help="Move to the visible log row, double-click it, and save a diagnostic screenshot.")
    parser.add_argument("--diagnose-window", action="store_true", help="Save a screenshot of the detected Wurm window and print its region.")
    parser.add_argument("--diagnose-windows", action="store_true", help="Print visible macOS windows used for Wurm window matching.")
    args = parser.parse_args()

    clean_screenshot_history()

    if args.diagnose_windows:
        diagnose_windows()
    elif args.diagnose_window:
        diagnose_window()
    elif args.diagnose_log_select:
        diagnose_log_select()
    elif args.diagnose_mouse is not None:
        diagnose_mouse(args.diagnose_mouse)
    else:
        run(dry_run=args.dry_run, limit=args.limit, max_improves=args.max_improves)


if __name__ == "__main__":
    main()
