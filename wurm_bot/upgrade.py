from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
import time

import sxtemp1

from .config import DEBUG_SCREENSHOTS, IMPROVE_KEY, LOGS_DIR, REPAIR_KEY, VITALS_POLL_SECONDS
from .debug import clean_screenshot_history, save_debug_image, save_log_select_diagnostic, save_mouse_diagnostic
from .events import (
    EventLogTail,
    SkillGain,
    SkillLogTail,
    event_action_started_or_done,
    event_damaged,
    event_failed_to_improve,
    event_improve_input_too_low_quality,
    event_improved,
    event_input_not_needed,
    event_log_too_low_quality,
    event_needs_other_tool,
    event_needs_repair,
    event_self_tool_error,
    event_too_far_away,
)
from .models import Candidate, OcrText, Table, candidate_action_point
from .text import normalize
from .vision import find_inventory_rows, is_inventory_item_active, scan, table_is_inventory, text_rows_for_table
from .vitals import Vitals, VitalsPixelOverlay, format_vitals, read_vitals, run_vitals_overlay, sample_summary, save_vitals_diagnostic
from .windows import click_wurm_local, double_click_wurm_local, drag_wurm_local, move_wurm_local, press, screen_to_wurm_local


TARGET_CONTAINER = "container"
TARGET_WORLD = "world"
MAX_CONSECUTIVE_NO_IMPROVE = 5


@dataclass(frozen=True)
class InventoryItemSpec:
    label: str
    aliases: tuple[str, ...]
    need_markers: tuple[str, ...]

    def matches(self, text: str) -> bool:
        cleaned = normalize(text)
        compact = cleaned.replace(" ", "").replace(",", "")
        compact = compact.replace("0", "o").replace("1", "l").replace("|", "l")
        return any(alias in cleaned or alias.replace(" ", "") in compact for alias in self.aliases)


@dataclass(frozen=True)
class UpgradeProfile:
    name: str
    input_item: InventoryItemSpec
    target_mode: str = TARGET_CONTAINER
    water_key: str | None = None
    water_drink_key: str | None = None
    water_wait_seconds: float = 2.0
    forge_title_terms: tuple[str, ...] = ()
    fuel_item: InventoryItemSpec | None = None
    fire_interval_seconds: float = 0.0
    heat_lump_seconds: float = 0.0
    heat_lump_on_start: bool = False

    @property
    def is_world_target(self) -> bool:
        return self.target_mode == TARGET_WORLD

    @property
    def is_smithing(self) -> bool:
        return self.input_item is LUMP_ITEM


@dataclass(frozen=True)
class ScanState:
    image: object
    texts: list[OcrText]
    tables: list[Table]
    candidates: list[Candidate]


@dataclass(frozen=True)
class ImproveResult:
    status: str
    presses: int
    improved: bool = False
    no_improve_attempt: bool = False


class VitalsOverlay:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.overlay: VitalsPixelOverlay | None = None
        if not enabled:
            return

        try:
            self.overlay = VitalsPixelOverlay("Wurm upgrade vitals frames")
        except RuntimeError as error:
            print(f"Vitals overlay disabled: {error}")
            self.enabled = False
            return

        print(f"Vitals overlay enabled: {self.overlay.backend_name}")
        self.update(None, "waiting for vitals...")

    def before_screenshot(self) -> None:
        # The frame overlay avoids the sampled pixels, so it can stay visible
        # during screenshots without polluting vitals reads.
        return

    def update(self, vitals: Vitals | None, message: str | None = None) -> None:
        if not self.enabled or self.overlay is None:
            return

        self.overlay.update_for_wurm(vitals, message)

    def close(self) -> None:
        if self.enabled and self.overlay is not None:
            self.overlay.close()


LOG_ITEM = InventoryItemSpec(
    label="log",
    aliases=("log",),
    need_markers=("could be improved with a log",),
)
STRING_ITEM = InventoryItemSpec(
    label="string",
    aliases=("string", "string of cloth"),
    need_markers=("could be improved with a string", "could be improved with some string"),
)
ROCK_SHARD_ITEM = InventoryItemSpec(
    label="rock shard",
    aliases=("rock shard", "rock shards", "stone shard", "stone shards"),
    need_markers=(
        "could be improved with a rock",
        "could be improved with rock",
        "could be improved with some rock",
        "could be improved with a stone",
        "could be improved with stone",
    ),
)
LUMP_ITEM = InventoryItemSpec(
    label="lump",
    aliases=("lump",),
    need_markers=("could be improved with a lump",),
)
FUEL_ITEM = InventoryItemSpec(
    label="fuel",
    aliases=("log", "kindling", "wood scrap", "peat"),
    need_markers=(),
)


CARPENTRY_PROFILE = UpgradeProfile(name="carpentry", input_item=LOG_ITEM)
CARPENTRY_IN_WORLD_PROFILE = replace(CARPENTRY_PROFILE, name="carpentry_in_world", target_mode=TARGET_WORLD)
TAILORING_PROFILE = UpgradeProfile(name="tailoring", input_item=STRING_ITEM)
MASONRY_PROFILE = UpgradeProfile(name="masonry", input_item=ROCK_SHARD_ITEM)
MASONRY_IN_WORLD_PROFILE = replace(MASONRY_PROFILE, name="masonry_in_world", target_mode=TARGET_WORLD)
SMITHING_PROFILE = UpgradeProfile(
    name="smithing",
    input_item=LUMP_ITEM,
    forge_title_terms=("forge", "furnace", "oven", "smelter"),
    fuel_item=FUEL_ITEM,
    fire_interval_seconds=float(os.environ.get("WURM_SMITHING_FIRE_INTERVAL_SECONDS", "300")),
    heat_lump_seconds=float(os.environ.get("WURM_SMITHING_HEAT_SECONDS", "25")),
    heat_lump_on_start=True,
)


def capture_state(overlay: VitalsOverlay) -> ScanState:
    overlay.before_screenshot()
    image, texts, tables, candidates = scan()
    overlay.update(read_vitals(image))
    return ScanState(image=image, texts=texts, tables=tables, candidates=candidates)


def ensure_vitals_ready(profile: UpgradeProfile, overlay: VitalsOverlay) -> None:
    last_wait_print = 0.0
    while True:
        overlay.before_screenshot()
        image = sxtemp1.screenshot()
        vitals = read_vitals(image)
        overlay.update(vitals)
        blocked_checks = vitals.blocking_checks
        if blocked_checks:
            if try_refill_water(profile, vitals):
                continue

            path = save_vitals_diagnostic(image, vitals)
            problem = ", ".join(
                f"{check.name} ~{check.filled_percent}% {check.status} "
                f"{sample_summary(check)} line={check.line} target={check.target_rgb}"
                for check in blocked_checks
            )
            raise RuntimeError(f"Vitals guard stopped: {problem}. Saved diagnostic: {path}")

        if vitals.stamina.ok:
            return

        now = time.monotonic()
        if now - last_wait_print >= 5.0:
            print(f"Waiting for stamina: {format_vitals(vitals)}")
            last_wait_print = now
        time.sleep(max(0.1, VITALS_POLL_SECONDS))


def try_refill_water(profile: UpgradeProfile, vitals: Vitals) -> bool:
    if vitals.water.ok or not profile.water_key:
        return False

    print(f"Water below threshold; pressing toolbelt key {profile.water_key!r}")
    press(profile.water_key)
    time.sleep(0.2)
    if profile.water_drink_key:
        press(profile.water_drink_key)
    time.sleep(max(0.1, profile.water_wait_seconds))
    return True


def select_input_item(profile: UpgradeProfile, state: ScanState | None, overlay: VitalsOverlay) -> None:
    if state is None:
        state = capture_state(overlay)

    if is_inventory_item_active(state.texts, profile.input_item.matches):
        return

    row = first_inventory_item(state.texts, state.tables, profile.input_item)
    if row is None:
        save_debug_image(state.image, [], f"upgrade_{profile.name}_error_no_{profile.input_item.label}", [])
        raise RuntimeError(f"No visible {profile.input_item.label!r} rows found in Inventory")

    x, y = inventory_row_click_point(row, state.tables)
    print(f"Selecting {profile.input_item.label}: {row.text!r} at ({x}, {y})")
    move_wurm_local(x, y)
    time.sleep(0.15)
    double_click_wurm_local(x, y)
    time.sleep(1.0)


def first_inventory_item(texts: list[OcrText], tables: list[Table], item: InventoryItemSpec) -> OcrText | None:
    rows = find_inventory_rows(texts, tables, item.matches)
    return rows[0] if rows else None


def inventory_row_click_point(row: OcrText, tables: list[Table]) -> tuple[int, int]:
    for table in tables:
        if table.x1 <= row.cx <= table.x2 and table.y1 <= row.cy <= table.y2:
            x = min(max(row.cx, table.name_x1 + 24), table.ql_x1 - 18)
            return x, row.cy
    return row.cx, row.cy


def repair_candidate(candidate: Candidate, log_tail: EventLogTail) -> None:
    log_tail.mark()
    press(REPAIR_KEY)
    repair_lines = log_tail.wait_for_relevant()
    for line in repair_lines:
        print(line)
    if event_too_far_away(repair_lines):
        raise RuntimeError(f"Too far away while repairing {candidate.name}")


def improve_candidate(
    profile: UpgradeProfile,
    candidate: Candidate,
    log_tail: EventLogTail,
    overlay: VitalsOverlay,
    maintenance: "UpgradeMaintenance",
    max_improves: int | None = None,
) -> ImproveResult:
    x, y = candidate_action_point(candidate)
    click_wurm_local(x, y)
    time.sleep(0.15)

    improve_presses = 0
    for attempt in range(4):
        if max_improves is not None and improve_presses >= max_improves:
            return ImproveResult("limit", improve_presses)

        maintenance.before_improve()
        ensure_vitals_ready(profile, overlay)
        log_tail.mark()
        move_wurm_local(x, y)
        time.sleep(0.05)
        press(IMPROVE_KEY)
        improve_presses += 1

        lines = log_tail.wait_for_relevant(done=lambda relevant: improve_event_ready(profile, relevant))
        if not lines:
            raise RuntimeError(f"No relevant event log lines after Improve for {candidate.name}")

        for line in lines:
            print(line)

        improved = event_improved(lines)
        no_improve_attempt = not improved

        if event_too_far_away(lines):
            raise RuntimeError(f"Too far away while improving {candidate.name}")

        if event_log_too_low_quality(lines):
            print(f"Skipping {candidate.name}: active log quality is too low")
            return ImproveResult("skip", improve_presses)

        if event_improve_input_too_low_quality(lines):
            print(f"Skipping {candidate.name}: improve input quality is too low")
            return ImproveResult("skip", improve_presses)

        if event_input_not_needed(lines):
            print(f"Skipping {candidate.name}: active input is not needed")
            return ImproveResult("skip", improve_presses)

        if event_self_tool_error(lines):
            print(f"Skipping {candidate.name}: active item/tool cannot improve itself")
            return ImproveResult("skip", improve_presses)

        if event_needs_profile_input(profile, lines):
            state = capture_state(overlay)
            select_input_item(profile, state, overlay)
            time.sleep(0.2)
            return ImproveResult("continue", improve_presses, improved=improved, no_improve_attempt=no_improve_attempt)

        if profile.is_smithing and event_needs_heated_lump(lines):
            maintenance.heat_lump()
            return ImproveResult("continue", improve_presses, improved=improved, no_improve_attempt=no_improve_attempt)

        if event_needs_repair(lines) or event_damaged(lines):
            repair_candidate(candidate, log_tail)
            time.sleep(0.2)
            return ImproveResult("continue", improve_presses, improved=improved, no_improve_attempt=no_improve_attempt)

        if event_needs_other_tool(lines):
            return ImproveResult("continue", improve_presses, improved=improved, no_improve_attempt=no_improve_attempt)

        if event_failed_to_improve(lines):
            return ImproveResult("continue", improve_presses, improved=improved, no_improve_attempt=no_improve_attempt)

        if event_action_started_or_done(lines):
            return ImproveResult("continue", improve_presses, improved=improved)

        if any("too busy" in line.lower() for line in lines):
            time.sleep(0.8)
            continue

        if attempt == 3:
            raise RuntimeError(f"Unable to improve {candidate.name}; last lines: {lines[-5:]}")

    return ImproveResult("continue", improve_presses)


def event_needs_profile_input(profile: UpgradeProfile, lines: list[str]) -> bool:
    return any(any(marker in normalize(line) for marker in profile.input_item.need_markers) for line in lines)


def improve_event_ready(profile: UpgradeProfile, lines: list[str]) -> bool:
    return (
        event_improved(lines)
        or event_failed_to_improve(lines)
        or event_too_far_away(lines)
        or event_log_too_low_quality(lines)
        or event_improve_input_too_low_quality(lines)
        or event_input_not_needed(lines)
        or event_self_tool_error(lines)
        or event_needs_profile_input(profile, lines)
        or (profile.is_smithing and event_needs_heated_lump(lines))
        or event_needs_repair(lines)
        or event_damaged(lines)
        or event_needs_other_tool(lines)
        or any("too busy" in line.lower() for line in lines)
    )


def event_needs_heated_lump(lines: list[str]) -> bool:
    markers = ("must be glowing hot", "needs to be glowing hot", "not hot enough", "too cold")
    return any(any(marker in normalize(line) for marker in markers) for line in lines)


class UpgradeMaintenance:
    def __init__(self, profile: UpgradeProfile, overlay: VitalsOverlay):
        self.profile = profile
        self.overlay = overlay
        self.last_fire_at = 0.0
        self.did_start_heat = False

    def before_improve(self) -> None:
        if self.profile.fuel_item is not None and self.profile.fire_interval_seconds > 0:
            self.maintain_fire()

        if self.profile.heat_lump_on_start and not self.did_start_heat:
            self.heat_lump()
            self.did_start_heat = True

    def maintain_fire(self) -> None:
        now = time.monotonic()
        if now - self.last_fire_at < self.profile.fire_interval_seconds:
            return

        state = capture_state(self.overlay)
        forge = find_table_by_title(state.tables, self.profile.forge_title_terms)
        if forge is None:
            print(f"Smithing fire maintenance skipped: forge window not found ({self.profile.forge_title_terms})")
            self.last_fire_at = now
            return

        fuel = first_inventory_item(state.texts, state.tables, self.profile.fuel_item)
        if fuel is None:
            print(f"Smithing fire maintenance skipped: fuel not found ({self.profile.fuel_item.label})")
            self.last_fire_at = now
            return

        print(f"Moving fuel to {forge.title!r}: {fuel.text!r}")
        drag_row_to_table(fuel, state.tables, forge)
        self.last_fire_at = time.monotonic()
        time.sleep(0.5)

    def heat_lump(self) -> None:
        if not self.profile.is_smithing or self.profile.heat_lump_seconds <= 0:
            return

        state = capture_state(self.overlay)
        forge = find_table_by_title(state.tables, self.profile.forge_title_terms)
        if forge is None:
            print(f"Smithing lump heating skipped: forge window not found ({self.profile.forge_title_terms})")
            return

        inventory_table = first_inventory_table(state.tables)
        if inventory_table is None:
            print("Smithing lump heating skipped: Inventory table not found")
            return

        lump = first_inventory_item(state.texts, state.tables, self.profile.input_item)
        if lump is not None:
            print(f"Moving lump into {forge.title!r}: {lump.text!r}")
            drag_row_to_table(lump, state.tables, forge)
            time.sleep(0.8)

        print(f"Heating lump for {self.profile.heat_lump_seconds:.1f}s")
        time.sleep(self.profile.heat_lump_seconds)

        state = capture_state(self.overlay)
        forge = find_table_by_title(state.tables, self.profile.forge_title_terms)
        inventory_table = first_inventory_table(state.tables)
        if forge is None or inventory_table is None:
            print("Smithing lump heating stopped: forge or Inventory table disappeared")
            return

        hot_lump = first_table_item(state.texts, forge, self.profile.input_item)
        if hot_lump is None:
            print(f"Smithing lump heating skipped: lump not found in {forge.title!r}")
            return

        print(f"Moving lump back to Inventory: {hot_lump.text!r}")
        drag_row_to_table(hot_lump, state.tables, inventory_table)
        time.sleep(0.8)
        select_input_item(self.profile, None, self.overlay)


def find_table_by_title(tables: list[Table], terms: tuple[str, ...]) -> Table | None:
    for table in tables:
        title = normalize(table.title)
        if any(term in title for term in terms):
            return table
    return None


def first_inventory_table(tables: list[Table]) -> Table | None:
    for table in tables:
        if table_is_inventory(table):
            return table
    return None


def first_table_item(texts: list[OcrText], table: Table, item: InventoryItemSpec) -> OcrText | None:
    for row in text_rows_for_table(texts, table):
        if item.matches(row.text):
            return row
    return None


def drag_row_to_table(row: OcrText, source_tables: list[Table], target: Table) -> None:
    x1, y1 = inventory_row_click_point(row, source_tables)
    x2 = min(target.x2 - 32, max(target.x1 + 32, (target.x1 + target.x2) // 2))
    y2 = min(target.y2 - 32, max(target.header_y + 42, (target.header_y + target.y2) // 2))
    drag_wurm_local(x1, y1, x2, y2)


def print_scan_summary(profile: UpgradeProfile, tables: list[Table], input_rows: list[OcrText], candidates: list[Candidate]) -> None:
    print(f"Profile: {profile.name}")
    print(f"Tables: {[table.title for table in tables]}")
    print(f"{profile.input_item.label.title()} rows: {[(row.text, row.cx, row.cy) for row in input_rows]}")
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


def candidate_streak_key(candidate: Candidate) -> tuple[str, str, int, int]:
    return candidate.table.title, candidate.name, candidate.click_x, candidate.click_y


def run(
    profile: UpgradeProfile,
    dry_run: bool = False,
    limit: int | None = None,
    max_improves: int | None = None,
    overlay_enabled: bool = True,
    target_local: tuple[int, int] | None = None,
) -> None:
    overlay = VitalsOverlay(enabled=overlay_enabled and not dry_run)
    skill_tail: SkillLogTail | None = None
    try:
        state = capture_state(overlay)
        input_rows = find_inventory_rows(state.texts, state.tables, profile.input_item.matches)
        candidates = [world_candidate(profile, target_local)] if profile.is_world_target else state.candidates
        if DEBUG_SCREENSHOTS and dry_run:
            path = save_debug_image(
                state.image,
                candidates,
                f"upgrade_{profile.name}_candidates",
                input_rows,
                profile.input_item.label,
            )
            print(f"Saved candidates screenshot: {path}")

        print_scan_summary(profile, state.tables, input_rows, candidates)

        if dry_run:
            return
        if not candidates:
            save_debug_image(state.image, [], f"upgrade_{profile.name}_error_no_candidates")
            raise RuntimeError("No improvable item candidates found")

        skill_tail = SkillLogTail(LOGS_DIR)
        first_target = candidates[0]
        if not profile.is_world_target:
            click_wurm_local((first_target.table.x1 + first_target.table.x2) // 2, (first_target.table.y1 + first_target.table.y2) // 2)
            time.sleep(0.2)

        select_input_item(profile, state, overlay)

        log_tail = EventLogTail(LOGS_DIR)
        maintenance = UpgradeMaintenance(profile, overlay)
        remaining_candidates = candidates[: min(limit, len(candidates))] if limit else list(candidates)
        improve_presses = 0
        candidate_index = 0
        no_improve_streaks: dict[tuple[str, str, int, int], int] = {}

        while candidate_index < len(remaining_candidates):
            if max_improves is not None and improve_presses >= max_improves:
                print(f"Reached improve press limit: {max_improves}")
                break

            candidate = remaining_candidates[candidate_index]
            candidate_key = candidate_streak_key(candidate)
            x, y = candidate_action_point(candidate)
            print(f"Improving: {candidate.name} at action point ({x}, {y})")
            remaining = None if max_improves is None else max_improves - improve_presses
            result = improve_candidate(profile, candidate, log_tail, overlay, maintenance, max_improves=remaining)
            improve_presses += result.presses

            if result.status == "skip":
                no_improve_streaks.pop(candidate_key, None)
                if profile.is_world_target:
                    break
                remaining_candidates.pop(candidate_index)
            elif result.status == "limit":
                break
            elif result.improved:
                no_improve_streaks[candidate_key] = 0
            elif result.no_improve_attempt:
                streak = no_improve_streaks.get(candidate_key, 0) + 1
                no_improve_streaks[candidate_key] = streak
                print(f"{candidate.name}: no improvement streak {streak}/{MAX_CONSECUTIVE_NO_IMPROVE}")
                if streak >= MAX_CONSECUTIVE_NO_IMPROVE:
                    print(
                        f"Skipping {candidate.name}: "
                        f"{MAX_CONSECUTIVE_NO_IMPROVE} attempts in a row did not improve it"
                    )
                    no_improve_streaks.pop(candidate_key, None)
                    if profile.is_world_target:
                        break
                    remaining_candidates.pop(candidate_index)
                    continue

            time.sleep(0.3)

        print(f"Improve key presses: {improve_presses}")
    finally:
        if skill_tail is not None:
            print_skill_gains(skill_tail.read_gains())
        overlay.close()


def world_candidate(profile: UpgradeProfile, target_local: tuple[int, int] | None) -> Candidate:
    if target_local is None:
        mouse_x, mouse_y = sxtemp1.pyautogui.position()
        target_local = screen_to_wurm_local(mouse_x, mouse_y)
        print(f"Using current mouse position as world target: local {target_local}")

    x, y = target_local
    table = Table(
        title="world",
        x1=max(0, x - 8),
        y1=max(0, y - 8),
        x2=x + 8,
        y2=y + 8,
        header_y=y,
        name_x1=x,
        ql_x1=x + 1,
        weight_x2=x + 1,
    )
    return Candidate(
        table=table,
        name=f"{profile.name} world target",
        x1=max(0, x - 8),
        y1=max(0, y - 8),
        x2=x + 8,
        y2=y + 8,
        click_x=x,
        click_y=y,
        action_pixels=0,
    )


def diagnose_mouse(profile: UpgradeProfile, candidate_index: int) -> None:
    state = capture_state(VitalsOverlay(enabled=False))
    candidates = state.candidates
    input_rows = find_inventory_rows(state.texts, state.tables, profile.input_item.matches)
    save_debug_image(state.image, candidates, f"upgrade_{profile.name}_candidates", input_rows, profile.input_item.label)
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


def diagnose_input_select(profile: UpgradeProfile) -> None:
    state = capture_state(VitalsOverlay(enabled=False))
    rows = find_inventory_rows(state.texts, state.tables, profile.input_item.matches)
    path = save_debug_image(state.image, [], f"upgrade_{profile.name}_candidates", rows, profile.input_item.label)
    print(f"Saved candidates screenshot: {path}")
    print(f"{profile.input_item.label.title()} rows: {[(row.text, row.cx, row.cy) for row in rows]}")
    if not rows:
        raise RuntimeError(f"No {profile.input_item.label} rows found")

    row = rows[0]
    x, y = inventory_row_click_point(row, state.tables)
    move_wurm_local(x, y)
    time.sleep(0.4)
    double_click_wurm_local(x, y)
    time.sleep(1.0)

    after, after_texts, _tables, _candidates = scan()
    mouse_x, mouse_y = sxtemp1.pyautogui.position()
    out_path = save_log_select_diagnostic(after, row, mouse_x, mouse_y)
    print(f"Selected {profile.input_item.label} row: {row.text} at ({x}, {y})")
    print(f"Mouse local position: {screen_to_wurm_local(mouse_x, mouse_y)}")
    print(f"Active {profile.input_item.label} detected: {is_inventory_item_active(after_texts, profile.input_item.matches)}")
    print(f"Saved diagnostic: {out_path}")


def diagnose_window() -> None:
    from .debug import save_debug_image

    region = sxtemp1.find_wurm_region()
    window = sxtemp1.find_wurm_window_macos() if sxtemp1.sys.platform == "darwin" else None
    image = sxtemp1.screenshot()
    path = save_debug_image(image, [], "upgrade_window")
    if window is not None:
        print(f"Window id: {window.window_id}")
    elif sxtemp1.sys.platform == "darwin":
        print("Window id: not found; using rectangle fallback")
    print(f"Window region: {region}")
    print(f"Screenshot size: {image.size}")
    print(f"Screenshot backend: {os.environ.get('WURM_MAC_SCREENSHOT_BACKEND', 'window')}")
    print(f"Saved window diagnostic: {path}")


def diagnose_vitals() -> None:
    image = sxtemp1.screenshot()
    vitals = read_vitals(image)
    path = save_vitals_diagnostic(image, vitals)
    print(format_vitals(vitals))
    print(f"Saved vitals diagnostic: {path}")


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


def build_profile_from_args(profile: UpgradeProfile, args: argparse.Namespace) -> UpgradeProfile:
    fire_interval = profile.fire_interval_seconds
    if args.fire_interval is not None:
        fire_interval = args.fire_interval
    if args.no_fire:
        fire_interval = 0.0

    heat_seconds = profile.heat_lump_seconds
    if args.heat_lump_seconds is not None:
        heat_seconds = args.heat_lump_seconds
    if args.no_lump_heat:
        heat_seconds = 0.0

    return replace(
        profile,
        water_key=args.water_key,
        water_drink_key=args.water_drink_key,
        fire_interval_seconds=fire_interval,
        heat_lump_seconds=heat_seconds,
        heat_lump_on_start=profile.heat_lump_on_start and heat_seconds > 0,
    )


def parse_target_local(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None

    parts = [part.strip() for part in value.split(",", 1)]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--target-local must use x,y format")

    try:
        return int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("--target-local must contain integer x,y values") from error


def main(profile: UpgradeProfile) -> None:
    parser = argparse.ArgumentParser(description=f"Run Wurm upgrade bot profile: {profile.name}")
    parser.add_argument("--dry-run", action="store_true", help="Only scan and save candidate rectangles.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidates to process.")
    parser.add_argument("--max-improves", type=int, default=None, help="Limit total Improve key presses.")
    parser.add_argument("--target-local", type=parse_target_local, default=None, help="World target local coordinate: x,y.")
    parser.add_argument("--water-key", default=os.environ.get("WURM_WATER_TOOLBELT_KEY"), help="Toolbelt key used to drink/refill water.")
    parser.add_argument("--water-drink-key", default=os.environ.get("WURM_WATER_DRINK_KEY"), help="Optional follow-up drink key after --water-key.")
    parser.add_argument("--fire-interval", type=float, default=None, help="Seconds between moving fuel into forge/furnace.")
    parser.add_argument("--no-fire", action="store_true", help="Disable smithing fire maintenance.")
    parser.add_argument("--heat-lump-seconds", type=float, default=None, help="Seconds to leave lump in forge/furnace before moving it back.")
    parser.add_argument("--no-lump-heat", action="store_true", help="Disable smithing lump heat/move cycle.")
    parser.add_argument("--diagnose-mouse", type=int, default=None, help="Move to candidate index and save a diagnostic screenshot.")
    parser.add_argument("--diagnose-input-select", action="store_true", help="Select the profile input item and save a diagnostic screenshot.")
    parser.add_argument("--diagnose-log-select", action="store_true", help="Compatibility alias for --diagnose-input-select.")
    parser.add_argument("--diagnose-window", action="store_true", help="Save a screenshot of the detected Wurm window and print its region.")
    parser.add_argument("--diagnose-windows", action="store_true", help="Print visible macOS windows used for Wurm window matching.")
    parser.add_argument("--diagnose-vitals", action="store_true", help="Print HUD vital pixel statuses and save a diagnostic screenshot.")
    parser.add_argument("--vitals-overlay", action="store_true", help="Show live always-on-top vitals frames.")
    parser.add_argument("--no-overlay", action="store_true", help="Do not show live vitals frames during normal runs.")
    args = parser.parse_args()

    clean_screenshot_history()
    profile = build_profile_from_args(profile, args)

    if args.vitals_overlay:
        run_vitals_overlay()
    elif args.diagnose_windows:
        diagnose_windows()
    elif args.diagnose_vitals:
        diagnose_vitals()
    elif args.diagnose_window:
        diagnose_window()
    elif args.diagnose_input_select or args.diagnose_log_select:
        diagnose_input_select(profile)
    elif args.diagnose_mouse is not None:
        diagnose_mouse(profile, args.diagnose_mouse)
    else:
        run(
            profile,
            dry_run=args.dry_run,
            limit=args.limit,
            max_improves=args.max_improves,
            overlay_enabled=not args.no_overlay,
            target_local=args.target_local,
        )
