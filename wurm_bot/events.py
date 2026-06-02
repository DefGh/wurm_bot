from __future__ import annotations

import time
from pathlib import Path

from .config import ACTION_TIMEOUT
from .text import normalize


class EventLogTail:
    def __init__(self, logs_dir: Path):
        self.path = self._latest_event_log(logs_dir)
        self.offset = self.path.stat().st_size

    @staticmethod
    def _latest_event_log(logs_dir: Path) -> Path:
        files = sorted(logs_dir.glob("_Event.*.txt"), key=lambda path: path.stat().st_mtime)
        if not files:
            raise RuntimeError(f"No event log files found in {logs_dir}")
        return files[-1]

    def mark(self) -> None:
        self.offset = self.path.stat().st_size

    def read_new(self) -> list[str]:
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            lines = handle.readlines()
            self.offset = handle.tell()
        return [line.strip() for line in lines if line.strip()]

    def wait_for_relevant(self, timeout: int = ACTION_TIMEOUT) -> list[str]:
        deadline = time.monotonic() + timeout
        collected: list[str] = []
        while time.monotonic() < deadline:
            new_lines = self.read_new()
            collected.extend(new_lines)
            relevant = [line for line in collected if is_relevant_event(line)]
            if relevant:
                return relevant
            time.sleep(0.25)
        return collected


def is_relevant_event(line: str) -> bool:
    text = normalize(line)
    markers = (
        "you improve",
        "you damage",
        "could be improved with a log",
        "too low quality",
        "too far away",
        "must use",
        "will want",
        "notches",
        "need to repair",
        "you repair",
        "doesn't need repairing",
        "too busy",
        "does not need",
    )
    return any(marker in text for marker in markers)


def event_needs_repair(lines: list[str]) -> bool:
    return any("need to repair" in normalize(line) for line in lines)


def event_damaged(lines: list[str]) -> bool:
    return any("you damage" in normalize(line) for line in lines)


def event_needs_log(lines: list[str]) -> bool:
    return any("could be improved with a log" in normalize(line) for line in lines)


def event_log_too_low_quality(lines: list[str]) -> bool:
    return any("log is too low quality" in normalize(line) for line in lines)


def event_too_far_away(lines: list[str]) -> bool:
    return any("too far away" in normalize(line) for line in lines)


def event_action_started_or_done(lines: list[str]) -> bool:
    return any(
        any(marker in normalize(line) for marker in ("you improve", "you damage"))
        for line in lines
    )
