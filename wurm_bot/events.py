from __future__ import annotations

from dataclasses import dataclass
import re
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


@dataclass(frozen=True)
class SkillGain:
    name: str
    amount: float
    value: float
    count: int


class SkillLogTail:
    def __init__(self, logs_dir: Path):
        self.path = self._latest_skill_log(logs_dir)
        self.offset = self.path.stat().st_size

    @staticmethod
    def _latest_skill_log(logs_dir: Path) -> Path:
        files = sorted(logs_dir.glob("_Skills.*.txt"), key=lambda path: path.stat().st_mtime)
        if not files:
            raise RuntimeError(f"No skills log files found in {logs_dir}")
        return files[-1]

    def mark(self) -> None:
        self.offset = self.path.stat().st_size

    def read_new(self) -> list[str]:
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            lines = handle.readlines()
            self.offset = handle.tell()
        return [line.strip() for line in lines if line.strip()]

    def read_gains(self) -> list[SkillGain]:
        return summarize_skill_gains(self.read_new())


SKILL_GAIN_RE = re.compile(
    r"^\[\d{2}:\d{2}:\d{2}\]\s+(.+?)\s+increased by\s+([0-9.,]+)\s+to\s+([0-9.,]+)$",
    re.IGNORECASE,
)


def summarize_skill_gains(lines: list[str]) -> list[SkillGain]:
    totals: dict[str, SkillGain] = {}
    for line in lines:
        match = SKILL_GAIN_RE.match(line)
        if not match:
            continue

        name = match.group(1).strip()
        amount = parse_skill_number(match.group(2))
        value = parse_skill_number(match.group(3))
        previous = totals.get(name)
        if previous is None:
            totals[name] = SkillGain(name=name, amount=amount, value=value, count=1)
        else:
            totals[name] = SkillGain(
                name=name,
                amount=previous.amount + amount,
                value=value,
                count=previous.count + 1,
            )

    return sorted(totals.values(), key=lambda item: (-item.amount, item.name.lower()))


def parse_skill_number(text: str) -> float:
    return float(text.replace(",", "."))


def is_relevant_event(line: str) -> bool:
    text = normalize(line)
    markers = (
        "you improve",
        "you damage",
        "could be improved with a log",
        "could be improved with a lump",
        "could be improved with a string",
        "could be improved with some string",
        "could be improved with a rock",
        "could be improved with rock",
        "could be improved with some rock",
        "could be improved with a stone",
        "could be improved with stone",
        "must be glowing hot",
        "needs to be glowing hot",
        "not hot enough",
        "too cold",
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
        "bend spacetime",
    )
    return any(marker in text for marker in markers)


def event_needs_repair(lines: list[str]) -> bool:
    return any("need to repair" in normalize(line) for line in lines)


def event_damaged(lines: list[str]) -> bool:
    return any("you damage" in normalize(line) for line in lines)


def event_needs_log(lines: list[str]) -> bool:
    return any("could be improved with a log" in normalize(line) for line in lines)


def event_needs_other_tool(lines: list[str]) -> bool:
    markers = (
        "must use",
        "will want",
        "notches",
        "could be improved with a lump",
    )
    return any(any(marker in normalize(line) for marker in markers) for line in lines)


def event_log_too_low_quality(lines: list[str]) -> bool:
    return any("log is too low quality" in normalize(line) for line in lines)


def event_improve_input_too_low_quality(lines: list[str]) -> bool:
    return any("is too low quality to improve" in normalize(line) for line in lines)


def event_input_not_needed(lines: list[str]) -> bool:
    markers = (
        "does not need the touch of",
        "doesn't need the touch of",
    )
    return any(any(marker in normalize(line) for marker in markers) for line in lines)


def event_too_far_away(lines: list[str]) -> bool:
    return any("too far away" in normalize(line) for line in lines)


def event_action_started_or_done(lines: list[str]) -> bool:
    return any(
        any(marker in normalize(line) for marker in ("you start", "you improve", "you damage"))
        for line in lines
    )


def event_self_tool_error(lines: list[str]) -> bool:
    return any("bend spacetime" in normalize(line) for line in lines)
