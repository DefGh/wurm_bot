from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrText:
    text: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


@dataclass(frozen=True)
class Table:
    title: str
    x1: int
    y1: int
    x2: int
    y2: int
    header_y: int
    name_x1: int
    ql_x1: int
    weight_x2: int


@dataclass(frozen=True)
class Candidate:
    table: Table
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    click_x: int
    click_y: int
    action_pixels: int


def candidate_center(candidate: Candidate) -> tuple[int, int]:
    return (candidate.x1 + candidate.x2) // 2, (candidate.y1 + candidate.y2) // 2


def candidate_action_point(candidate: Candidate) -> tuple[int, int]:
    return candidate_center(candidate)
