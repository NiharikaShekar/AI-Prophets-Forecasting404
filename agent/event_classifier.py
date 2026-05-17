import re
from dataclasses import dataclass
from enum import Enum


class EventType(str, Enum):
    BINARY = "binary"
    MULTI = "multi"
    NUMERIC = "numeric"


@dataclass
class ClassifiedEvent:
    type: EventType
    outcomes: list[str]

    @property
    def n(self) -> int:
        return len(self.outcomes)


_NUMERIC_RE = re.compile(
    r"^(\d+(\.\d+)?(\s*(or\s*(above|below|more|fewer|less)|[-–]\s*\d+))?)$",
    re.IGNORECASE,
)


def _looks_numeric(outcomes: list[str]) -> bool:
    if len(outcomes) < 3:
        return False
    hits = sum(1 for o in outcomes if _NUMERIC_RE.match(o.strip()))
    return hits >= len(outcomes) * 0.6


def classify(event: dict) -> ClassifiedEvent:
    outcomes = event.get("outcomes") or []

    if len(outcomes) <= 2:
        return ClassifiedEvent(type=EventType.BINARY, outcomes=outcomes)

    if _looks_numeric(outcomes):
        return ClassifiedEvent(type=EventType.NUMERIC, outcomes=outcomes)

    return ClassifiedEvent(type=EventType.MULTI, outcomes=outcomes)
