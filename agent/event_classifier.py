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
    is_multi_label: bool = False

    @property
    def n(self) -> int:
        return len(self.outcomes)


_NUMERIC_RE = re.compile(
    r"^(\d+(\.\d+)?(\s*(or\s*(above|below|more|fewer|less)|[-–]\s*\d+))?)$",
    re.IGNORECASE,
)

_MULTI_LABEL_RE = re.compile(
    r"top\s*[345]|finish(?:es)?\s+in|qualify|semifinal|nominated\s+and",
    re.IGNORECASE,
)

_WINNER_TAKE_ALL_RE = re.compile(
    r"who\s+won|who\s+wins|winner|champion|wins\s+the",
    re.IGNORECASE,
)


def _event_text(event: dict) -> str:
    parts = [
        event.get("title") or "",
        event.get("rules") or "",
        event.get("description") or "",
    ]
    return " ".join(parts)


def is_multi_label(event: dict) -> bool:
    """True when several outcomes can resolve positive (e.g. top 5), not winner-take-all."""
    text = _event_text(event)
    if not text.strip():
        return False
    if _WINNER_TAKE_ALL_RE.search(text):
        return False
    return bool(_MULTI_LABEL_RE.search(text))


def _looks_numeric(outcomes: list[str]) -> bool:
    if len(outcomes) < 3:
        return False
    hits = sum(1 for o in outcomes if _NUMERIC_RE.match(o.strip()))
    return hits >= len(outcomes) * 0.6


def classify(event: dict) -> ClassifiedEvent:
    outcomes = event.get("outcomes") or []

    if len(outcomes) <= 2:
        return ClassifiedEvent(type=EventType.BINARY, outcomes=outcomes, is_multi_label=False)

    if _looks_numeric(outcomes):
        return ClassifiedEvent(type=EventType.NUMERIC, outcomes=outcomes, is_multi_label=False)

    multi_label = is_multi_label(event)
    return ClassifiedEvent(type=EventType.MULTI, outcomes=outcomes, is_multi_label=multi_label)
