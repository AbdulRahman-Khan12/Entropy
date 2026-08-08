"""Stage 5 - temporal resolution and event ordering.

Turns the raw DATE/TIME strings captured in Stage 3/4 ("Tuesday", "next month",
"in 1997") into ISO-8601 values anchored to each document's creation time, then
orders the Stage 4 events on that timeline.

Deliberately dependency-free: normalisation is hand-rolled on ``datetime`` and
``re`` rather than dateutil/HeidelTime, so nothing new has to fit inside the
~1 GB Streamlit Community Cloud budget.

Public surface:
    resolve_timex(text, dct, prefer)     -> TimexValue | None
    resolve_event_time(event, dct)       -> TimexValue | None
    order_events(timed_events)           -> list[TemporalLink]
    build_timeline(timed_events)         -> list[TimedEvent] sorted
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, fields
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "TimexValue",
    "TimedEvent",
    "TemporalLink",
    "resolve_timex",
    "resolve_event_time",
    "attach_times",
    "order_events",
    "build_timeline",
    "write_timex_csv",
    "write_timed_events_csv",
    "write_temporal_links_csv",
]


# --------------------------------------------------------------------------
# lexical tables
# --------------------------------------------------------------------------
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2,
    "wed": 2, "thursday": 3, "thu": 3, "thurs": 3, "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "hundred": 100,
}
UNITS = {
    "second": "S", "seconds": "S", "minute": "M", "minutes": "M",
    "hour": "H", "hours": "H", "day": "D", "days": "D", "week": "W",
    "weeks": "W", "month": "MO", "months": "MO", "year": "Y", "years": "Y",
    "decade": "DE", "decades": "DE",
}
GRANULARITY = {"D": "day", "W": "week", "MO": "month", "Y": "year", "DE": "decade"}

# Expressions that carry no resolvable point in time
VAGUE = {
    "recently", "soon", "later", "earlier", "now", "currently", "today's",
    "the future", "the past", "some time", "sometime", "eventually",
    "previously", "afterwards", "afterward", "then", "meanwhile", "nowadays",
}
# Sentence cues that flip a bare weekday from "most recent past" to "next"
FUTURE_CUES = re.compile(
    r"\b(will|shall|plans? to|planned|scheduled|set to|due to|expects? to|"
    r"expected to|is to|are to|upcoming|next|going to|slated)\b",
    re.IGNORECASE,
)
# Discourse cues used when neither event resolves to a point in time
ORDER_CUES = {
    "after": "AFTER", "following": "AFTER", "since": "AFTER", "once": "AFTER",
    "before": "BEFORE", "prior to": "BEFORE", "ahead of": "BEFORE",
    "until": "BEFORE", "by": "BEFORE",
    "during": "SIMULTANEOUS", "while": "SIMULTANEOUS", "as": "SIMULTANEOUS",
    "when": "SIMULTANEOUS", "amid": "SIMULTANEOUS",
}


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TimexValue:
    """A normalised temporal expression.

    ``value`` is ISO-8601 and may be partial: "2005-08-09", "2005-08", "2005",
    "P16D" for durations, "" when the expression is real but unresolvable
    ("recently"). ``sort_key`` is always populated so a timeline can be built
    even from partial values.
    """

    text: str
    value: str
    granularity: str      # day | month | year | decade | time | duration | set | vague
    timex_type: str       # DATE | TIME | DURATION | SET | UNRESOLVED
    anchor: str           # the DCT this was resolved against, ISO
    method: str           # which rule fired
    confidence: float
    sort_key: str = ""    # padded ISO for ordering; "" when unorderable

    @property
    def is_point(self) -> bool:
        return self.timex_type in {"DATE", "TIME"} and bool(self.value)


@dataclass(frozen=True)
class TimedEvent:
    """A Stage 4 event with its resolved position on the timeline."""

    event_id: str
    doc_id: str
    sent_id: int
    event_type: str
    trigger_word: str
    agent: str
    agent_key: str
    patient: str
    patient_key: str
    location: str
    time_expr: str
    time_value: str
    time_granularity: str
    time_method: str
    time_confidence: float
    sort_key: str
    dct: str
    negated: bool
    confidence: float
    sentence: str


@dataclass(frozen=True)
class TemporalLink:
    """An ordering assertion between two events in the same document."""

    doc_id: str
    source_event_id: str
    target_event_id: str
    relation: str      # BEFORE | AFTER | SIMULTANEOUS
    basis: str         # resolved_time | discourse_cue | document_order
    confidence: float
    source_trigger: str
    target_trigger: str


TIMEX_FIELDS = tuple(f.name for f in fields(TimexValue))
TIMED_EVENT_FIELDS = tuple(f.name for f in fields(TimedEvent))
TEMPORAL_LINK_FIELDS = tuple(f.name for f in fields(TemporalLink))


# --------------------------------------------------------------------------
# calendar arithmetic
# --------------------------------------------------------------------------
def _clamp_day(year: int, month: int, day: int) -> date:
    """Build a date, pulling the day back to the end of a short month."""
    month_lengths = [31, 29 if _leap(year) else 28, 31, 30, 31, 30,
                     31, 31, 30, 31, 30, 31]
    return date(year, month, min(day, month_lengths[month - 1]))


def _leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _shift(anchor: date, amount: int, unit: str) -> date:
    """Move a date by n units, handling month/year rollover properly."""
    if unit == "D":
        return anchor + timedelta(days=amount)
    if unit == "W":
        return anchor + timedelta(weeks=amount)
    if unit == "MO":
        total = anchor.year * 12 + (anchor.month - 1) + amount
        return _clamp_day(total // 12, total % 12 + 1, anchor.day)
    if unit == "Y":
        return _clamp_day(anchor.year + amount, anchor.month, anchor.day)
    if unit == "DE":
        return _clamp_day(anchor.year + amount * 10, anchor.month, anchor.day)
    return anchor


def _nearest_weekday(anchor: date, weekday: int, direction: str) -> date:
    """Resolve a bare weekday relative to the DCT.

    News convention: an unqualified weekday in past-tense prose means the most
    recent one ("landed Tuesday"), which is why ``direction`` defaults to past
    unless the sentence carries a future cue.
    """
    delta = (anchor.weekday() - weekday) % 7
    if direction == "future":
        forward = (weekday - anchor.weekday()) % 7
        return anchor + timedelta(days=forward or 7)
    return anchor - timedelta(days=delta or 7)


def _number(token: str) -> int | None:
    token = token.strip().lower().replace(",", "")
    if token.isdigit():
        return int(token)
    if token in NUM_WORDS:
        return NUM_WORDS[token]
    # "twenty-four", "thirty five"
    parts = re.split(r"[-\s]+", token)
    if len(parts) == 2 and all(p in NUM_WORDS for p in parts):
        return NUM_WORDS[parts[0]] + NUM_WORDS[parts[1]]
    return None


def _sort_key(value: str, granularity: str) -> str:
    """Pad a partial ISO value so string comparison orders it correctly."""
    if not value or granularity in {"duration", "set", "vague"}:
        return ""
    if granularity == "time":
        return value
    parts = value.split("-")
    year = parts[0].zfill(4)
    month = parts[1] if len(parts) > 1 else "01"
    day = parts[2] if len(parts) > 2 else "01"
    return f"{year}-{month}-{day}"


def _make(
    text: str, value: str, granularity: str, timex_type: str,
    anchor: date, method: str, confidence: float,
) -> TimexValue:
    return TimexValue(
        text=text,
        value=value,
        granularity=granularity,
        timex_type=timex_type,
        anchor=anchor.isoformat(),
        method=method,
        confidence=round(confidence, 2),
        sort_key=_sort_key(value, granularity),
    )


# --------------------------------------------------------------------------
# normalisation rules
# --------------------------------------------------------------------------
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_WEEKDAY_RE = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_UNIT_RE = "|".join(sorted(UNITS, key=len, reverse=True))
_MONTH_TITLE_RE = "|".join(
    sorted({m.capitalize() for m in MONTHS}, key=len, reverse=True))
_NUM_RE = r"\d+|" + "|".join(sorted(NUM_WORDS, key=len, reverse=True))

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("iso", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")),
    ("month_day_year", re.compile(
        rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)),
    ("day_month_year", re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.I)),
    ("month_day", re.compile(
        rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{{4}})", re.I)),
    ("day_month", re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\b(?!\s*,?\s*\d{{4}})", re.I)),
    ("month_year", re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{4}})\b", re.I)),
    ("decade", re.compile(r"\b(?:the\s+)?((?:19|20)\d)0'?s\b", re.I)),
    ("clock", re.compile(
        r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(a\.?m\.?|p\.?m\.?)?", re.I)),
    ("military", re.compile(r"\b(\d{2})(\d{2})\s*(GMT|UTC|EDT|EST|CDT|CST|PDT|PST)\b")),
    ("named_time", re.compile(r"\b(noon|midday|midnight)\b", re.I)),
    ("set", re.compile(
        rf"\b(?:every|each)\s+(?:({_NUM_RE})\s+)?({_UNIT_RE})\b|"
        r"\b(daily|weekly|monthly|yearly|annually|hourly|nightly)\b", re.I)),
    ("offset", re.compile(
        rf"\b({_NUM_RE})\s+({_UNIT_RE})\s+(ago|earlier|later|"
        r"from\s+now|before|after)\b", re.I)),
    ("relative_unit", re.compile(
        rf"\b(last|next|this|past|coming|previous|following)\s+({_UNIT_RE})\b", re.I)),
    ("relative_weekday", re.compile(
        rf"\b(?:(last|next|this|coming|past)\s+)?({_WEEKDAY_RE})\b", re.I)),
    ("day_word", re.compile(r"\b(yesterday|today|tonight|tomorrow)\b", re.I)),
    # Case-sensitive on purpose: "May"/"March" are also a modal and a verb, and
    # only a capitalised token inside a DATE span is really a month.
    ("bare_month", re.compile(rf"\b({_MONTH_TITLE_RE})\b")),
    ("duration", re.compile(rf"\b({_NUM_RE})[-\s]+({_UNIT_RE})\b", re.I)),
    ("year", re.compile(r"\b((?:1[89]|20)\d{2})\b")),
]


def resolve_timex(
    text: str, dct: date, prefer: str = "past"
) -> TimexValue | None:
    """Normalise one temporal expression against a document creation time.

    ``prefer`` steers under-specified expressions (a bare weekday, a bare
    month) toward the past or the future; callers should pass "future" when the
    surrounding sentence is prospective.
    """
    if not text or not text.strip():
        return None
    raw = text.strip()
    lowered = raw.lower().strip(" .,")

    if lowered in VAGUE:
        return _make(raw, "", "vague", "UNRESOLVED", dct, "vague", 0.2)

    for name, pattern in PATTERNS:
        match = pattern.search(raw)
        if not match:
            continue
        resolved = _apply(name, match, raw, dct, prefer)
        if resolved is not None:
            return resolved
    return _make(raw, "", "vague", "UNRESOLVED", dct, "no_pattern", 0.1)


def _apply(
    name: str, match: re.Match, raw: str, dct: date, prefer: str
) -> TimexValue | None:
    groups = match.groups()

    if name == "iso":
        year, month, day = (int(g) for g in groups[:3])
        return _make(raw, _clamp_day(year, month, day).isoformat(), "day",
                     "DATE", dct, name, 0.98)

    if name in {"month_day_year", "day_month_year"}:
        if name == "month_day_year":
            month, day, year = MONTHS[groups[0].lower()], int(groups[1]), int(groups[2])
        else:
            day, month, year = int(groups[0]), MONTHS[groups[1].lower()], int(groups[2])
        return _make(raw, _clamp_day(year, month, day).isoformat(), "day",
                     "DATE", dct, name, 0.97)

    if name in {"month_day", "day_month"}:
        if name == "month_day":
            month, day = MONTHS[groups[0].lower()], int(groups[1])
        else:
            day, month = int(groups[0]), MONTHS[groups[1].lower()]
        # No year given: pick the reading nearest the DCT.
        candidates = [_clamp_day(dct.year + offset, month, day) for offset in (-1, 0, 1)]
        if prefer == "future":
            future = [c for c in candidates if c >= dct]
            chosen = min(future) if future else max(candidates)
        else:
            past = [c for c in candidates if c <= dct]
            chosen = max(past) if past else min(candidates)
        return _make(raw, chosen.isoformat(), "day", "DATE", dct, name, 0.85)

    if name == "bare_month":
        month = MONTHS[groups[0].lower()]
        candidates = [(dct.year + offset, month) for offset in (-1, 0, 1)]
        current = (dct.year, dct.month)
        if prefer == "future":
            future = [c for c in candidates if c >= current]
            year = min(future)[0] if future else max(candidates)[0]
        else:
            past = [c for c in candidates if c <= current]
            year = max(past)[0] if past else min(candidates)[0]
        return _make(raw, f"{year:04d}-{month:02d}", "month", "DATE", dct, name, 0.75)

    if name == "month_year":
        month, year = MONTHS[groups[0].lower()], int(groups[1])
        return _make(raw, f"{year:04d}-{month:02d}", "month", "DATE", dct, name, 0.95)

    if name == "decade":
        return _make(raw, f"{int(groups[0]) * 10}", "decade", "DATE", dct, name, 0.8)

    if name == "clock":
        hour, minute = int(groups[0]), int(groups[1])
        meridiem = (groups[2] or "").replace(".", "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            return None
        return _make(raw, f"{dct.isoformat()}T{hour:02d}:{minute:02d}", "time",
                     "TIME", dct, name, 0.8)

    if name == "military":
        hour, minute = int(groups[0]), int(groups[1])
        if hour > 23 or minute > 59:
            return None
        return _make(raw, f"{dct.isoformat()}T{hour:02d}:{minute:02d}", "time",
                     "TIME", dct, name, 0.8)

    if name == "named_time":
        hour = 0 if groups[0].lower() == "midnight" else 12
        return _make(raw, f"{dct.isoformat()}T{hour:02d}:00", "time", "TIME",
                     dct, name, 0.75)

    if name == "set":
        if groups[2]:  # daily / weekly / ...
            word = groups[2].lower()
            unit = {"daily": "D", "nightly": "D", "weekly": "W", "monthly": "MO",
                    "yearly": "Y", "annually": "Y", "hourly": "H"}[word]
            return _make(raw, f"P1{unit}", "set", "SET", dct, name, 0.85)
        count = _number(groups[0]) if groups[0] else 1
        unit = UNITS[groups[1].lower()]
        return _make(raw, f"P{count}{unit}", "set", "SET", dct, name, 0.8)

    if name == "offset":
        count = _number(groups[0])
        if count is None:
            return None
        unit = UNITS[groups[1].lower()]
        direction = groups[2].lower().replace(" ", "").replace("\t", "")
        backwards = direction in {"ago", "earlier", "before"}
        if unit in {"S", "M", "H"}:  # sub-day offsets stay at day granularity
            return _make(raw, dct.isoformat(), "day", "DATE", dct, name, 0.6)
        shifted = _shift(dct, -count if backwards else count, unit)
        return _make(raw, shifted.isoformat(), "day", "DATE", dct, name, 0.85)

    if name == "relative_unit":
        qualifier, unit_word = groups[0].lower(), groups[1].lower()
        unit = UNITS[unit_word]
        if unit in {"S", "M", "H"}:
            return None
        step = {"last": -1, "previous": -1, "past": -1,
                "next": 1, "coming": 1, "following": 1,
                "this": 0}[qualifier]
        shifted = _shift(dct, step, unit)
        granularity = GRANULARITY.get(unit, "day")
        if unit == "Y":
            value = f"{shifted.year:04d}"
        elif unit == "MO":
            value = f"{shifted.year:04d}-{shifted.month:02d}"
        elif unit == "DE":
            value = f"{shifted.year // 10 * 10}"
        else:
            value = shifted.isoformat()
        return _make(raw, value, granularity, "DATE", dct, name, 0.85)

    if name == "relative_weekday":
        qualifier = (groups[0] or "").lower()
        weekday = WEEKDAYS[groups[1].lower()]
        if qualifier in {"next", "coming"}:
            direction = "future"
        elif qualifier in {"last", "past"}:
            direction = "past"
        else:
            direction = prefer
        if qualifier == "this":
            monday = dct - timedelta(days=dct.weekday())
            resolved = monday + timedelta(days=weekday)
        else:
            resolved = _nearest_weekday(dct, weekday, direction)
        confidence = 0.85 if qualifier else 0.7
        return _make(raw, resolved.isoformat(), "day", "DATE", dct, name, confidence)

    if name == "day_word":
        word = groups[0].lower()
        offset = {"yesterday": -1, "today": 0, "tonight": 0, "tomorrow": 1}[word]
        return _make(raw, (dct + timedelta(days=offset)).isoformat(), "day",
                     "DATE", dct, name, 0.9)

    if name == "duration":
        count = _number(groups[0])
        if count is None:
            return None
        unit = UNITS[groups[1].lower()]
        prefix = "PT" if unit in {"S", "M", "H"} else "P"
        return _make(raw, f"{prefix}{count}{unit}", "duration", "DURATION",
                     dct, name, 0.8)

    if name == "year":
        return _make(raw, groups[0], "year", "DATE", dct, name, 0.9)

    return None


# --------------------------------------------------------------------------
# event ordering
# --------------------------------------------------------------------------
def _prefers_future(sentence: str, time_prep: str) -> str:
    if time_prep.lower() in {"until", "by"}:
        return "future"
    return "future" if FUTURE_CUES.search(sentence or "") else "past"


def resolve_event_time(event: dict, dct: date) -> TimexValue | None:
    """Resolve one events.csv row to a point on the timeline.

    Falls back to the document creation time when the sentence carries no
    temporal expression - roughly half the corpus - so every event still gets a
    position, flagged by ``method="dct_fallback"`` and low confidence.
    """
    expression = (event.get("time_expr") or "").strip()
    prefer = _prefers_future(event.get("sentence", ""), event.get("time_prep", ""))
    if expression:
        resolved = resolve_timex(expression, dct, prefer=prefer)
        if resolved is not None:
            return resolved
    return _make("", dct.isoformat(), "day", "DATE", dct, "dct_fallback", 0.25)


def attach_times(
    events: Iterable[dict], dct_by_doc: dict[str, date]
) -> list[TimedEvent]:
    """Join events.csv rows to their resolved times."""
    timed: list[TimedEvent] = []
    for event in events:
        doc_id = event["doc_id"]
        dct = dct_by_doc.get(doc_id)
        if dct is None:
            continue
        resolved = resolve_event_time(event, dct)
        timed.append(
            TimedEvent(
                event_id=event["event_id"],
                doc_id=doc_id,
                sent_id=int(event["sent_id"]),
                event_type=event["event_type"],
                trigger_word=event["trigger_word"],
                agent=event.get("agent", ""),
                agent_key=event.get("agent_key", ""),
                patient=event.get("patient", ""),
                patient_key=event.get("patient_key", ""),
                location=event.get("location", ""),
                time_expr=event.get("time_expr", ""),
                time_value=resolved.value if resolved else "",
                time_granularity=resolved.granularity if resolved else "",
                time_method=resolved.method if resolved else "",
                time_confidence=resolved.confidence if resolved else 0.0,
                sort_key=resolved.sort_key if resolved else "",
                dct=dct.isoformat(),
                negated=str(event.get("negated", "")).lower() in {"true", "1"},
                confidence=float(event.get("confidence", 0.0) or 0.0),
                sentence=event.get("sentence", ""),
            )
        )
    return timed


def _cue_between(source: TimedEvent, target: TimedEvent) -> str | None:
    """Discourse cue linking two events in the same sentence."""
    if source.sent_id != target.sent_id:
        return None
    sentence = source.sentence.lower()
    for cue, relation in ORDER_CUES.items():
        if re.search(rf"\b{re.escape(cue)}\b", sentence):
            return relation
    return None


def order_events(
    timed: Sequence[TimedEvent], max_events_per_doc: int = 40
) -> list[TemporalLink]:
    """Pairwise ordering within each document.

    Three bases, in descending reliability: both events resolved to real dates;
    a discourse cue in a shared sentence; narrative order as a last resort.
    """
    by_doc: dict[str, list[TimedEvent]] = {}
    for event in timed:
        by_doc.setdefault(event.doc_id, []).append(event)

    links: list[TemporalLink] = []
    for doc_id, events in by_doc.items():
        events = sorted(events, key=lambda e: (e.sent_id, e.event_id))
        if len(events) > max_events_per_doc:
            events = events[:max_events_per_doc]
        for i, source in enumerate(events):
            for target in events[i + 1:]:
                relation, basis, confidence = _compare(source, target)
                if relation is None:
                    continue
                links.append(
                    TemporalLink(
                        doc_id=doc_id,
                        source_event_id=source.event_id,
                        target_event_id=target.event_id,
                        relation=relation,
                        basis=basis,
                        confidence=round(confidence, 2),
                        source_trigger=source.trigger_word,
                        target_trigger=target.trigger_word,
                    )
                )
    return links


def _compare(
    source: TimedEvent, target: TimedEvent
) -> tuple[str | None, str, float]:
    real = source.time_method != "dct_fallback" and target.time_method != "dct_fallback"
    if real and source.sort_key and target.sort_key:
        if source.sort_key < target.sort_key:
            relation = "BEFORE"
        elif source.sort_key > target.sort_key:
            relation = "AFTER"
        else:
            relation = "SIMULTANEOUS"
        confidence = min(source.time_confidence, target.time_confidence)
        return relation, "resolved_time", confidence

    cue = _cue_between(source, target)
    if cue is not None:
        return cue, "discourse_cue", 0.5

    if source.sent_id != target.sent_id:
        return "BEFORE", "document_order", 0.3
    return None, "", 0.0


def build_timeline(timed: Sequence[TimedEvent]) -> list[TimedEvent]:
    """Corpus-wide chronological ordering; unresolvable events sort last."""
    return sorted(
        timed,
        key=lambda e: (e.sort_key == "", e.sort_key, e.doc_id, e.sent_id),
    )


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def _write(rows: Sequence[dict], path: Path, fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_timex_csv(values: Sequence[TimexValue], path: Path) -> Path:
    return _write([asdict(v) for v in values], path, TIMEX_FIELDS)


def write_timed_events_csv(events: Sequence[TimedEvent], path: Path) -> Path:
    return _write([asdict(e) for e in events], path, TIMED_EVENT_FIELDS)


def write_temporal_links_csv(links: Sequence[TemporalLink], path: Path) -> Path:
    return _write([asdict(link) for link in links], path, TEMPORAL_LINK_FIELDS)
