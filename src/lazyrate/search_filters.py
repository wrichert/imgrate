"""Search filters by camera, lens and date: ``camera:nikon``, ``camera:phone``, ``lens:24-70``, ``2024``, ``2024-08``,
``2024-08-15``, ``2023..2024``, ``2024-06..2024-08``.

They narrow any other search (``madeira camera:nikon``, ``beach 2023``) or stand alone (``camera:phone 2024``:
those photos in date order, one section per month).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from lazyrate.geo_search import SearchResult

if TYPE_CHECKING:
    from collections.abc import Iterable

CAMERA_FILTER_PATTERN = re.compile(r"""(?<!\S)(?:camera|cam):(?:"(?P<quoted>[^"]+)"|(?P<word>\S+))""", re.IGNORECASE)
LENS_FILTER_PATTERN = re.compile(r"""(?<!\S)lens:(?:"(?P<quoted>[^"]+)"|(?P<word>\S+))""", re.IGNORECASE)
DATE_PATTERN_TEXT = r"(?:19|20)\d\d(?:-\d{1,2}(?:-\d{1,2})?)?"
DATE_FILTER_PATTERN = re.compile(
    rf"(?<![\w.,-])(?P<start>{DATE_PATTERN_TEXT})(?:\.\.(?P<end>{DATE_PATTERN_TEXT}))?(?![\w.,])"
)
PHONE_CAMERA_WORDS = ("iphone", "ipad", "pixel", "galaxy", "samsung", "xiaomi", "oneplus", "huawei", "motorola")
"""``camera:phone`` matches cameras whose name contains one of these."""
PHONE_ALIASES = frozenset({"phone", "mobile", "smartphone"})


@dataclass(frozen=True, slots=True)
class DateRange:
    start_timestamp_utc: float
    end_timestamp_utc: float
    """Exclusive."""
    label: str

    def contains(self, timestamp_utc: float) -> bool:
        return self.start_timestamp_utc <= timestamp_utc < self.end_timestamp_utc


@dataclass(frozen=True, slots=True)
class SearchFilters:
    camera_terms: tuple[str, ...]
    lens_terms: tuple[str, ...]
    date_range: DateRange | None
    remaining_text: str
    """The query without the filters (a place, coordinates or a description, or empty)."""

    @property
    def is_active(self) -> bool:
        return bool(self.camera_terms or self.lens_terms or self.date_range)

    @property
    def summary(self) -> str:
        parts = [f"camera:{term}" for term in self.camera_terms] + [f"lens:{term}" for term in self.lens_terms]
        if self.date_range is not None:
            parts.append(self.date_range.label)
        return " ".join(parts)

    def matches(self, facts: PhotoFilterFacts) -> bool:
        if self.date_range is not None and not self.date_range.contains(facts.captured_timestamp_utc):
            return False
        if not all(matches_camera(term, facts.camera) for term in self.camera_terms):
            return False
        lens = (facts.lens or "").casefold()
        return all(term in lens for term in self.lens_terms)


@dataclass(frozen=True, slots=True)
class PhotoFilterFacts:
    relative_path: str
    camera: str | None
    lens: str | None
    captured_timestamp_utc: float
    """Date taken, else the file's modification time."""


def parse_search_filters(text: str) -> SearchFilters:
    """Split ``text`` into filters and the rest. Raises ``ValueError`` for impossible dates (``2024-13``)."""
    camera_terms = tuple(extract_term(match) for match in CAMERA_FILTER_PATTERN.finditer(text))
    text = CAMERA_FILTER_PATTERN.sub(" ", text)
    lens_terms = tuple(extract_term(match) for match in LENS_FILTER_PATTERN.finditer(text))
    text = LENS_FILTER_PATTERN.sub(" ", text)
    date_match = DATE_FILTER_PATTERN.search(text)
    date_range = None
    if date_match is not None:
        date_range = build_date_range(start_text=date_match["start"], end_text=date_match["end"] or date_match["start"])
        text = text[: date_match.start()] + " " + text[date_match.end() :]
    return SearchFilters(
        camera_terms=camera_terms,
        lens_terms=lens_terms,
        date_range=date_range,
        remaining_text=" ".join(text.split()),
    )


def extract_term(match: re.Match[str]) -> str:
    return (match["quoted"] or match["word"]).casefold()


def matches_camera(term: str, camera: str | None) -> bool:
    camera_name = (camera or "").casefold()
    if term in PHONE_ALIASES:
        return any(phone_word in camera_name for phone_word in PHONE_CAMERA_WORDS)
    return term in camera_name


def build_date_range(*, start_text: str, end_text: str) -> DateRange:
    """From the first moment of ``start_text`` to the end of ``end_text`` (each a year, month or day)."""
    start_timestamp_utc = parse_date_start(start_text).timestamp()
    end_timestamp_utc = parse_date_end(end_text).timestamp()
    if end_timestamp_utc <= start_timestamp_utc:
        raise ValueError(f"Date range ends before it starts: {start_text}..{end_text}")
    label = start_text if start_text == end_text else f"{start_text}..{end_text}"
    return DateRange(start_timestamp_utc=start_timestamp_utc, end_timestamp_utc=end_timestamp_utc, label=label)


def parse_date_start(text: str) -> datetime:
    parts = [int(part) for part in text.split("-")]
    year, month, day = ([*parts, 1, 1])[:3]
    return create_utc_datetime(year=year, month=month, day=day)


def parse_date_end(text: str) -> datetime:
    """The first moment after the year, month or day ``text`` names."""
    parts = [int(part) for part in text.split("-")]
    if len(parts) == 1:
        return create_utc_datetime(year=parts[0] + 1, month=1, day=1)
    year, month = parts[0], parts[1]
    if len(parts) == 2:
        create_utc_datetime(year=year, month=month, day=1)  # validates the month
        return create_utc_datetime(year=year + month // 12, month=month % 12 + 1, day=1)
    return create_utc_datetime(year=year, month=month, day=parts[2]) + timedelta(days=1)


def create_utc_datetime(*, year: int, month: int, day: int) -> datetime:
    """EXIF dates have no time zone; lazyrate stores them as if they were UTC, and so compares them."""
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"No such date: {year}-{month:02d}-{day:02d}") from error


def select_matching_paths(filters: SearchFilters, facts: Iterable[PhotoFilterFacts]) -> list[PhotoFilterFacts]:
    return [photo_facts for photo_facts in facts if filters.matches(photo_facts)]


def build_filter_result(filters: SearchFilters, matching_facts: list[PhotoFilterFacts]) -> SearchResult:
    """Filters alone: the matching photos in date order, one section per month."""
    scores = {photo_facts.relative_path: -photo_facts.captured_timestamp_utc for photo_facts in matching_facts}
    section_labels = {
        photo_facts.relative_path: datetime.fromtimestamp(photo_facts.captured_timestamp_utc, tz=UTC).strftime("%Y-%m")
        for photo_facts in matching_facts
    }
    return SearchResult(
        scores=scores,
        section_labels=section_labels,
        summary=f"{len(scores):,} photos · {filters.summary}, oldest first, by month",
    )


def restrict_search_result(result: SearchResult, *, filters: SearchFilters, allowed_paths: set[str]) -> SearchResult:
    """Another search's result narrowed to the photos the filters allow."""
    scores = {relative_path: score for relative_path, score in result.scores.items() if relative_path in allowed_paths}
    section_labels = {
        relative_path: label for relative_path, label in result.section_labels.items() if relative_path in scores
    }
    return SearchResult(
        scores=scores,
        section_labels=section_labels,
        summary=f"{result.summary} · {filters.summary} ({len(scores):,})",
    )
