"""UI-independent rating session: visible file list, sorting, duplicate groups, navigation, rating and undo."""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from lazyrate.duplicates import DuplicateFinder, DuplicateGroup, HashedFile
from lazyrate.geo_map import MapPoint
from lazyrate.locations import NO_LOCATION_LABEL, find_files_missing_locations, normalize_place_name
from lazyrate.media import (
    PHOTO_EXTENSIONS,
    RAW_EXTENSIONS,
    MediaFile,
    ScanResult,
    build_path_sort_key,
    find_raw_companions,
)
from lazyrate.ratings import ALL_FILTER, MAX_STARS, REJECTED, RatingBucket, RatingFilter, describe_bucket
from lazyrate.search_filters import PhotoFilterFacts
from lazyrate.store import RatingStore
from lazyrate.xmp import RatingFileMode, RatingFileWriter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from lazyrate.fingerprints import Fingerprint
    from lazyrate.locations import PhotoLocation
    from lazyrate.suggestions import Suggestion


class SortOrder(StrEnum):
    PATH = "path"
    RATING = "rating"
    DATE = "date taken"
    SUGGESTED = "suggested rating"
    """Unrated photos by the learned suggestion (best first, rejects last); needs suggestions."""
    INFORMATIVE = "most informative"
    """Unrated photos the model is least sure about first: rating these teaches it the most."""
    PLACE = "place"
    """By country, region and town where the photo was taken (GPS); needs photos with GPS data."""
    RELEVANCE = "search"
    """Best matches of the search first; only while a search is active."""


BASIC_SORT_ORDERS = (SortOrder.RATING, SortOrder.DATE, SortOrder.PATH)
SUGGESTION_SORT_ORDERS = (SortOrder.SUGGESTED, SortOrder.INFORMATIVE)
NO_SUGGESTION_LABEL = "no suggestion yet"
UNCERTAINTY_BANDS = ((0.75, "very unsure"), (0.5, "unsure"), (0.25, "fairly sure"), (0.0, "sure"))
SEARCH_RANK_BANDS = ((20, "best 20 matches"), (100, "matches 21-100"), (500, "matches 101-500"))
OTHER_SEARCH_RESULTS_LABEL = "other photos"


@dataclass(frozen=True, slots=True)
class RatingChange:
    relative_path: str
    previous_rating: int | None
    new_rating: int | None


@dataclass(frozen=True, slots=True)
class Section:
    """A contiguous run ``[start_index, end_index)`` of the visible list: a folder, rating, month or duplicate group."""

    label: str
    start_index: int
    end_index: int

    @property
    def size(self) -> int:
        return self.end_index - self.start_index


@dataclass(frozen=True, slots=True)
class SectionCounts:
    member_count: int
    """Files that belong to the section now, after re-ratings."""
    moved_out_count: int
    moved_in_count: int

    @property
    def is_pending_resort(self) -> bool:
        return self.moved_out_count > 0 or self.moved_in_count > 0


@dataclass(frozen=True, slots=True)
class SectionPosition:
    """Where the cursor is (1-based numbers for display)."""

    section: Section
    section_number: int
    section_count: int
    member_number: int


@dataclass(frozen=True, slots=True)
class SectionedFiles:
    files: list[MediaFile]
    sections: list[Section]


def split_into_sections(files: list[MediaFile], *, build_label: Callable[[MediaFile], str]) -> list[Section]:
    """One section per run of consecutive files with the same label."""
    sections: list[Section] = []
    start_index = 0
    for index in range(1, len(files) + 1):
        if index == len(files) or build_label(files[index]) != build_label(files[start_index]):
            sections.append(Section(label=build_label(files[start_index]), start_index=start_index, end_index=index))
            start_index = index
    return sections


def format_folder_label(media_file: MediaFile) -> str:
    return media_file.relative_path.rpartition("/")[0] or "(top folder)"


def format_month_label(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime("%B %Y")


def describe_suggestion(suggestion: Suggestion | None) -> str:
    """Section label of an unrated photo in the suggested-rating sort."""
    if suggestion is None:
        return NO_SUGGESTION_LABEL
    if suggestion.suggested_rating is None:
        return "suggested keeper"
    if suggestion.suggested_rating == REJECTED:
        return "suggested rejected"
    return f"suggested {describe_bucket(suggestion.suggested_rating)}"


def rank_suggestion_for_sort(suggestion: Suggestion | None) -> int:
    """Suggested ★★★★★ … ★, then keeper (stars unknown), then rejected, then no suggestion."""
    if suggestion is None:
        return MAX_STARS + 3
    if suggestion.suggested_rating is None:
        return MAX_STARS + 1
    if suggestion.suggested_rating == REJECTED:
        return MAX_STARS + 2
    return MAX_STARS - suggestion.suggested_rating


def describe_uncertainty(uncertainty: float) -> str:
    for threshold, label in UNCERTAINTY_BANDS:
        if uncertainty >= threshold:
            return label
    return UNCERTAINTY_BANDS[-1][1]


def describe_search_rank(rank: int) -> str:
    for rank_limit, label in SEARCH_RANK_BANDS:
        if rank < rank_limit:
            return label
    return OTHER_SEARCH_RESULTS_LABEL


def rank_rating_for_sort(rating: int | None) -> int:
    """Best first: ★★★★★ … ★, then unrated, then rejected."""
    if rating is None:
        return MAX_STARS + 1
    if rating == REJECTED:
        return MAX_STARS + 2
    return MAX_STARS - rating


class RatingSession:
    """Holds the filtered, ordered view over all scanned files and the cursor into it.

    The visible list is only rebuilt when the filter, sort order, formats, mode or scan change —
    not when a rating changes — so the cursor never jumps while rating through a filtered view.
    """

    MAX_UNDO_DEPTH = 1000
    RATING_FILE_MODE_SETTING_KEY = "rating_files"
    RAW_FORMATS_OFFERED_SETTING_KEY = "raw_formats_offered"
    """Set once RAW formats were switched on for a saved format selection from before RAW previews existed."""

    def __init__(
        self, *, scan_result: ScanResult, store: RatingStore, duplicate_finder: DuplicateFinder | None = None
    ) -> None:
        self._scan_result = scan_result
        self._store = store
        self._duplicate_finder = duplicate_finder or DuplicateFinder()
        self._ratings = store.load_ratings()
        self._rating_file_writer: RatingFileWriter | None = None
        self._writer_thread_store: RatingStore | None = None
        """Connection used by the rating file writer's thread (SQLite connections stay on their thread)."""
        self._on_rating_file_error: Callable[[str], None] = lambda _message: None
        self._enabled_extensions = self._load_enabled_extensions()
        self._raw_companions: dict[str, list[MediaFile]] = {}
        self._companion_paths: set[str] = set()
        self._refresh_raw_companions()
        self._fingerprints = self._select_current_fingerprints(store.load_fingerprints())
        self._locations = self._select_current_locations(store.load_locations())
        self._import_file_ratings(self._locations.values())
        saved_mode = store.load_setting(self.RATING_FILE_MODE_SETTING_KEY)
        if saved_mode is not None and saved_mode != RatingFileMode.OFF.value:
            self._start_rating_file_writer(RatingFileMode(saved_mode))
        self._duplicate_groups: list[DuplicateGroup] = []
        self._is_duplicates_mode = False
        self._rating_filter = ALL_FILTER
        self._sort_order = SortOrder.PATH
        self._visible_files: list[MediaFile] = []
        self._sections: list[Section] = []
        self._section_counts_cache: dict[int, SectionCounts] | None = None
        self._section_starts: list[int] = []
        self._section_label_builder: Callable[[MediaFile], str] = format_folder_label
        self._suggestions: dict[str, Suggestion] = {}
        self._view_suggestions: dict[str, Suggestion] = {}
        """Snapshot used for ordering and section labels, so retraining never reshuffles the list mid-rating."""
        self._search_query: str | None = None
        self._search_scores: dict[str, float] = {}
        self._search_section_labels: dict[str, str] = {}
        """Section per scored file as given by the search (place, distance); empty: label by rank."""
        self._search_labels: dict[str, str] = {}
        self._sort_order_before_search = SortOrder.RATING
        self._view_version = 0
        self._current_index = 0
        self._undo_stack: list[list[RatingChange]] = []
        self.rebuild_duplicate_groups()
        self._rebuild_visible_files()

    def __repr__(self) -> str:
        return (
            f"RatingSession(root={self.root!s}, visible={len(self._visible_files)}, "
            f"filter={self._rating_filter.name!r}, sort={self._sort_order.value!r}, "
            f"duplicates_mode={self._is_duplicates_mode})"
        )

    # -------------------------------------------------------------- properties

    @property
    def root(self) -> Path:
        return self._scan_result.root

    @property
    def all_files(self) -> list[MediaFile]:
        """Every scanned file, regardless of formats, filter and sort."""
        return self._scan_result.files

    @property
    def indexable_files(self) -> list[MediaFile]:
        """Every scanned file except RAW files that follow a JPEG twin (fingerprints, GPS and AI use the twin)."""
        return [
            media_file
            for media_file in self._scan_result.files
            if media_file.relative_path not in self._companion_paths
        ]

    @property
    def extension_counts(self) -> dict[str, int]:
        return self._scan_result.extension_counts

    @property
    def enabled_extensions(self) -> frozenset[str]:
        return frozenset(self._enabled_extensions)

    @property
    def rating_filter(self) -> RatingFilter:
        return self._rating_filter

    @property
    def sort_order(self) -> SortOrder:
        return self._sort_order

    @property
    def is_duplicates_mode(self) -> bool:
        return self._is_duplicates_mode

    @property
    def duplicate_group_count(self) -> int:
        return len(self._duplicate_groups)

    @property
    def fingerprinted_count(self) -> int:
        return len(self._fingerprints)

    @property
    def suggestion_count(self) -> int:
        return len(self._suggestions)

    @property
    def search_query(self) -> str | None:
        return self._search_query

    @property
    def locations(self) -> dict[str, PhotoLocation]:
        """Up-to-date GPS data per relative path (also "no GPS" entries); copy."""
        return dict(self._locations)

    @property
    def located_count(self) -> int:
        return sum(1 for location in self._locations.values() if location.coordinates is not None)

    @property
    def available_sort_orders(self) -> tuple[SortOrder, ...]:
        """What ``s`` cycles through: suggestion sorts once a model exists, search while one is active."""
        sort_orders = BASIC_SORT_ORDERS
        if any(location.place is not None for location in self._locations.values()):
            sort_orders += (SortOrder.PLACE,)
        if self._suggestions:
            sort_orders += SUGGESTION_SORT_ORDERS
        if self._search_query is not None:
            sort_orders += (SortOrder.RELEVANCE,)
        return sort_orders

    @property
    def ratings(self) -> dict[str, int]:
        """Copy of all ratings (relative path -> 0..5), e.g. as training labels."""
        return dict(self._ratings)

    @property
    def visible_files(self) -> list[MediaFile]:
        return self._visible_files

    @property
    def visible_count(self) -> int:
        return len(self._visible_files)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def current_file(self) -> MediaFile | None:
        if not self._visible_files:
            return None
        return self._visible_files[self._current_index]

    @property
    def current_rating(self) -> int | None:
        current_file = self.current_file
        return None if current_file is None else self._ratings.get(current_file.relative_path)

    @property
    def sections(self) -> list[Section]:
        return self._sections

    @property
    def view_version(self) -> int:
        """Incremented whenever the visible list (and so its sections) is rebuilt."""
        return self._view_version

    @property
    def current_section_position(self) -> SectionPosition | None:
        if not self._sections:
            return None
        section_offset = bisect.bisect_right(self._section_starts, self._current_index) - 1
        section = self._sections[section_offset]
        return SectionPosition(
            section=section,
            section_number=section_offset + 1,
            section_count=len(self._sections),
            member_number=self._current_index - section.start_index + 1,
        )

    # ----------------------------------------------------------------- queries

    def count_section_members(self, section: Section) -> SectionCounts:
        """Live counts while the list is frozen: re-rated files stay in place until ``refresh_view`` (S).

        A file moves out when its new rating puts it in another rating section or outside the rating filter.
        It moves in (rating sort only) when its new rating is this section's rating. Duplicate groups never change.
        Computed for all sections in one pass and cached until the next rating change or rebuild.
        """
        if self._section_counts_cache is None:
            self._section_counts_cache = self._compute_section_counts()
        return self._section_counts_cache[section.start_index]

    def get_rating(self, media_file: MediaFile) -> int | None:
        return self._ratings.get(media_file.relative_path)

    def get_raw_companions(self, media_file: MediaFile) -> list[MediaFile]:
        """RAW files that follow this JPEG / HEIC (RAW+JPEG shooting)."""
        return self._raw_companions.get(media_file.relative_path, [])

    def get_suggestion(self, media_file: MediaFile) -> Suggestion | None:
        """Latest suggestion (also for rated files); the UI shows it for unrated ones."""
        return self._suggestions.get(media_file.relative_path)

    def find_suggested_best_index(self, section: Section) -> int | None:
        """Index of the member with the highest expected rating, if at least two members have suggestions."""
        scored_indices = [
            (suggestion.expected_stars, index)
            for index in range(section.start_index, section.end_index)
            if (suggestion := self._suggestions.get(self._visible_files[index].relative_path)) is not None
        ]
        if len(scored_indices) < 2:
            return None
        return max(scored_indices)[1]

    def resolve_path(self, media_file: MediaFile) -> Path:
        return self.root / media_file.relative_path

    def count_ratings(self) -> dict[RatingBucket, int]:
        """Count files of the enabled formats per rating bucket (ignores the rating filter)."""
        return dict(
            Counter(
                self._ratings.get(media_file.relative_path)
                for media_file in self._scan_result.files
                if self._is_eligible(media_file)
            )
        )

    def find_relative_paths_with_ratings(self, buckets: frozenset[RatingBucket]) -> list[str]:
        """Files of the enabled formats whose rating is in ``buckets`` (ignores the current view), each followed by
        its RAW companions."""
        relative_paths: list[str] = []
        for media_file in self._scan_result.files:
            if self._is_eligible(media_file) and self.get_rating(media_file) in buckets:
                relative_paths.append(media_file.relative_path)
                relative_paths.extend(companion.relative_path for companion in self.get_raw_companions(media_file))
        return relative_paths

    def load_setting(self, key: str) -> str | None:
        return self._store.load_setting(key)

    def save_setting(self, *, key: str, value: str) -> None:
        self._store.save_setting(key=key, value=value)

    def get_location(self, media_file: MediaFile) -> PhotoLocation | None:
        return self._locations.get(media_file.relative_path)

    def collect_map_points(self) -> list[MapPoint]:
        """Photos with GPS in the current view; while searching only the matches (not the "other photos")."""
        map_points: list[MapPoint] = []
        for media_file in self._visible_files:
            if self._search_query is not None and media_file.relative_path not in self._search_scores:
                continue
            location = self._locations.get(media_file.relative_path)
            if location is None or location.coordinates is None:
                continue
            map_points.append(
                MapPoint(
                    relative_path=media_file.relative_path,
                    coordinates=location.coordinates,
                    rating=self.get_rating(media_file),
                    place=location.place,
                )
            )
        return map_points

    def collect_filter_facts(self) -> list[PhotoFilterFacts]:
        """Camera, lens and date of every photo of the enabled formats, for ``camera:`` / date filters."""
        facts: list[PhotoFilterFacts] = []
        for media_file in self._scan_result.files:
            if not self._is_eligible(media_file):
                continue
            location = self._locations.get(media_file.relative_path)
            captured_timestamp_utc = None if location is None else location.captured_timestamp_utc
            facts.append(
                PhotoFilterFacts(
                    relative_path=media_file.relative_path,
                    camera=None if location is None else location.camera,
                    lens=None if location is None else location.lens,
                    captured_timestamp_utc=captured_timestamp_utc or self._compute_sort_timestamp(media_file),
                )
            )
        return facts

    def find_files_missing_locations(self) -> list[MediaFile]:
        """Image files whose GPS data was not read yet (or changed since)."""
        return find_files_missing_locations(files=self.indexable_files, locations=self._locations)

    def add_locations(self, locations: Iterable[PhotoLocation]) -> int:
        """Register freshly read metadata (the place sort follows on the next rebuild, like re-ratings). Ratings
        found in the files are taken over for unrated photos; returns how many."""
        new_locations = list(locations)
        for location in new_locations:
            self._locations[location.relative_path] = location
        return self._import_file_ratings(new_locations)

    @property
    def rating_file_mode(self) -> RatingFileMode:
        return RatingFileMode.OFF if self._rating_file_writer is None else self._rating_file_writer.mode

    def set_rating_file_mode(self, mode: RatingFileMode) -> int:
        """Store ratings also in XMP files from now on; every existing rating is written once. Returns how many."""
        self._store.save_setting(key=self.RATING_FILE_MODE_SETTING_KEY, value=mode.value)
        if mode is RatingFileMode.OFF:
            if self._rating_file_writer is not None:
                self._rating_file_writer.set_mode(mode)
            return 0
        writer = self._start_rating_file_writer(mode)
        for relative_path, rating in self._ratings.items():
            writer.write(relative_path=relative_path, rating=rating)
        return len(self._ratings)

    def set_rating_file_error_handler(self, on_error: Callable[[str], None]) -> None:
        """Called (on the writer thread) when a rating cannot be written to a file."""
        self._on_rating_file_error = on_error

    @property
    def pending_rating_file_count(self) -> int:
        return 0 if self._rating_file_writer is None else self._rating_file_writer.pending_count

    def wait_for_rating_files(self) -> None:
        if self._rating_file_writer is not None:
            self._rating_file_writer.wait_until_written()

    def find_files_missing_fingerprints(self) -> list[MediaFile]:
        """Image files (any enabled state) without an up-to-date fingerprint."""
        return [
            media_file
            for media_file in self.indexable_files
            if media_file.extension in PHOTO_EXTENSIONS and media_file.relative_path not in self._fingerprints
        ]

    # -------------------------------------------------------------- navigation

    def move_to(self, index: int) -> None:
        if not self._visible_files:
            return
        self._current_index = max(0, min(index, len(self._visible_files) - 1))

    def move_to_file(self, relative_path: str) -> bool:
        """Put the cursor on a visible file; returns whether it is in the view."""
        for index, media_file in enumerate(self._visible_files):
            if media_file.relative_path == relative_path:
                self._current_index = index
                return True
        return False

    def move_by(self, delta: int) -> None:
        self.move_to(self._current_index + delta)

    def move_by_section(self, delta: int) -> None:
        """Jump to the first file of the next / previous section; backwards from inside a section goes to its start."""
        section_position = self.current_section_position
        if section_position is None:
            return
        section_offset = section_position.section_number - 1
        if delta < 0 and section_position.member_number > 1:
            delta += 1
        target_offset = max(0, min(section_offset + delta, len(self._sections) - 1))
        self._current_index = self._sections[target_offset].start_index

    def move_to_next_unrated(self) -> bool:
        """Move to the next unrated file after the cursor (wrapping). Returns whether one was found."""
        file_count = len(self._visible_files)
        for offset in range(1, file_count + 1):
            candidate_index = (self._current_index + offset) % file_count
            if self._visible_files[candidate_index].relative_path not in self._ratings:
                self._current_index = candidate_index
                return True
        return False

    def move_to_first_unrated(self) -> None:
        """First unrated file in the visible order, or the first file if everything is rated."""
        self._current_index = next(
            (
                index
                for index, media_file in enumerate(self._visible_files)
                if media_file.relative_path not in self._ratings
            ),
            0,
        )

    # ----------------------------------------------------------------- ratings

    def rate_current(self, rating: int | None) -> RatingChange | None:
        current_file = self.current_file
        if current_file is None:
            return None
        return self.rate(current_file, rating=rating)

    def rate(self, media_file: MediaFile, *, rating: int | None) -> RatingChange:
        change = self._apply_rating(relative_path=media_file.relative_path, rating=rating)
        self._push_undo([change])
        return change

    def reject_others_in_group(self) -> list[RatingChange]:
        """Duplicates mode: keep the current file, reject every other member of its group (one undo step)."""
        section_position = self.current_section_position
        if not self._is_duplicates_mode or section_position is None:
            return []
        section = section_position.section
        changes = [
            self._apply_rating(relative_path=media_file.relative_path, rating=REJECTED)
            for index, media_file in enumerate(
                self._visible_files[section.start_index : section.end_index], start=section.start_index
            )
            if index != self._current_index
        ]
        self._push_undo(changes)
        return changes

    def accept_suggestion_for_current(self) -> RatingChange | None:
        """Rate the current unrated file with its suggestion; None if rated already or nothing is suggested."""
        current_file = self.current_file
        if current_file is None or current_file.relative_path in self._ratings:
            return None
        suggestion = self._suggestions.get(current_file.relative_path)
        if suggestion is None or suggestion.suggested_rating is None:
            return None
        return self.rate(current_file, rating=suggestion.suggested_rating)

    def accept_suggestions_in_current_section(self) -> list[RatingChange]:
        """Rate every unrated member of the current section with its suggestion (one undo step)."""
        section_position = self.current_section_position
        if section_position is None:
            return []
        section = section_position.section
        changes = [
            self._apply_rating(relative_path=media_file.relative_path, rating=suggestion.suggested_rating)
            for media_file in self._visible_files[section.start_index : section.end_index]
            if media_file.relative_path not in self._ratings
            and (suggestion := self._suggestions.get(media_file.relative_path)) is not None
            and suggestion.suggested_rating is not None
        ]
        self._push_undo(changes)
        return changes

    def undo(self) -> list[RatingChange]:
        """Revert the last rating step and move the cursor to the first affected file if it is visible."""
        if not self._undo_stack:
            return []
        changes = self._undo_stack.pop()
        for change in reversed(changes):
            self._apply_rating(relative_path=change.relative_path, rating=change.previous_rating)
        changed_paths = {change.relative_path for change in changes}
        for index, media_file in enumerate(self._visible_files):
            if media_file.relative_path in changed_paths:
                self._current_index = index
                break
        return changes

    # ------------------------------------------------------------- view setup

    def refresh_view(self) -> None:
        """Re-apply filter and sort to the current ratings (e.g. move re-rated files into their new section)."""
        self._rebuild_visible_files()

    def set_rating_filter(self, rating_filter: RatingFilter) -> None:
        self._rating_filter = rating_filter
        self._rebuild_visible_files()

    def set_sort_order(self, sort_order: SortOrder) -> None:
        self._sort_order = sort_order
        self._rebuild_visible_files()

    def set_suggestions(self, suggestions: dict[str, Suggestion]) -> None:
        """Replace the suggestions; the visible order only follows on the next rebuild (S), like re-ratings."""
        self._suggestions = dict(suggestions)
        if not self._suggestions and self._sort_order in SUGGESTION_SORT_ORDERS:
            self.set_sort_order(SortOrder.RATING)

    def set_search(self, *, query: str, scores: dict[str, float], section_labels: dict[str, str] | None = None) -> None:
        """Show the best matches of ``query`` first (higher score first; files without a score last).

        ``section_labels`` names each scored file's section (town, distance band); without, sections are rank bands.
        """
        if self._sort_order is not SortOrder.RELEVANCE:
            self._sort_order_before_search = self._sort_order
        self._search_query = query
        self._search_scores = dict(scores)
        self._search_section_labels = dict(section_labels or {})
        self._sort_order = SortOrder.RELEVANCE
        self._rebuild_visible_files()
        self._current_index = 0

    def clear_search(self) -> None:
        if self._search_query is None:
            return
        self._search_query = None
        self._search_scores = {}
        self._search_section_labels = {}
        self.set_sort_order(self._sort_order_before_search)

    def set_duplicates_mode(self, *, is_duplicates_mode: bool) -> None:
        self._is_duplicates_mode = is_duplicates_mode
        self._rebuild_visible_files()

    def set_enabled_extensions(self, extensions: set[str]) -> None:
        self._enabled_extensions = set(extensions)
        self._store.save_enabled_extensions(self._enabled_extensions)
        self._refresh_raw_companions()
        self._rebuild_visible_files()

    def replace_scan(self, scan_result: ScanResult) -> None:
        if scan_result.root != self._scan_result.root:
            raise ValueError(f"Rescan root {scan_result.root} differs from {self._scan_result.root}")
        self._scan_result = scan_result
        self._refresh_raw_companions()
        self._fingerprints = self._select_current_fingerprints(self._fingerprints)
        self._locations = self._select_current_locations(self._locations)
        self.rebuild_duplicate_groups()
        self._rebuild_visible_files()

    def add_fingerprints(self, fingerprints: Iterable[Fingerprint]) -> None:
        """Register freshly computed fingerprints; call :meth:`rebuild_duplicate_groups` to regroup."""
        for fingerprint in fingerprints:
            self._fingerprints[fingerprint.relative_path] = fingerprint

    def rebuild_duplicate_groups(self) -> None:
        files_by_path = {media_file.relative_path: media_file for media_file in self._scan_result.files}
        hashed_files = [
            HashedFile(
                relative_path=fingerprint.relative_path,
                difference_hash=fingerprint.difference_hash,
                sort_timestamp=self._compute_sort_timestamp(files_by_path[fingerprint.relative_path]),
            )
            for fingerprint in self._fingerprints.values()
            if fingerprint.difference_hash is not None
        ]
        self._duplicate_groups = self._duplicate_finder.find_groups(hashed_files)
        # Only the duplicates view depends on the groups; a normal view must not reshuffle mid-rating.
        if self._is_duplicates_mode:
            self._rebuild_visible_files()

    # ----------------------------------------------------------------- private

    def _apply_rating(self, *, relative_path: str, rating: int | None) -> RatingChange:
        self._section_counts_cache = None
        change = RatingChange(
            relative_path=relative_path, previous_rating=self._ratings.get(relative_path), new_rating=rating
        )
        companion_paths = [companion.relative_path for companion in self._raw_companions.get(relative_path, [])]
        for affected_path in [relative_path, *companion_paths]:
            self._store_rating(relative_path=affected_path, rating=rating)
        return change

    def _store_rating(self, *, relative_path: str, rating: int | None, is_from_file: bool = False) -> None:
        self._store.set_rating(relative_path=relative_path, rating=rating)
        if rating is None:
            self._ratings.pop(relative_path, None)
        else:
            self._ratings[relative_path] = rating
        if self._rating_file_writer is not None and not is_from_file:
            self._rating_file_writer.write(relative_path=relative_path, rating=rating)

    def _start_rating_file_writer(self, mode: RatingFileMode) -> RatingFileWriter:
        if self._rating_file_writer is None:
            self._rating_file_writer = RatingFileWriter(
                root=self.root, on_resized=self._handle_rewritten_file, on_error=self._report_rating_file_error
            )
        self._rating_file_writer.set_mode(mode)
        return self._rating_file_writer

    def _handle_rewritten_file(self, relative_path: str, size_bytes: int) -> None:
        """Writer thread: an embedded rating changed the photo's size; its caches stay valid."""
        if self._writer_thread_store is None:
            self._writer_thread_store = RatingStore(self._store.database_path)
        self._writer_thread_store.update_file_size(relative_path=relative_path, size_bytes=size_bytes)

    def _report_rating_file_error(self, message: str) -> None:
        self._on_rating_file_error(message)

    def _import_file_ratings(self, locations: Iterable[PhotoLocation]) -> int:
        """Ratings stored in the files (camera, Lightroom, an earlier lazyrate) for photos lazyrate has not rated;
        lazyrate's own rating always wins. A RAW's rating also goes to its unrated JPEG twin."""
        image_by_companion = {
            companion.relative_path: image_path
            for image_path, companions in self._raw_companions.items()
            for companion in companions
        }
        imported_count = 0
        for location in locations:
            if location.file_rating is None or location.relative_path in self._ratings:
                continue
            self._store_rating(relative_path=location.relative_path, rating=location.file_rating, is_from_file=True)
            imported_count += 1
            image_path = image_by_companion.get(location.relative_path)
            if image_path is not None and image_path not in self._ratings:
                self._store_rating(relative_path=image_path, rating=location.file_rating, is_from_file=True)
        if imported_count:
            self._section_counts_cache = None
        return imported_count

    def _load_enabled_extensions(self) -> set[str]:
        """The saved format selection; photos (RAW included) by default. A selection saved before RAW previews
        existed gets the RAW formats once."""
        found_extensions = set(self._scan_result.extension_counts)
        saved_extensions = self._store.load_enabled_extensions()
        if saved_extensions is None:
            return found_extensions & PHOTO_EXTENSIONS
        if self._store.load_setting(self.RAW_FORMATS_OFFERED_SETTING_KEY) is None:
            saved_extensions |= found_extensions & RAW_EXTENSIONS
            self._store.save_enabled_extensions(saved_extensions)
            self._store.save_setting(key=self.RAW_FORMATS_OFFERED_SETTING_KEY, value="1")
        return saved_extensions

    def _refresh_raw_companions(self) -> None:
        """Pair RAW files with their JPEG twins; a twin without a rating takes over its RAW's rating."""
        self._raw_companions = find_raw_companions(self._scan_result.files, enabled_extensions=self._enabled_extensions)
        self._companion_paths = {
            companion.relative_path for companions in self._raw_companions.values() for companion in companions
        }
        for image_path, companions in self._raw_companions.items():
            companion_rating = next(
                (
                    self._ratings[companion.relative_path]
                    for companion in companions
                    if companion.relative_path in self._ratings
                ),
                None,
            )
            if image_path not in self._ratings and companion_rating is not None:
                self._store_rating(relative_path=image_path, rating=companion_rating)

    def _push_undo(self, changes: list[RatingChange]) -> None:
        if not changes:
            return
        self._undo_stack.append(changes)
        del self._undo_stack[: -self.MAX_UNDO_DEPTH]

    def _select_current_locations(self, locations: dict[str, PhotoLocation]) -> dict[str, PhotoLocation]:
        """Drop GPS data of files that vanished or changed since it was read."""
        files_by_path = {media_file.relative_path: media_file for media_file in self._scan_result.files}
        return {
            relative_path: location
            for relative_path, location in locations.items()
            if (media_file := files_by_path.get(relative_path)) is not None
            and location.size_bytes == media_file.size_bytes
            and location.modified_timestamp_utc == media_file.modified_timestamp_utc
        }

    def _select_current_fingerprints(self, fingerprints: dict[str, Fingerprint]) -> dict[str, Fingerprint]:
        """Drop fingerprints of files that vanished or changed since they were computed."""
        current_fingerprints: dict[str, Fingerprint] = {}
        for media_file in self._scan_result.files:
            fingerprint = fingerprints.get(media_file.relative_path)
            if (
                fingerprint is not None
                and fingerprint.size_bytes == media_file.size_bytes
                and fingerprint.modified_timestamp_utc == media_file.modified_timestamp_utc
            ):
                current_fingerprints[media_file.relative_path] = fingerprint
        return current_fingerprints

    def _compute_sort_timestamp(self, media_file: MediaFile) -> float:
        fingerprint = self._fingerprints.get(media_file.relative_path)
        if fingerprint is not None and fingerprint.capture_timestamp is not None:
            return fingerprint.capture_timestamp
        return media_file.modified_timestamp_utc

    def _is_eligible(self, media_file: MediaFile) -> bool:
        return (
            media_file.extension in self._enabled_extensions and media_file.relative_path not in self._companion_paths
        )

    def _rebuild_visible_files(self) -> None:
        anchor_file = self.current_file
        previous_index = self._current_index
        sectioned_files = self._build_duplicate_view() if self._is_duplicates_mode else self._build_sorted_view()
        self._visible_files = sectioned_files.files
        self._sections = sectioned_files.sections
        self._section_counts_cache = None
        self._section_starts = [section.start_index for section in self._sections]
        self._view_version += 1
        self._restore_cursor(anchor_file=anchor_file, previous_index=previous_index)

    def _build_sorted_view(self) -> SectionedFiles:
        matching_files = [
            media_file
            for media_file in self._scan_result.files
            if self._is_eligible(media_file) and self._rating_filter.matches(self.get_rating(media_file))
        ]
        self._view_suggestions = dict(self._suggestions)
        if self._sort_order is SortOrder.RATING:
            # Stable sort: within one rating the path order of the scan is kept.
            matching_files.sort(key=lambda media_file: rank_rating_for_sort(self.get_rating(media_file)))
            build_label = self._format_rating_label
        elif self._sort_order is SortOrder.DATE:
            matching_files.sort(key=self._compute_sort_timestamp)
            build_label = self._format_date_label
        elif self._sort_order is SortOrder.SUGGESTED:
            matching_files.sort(key=self._build_suggested_sort_key)
            build_label = self._format_suggested_label
        elif self._sort_order is SortOrder.INFORMATIVE:
            matching_files.sort(key=self._build_informative_sort_key)
            build_label = self._format_informative_label
        elif self._sort_order is SortOrder.PLACE:
            matching_files.sort(key=self._build_place_sort_key)
            build_label = self._format_place_label
        elif self._sort_order is SortOrder.RELEVANCE:
            matching_files.sort(key=self._build_search_sort_key)
            self._search_labels = self._build_search_labels(matching_files)
            build_label = self._format_search_label
        else:
            build_label = format_folder_label
        self._section_label_builder = build_label
        return SectionedFiles(
            files=matching_files, sections=split_into_sections(matching_files, build_label=build_label)
        )

    def _compute_section_counts(self) -> dict[int, SectionCounts]:
        """Keyed by ``Section.start_index``. A file moves out when it leaves the rating filter or when its section
        label changes with its rating (rating, suggested and informative sort); folders, months and search ranks
        never change with a rating."""
        if self._is_duplicates_mode:
            return {
                section.start_index: SectionCounts(member_count=section.size, moved_out_count=0, moved_in_count=0)
                for section in self._sections
            }
        moved_out_counts: Counter[str] = Counter()
        moved_in_counts: Counter[str] = Counter()
        for section in self._sections:
            for media_file in self._visible_files[section.start_index : section.end_index]:
                rating = self.get_rating(media_file)
                if not self._rating_filter.matches(rating):
                    moved_out_counts[section.label] += 1
                    continue
                current_label = self._section_label_builder(media_file)
                if current_label != section.label:
                    moved_out_counts[section.label] += 1
                    moved_in_counts[current_label] += 1
        return {
            section.start_index: SectionCounts(
                member_count=section.size - moved_out_counts[section.label] + moved_in_counts[section.label],
                moved_out_count=moved_out_counts[section.label],
                moved_in_count=moved_in_counts[section.label],
            )
            for section in self._sections
        }

    def _format_rating_label(self, media_file: MediaFile) -> str:
        return describe_bucket(self.get_rating(media_file))

    def _format_suggested_label(self, media_file: MediaFile) -> str:
        rating = self.get_rating(media_file)
        if rating is not None:
            return describe_bucket(rating)
        return describe_suggestion(self._view_suggestions.get(media_file.relative_path))

    def _build_suggested_sort_key(self, media_file: MediaFile) -> tuple[int, float]:
        """Unrated by suggestion (best expected rating first; surest rejects first), then rated photos by rating."""
        rating = self.get_rating(media_file)
        if rating is not None:
            return (MAX_STARS + 4 + rank_rating_for_sort(rating), 0.0)
        suggestion = self._view_suggestions.get(media_file.relative_path)
        if suggestion is None:
            return (rank_suggestion_for_sort(None), 0.0)
        secondary_key = -suggestion.junk_probability if suggestion.is_junk else -suggestion.expected_stars
        return (rank_suggestion_for_sort(suggestion), secondary_key)

    def _format_informative_label(self, media_file: MediaFile) -> str:
        if self.get_rating(media_file) is not None:
            return "rated"
        suggestion = self._view_suggestions.get(media_file.relative_path)
        return NO_SUGGESTION_LABEL if suggestion is None else describe_uncertainty(suggestion.uncertainty)

    def _build_informative_sort_key(self, media_file: MediaFile) -> tuple[int, float]:
        if self.get_rating(media_file) is not None:
            return (2, 0.0)
        suggestion = self._view_suggestions.get(media_file.relative_path)
        return (1, 0.0) if suggestion is None else (0, -suggestion.uncertainty)

    def _build_search_sort_key(self, media_file: MediaFile) -> tuple[bool, float]:
        score = self._search_scores.get(media_file.relative_path)
        return (score is None, 0.0 if score is None else -score)

    def _build_search_labels(self, sorted_files: list[MediaFile]) -> dict[str, str]:
        """Scored files: the label the search gave, else their rank band; unscored files are "other photos"."""
        search_labels: dict[str, str] = {}
        for rank, media_file in enumerate(sorted_files):
            relative_path = media_file.relative_path
            if relative_path not in self._search_scores:
                break
            if self._search_section_labels:
                search_labels[relative_path] = self._search_section_labels.get(
                    relative_path, OTHER_SEARCH_RESULTS_LABEL
                )
            else:
                search_labels[relative_path] = describe_search_rank(rank)
        return search_labels

    def _build_place_sort_key(self, media_file: MediaFile) -> tuple[bool, str, str, str, float]:
        """Country, region, town (case- and accent-insensitive), then capture time; photos without GPS last."""
        location = self._locations.get(media_file.relative_path)
        if location is None or location.place is None:
            return (True, "", "", "", self._compute_sort_timestamp(media_file))
        country, region, town = (
            normalize_place_name(name) for name in (location.place.country, location.place.region, location.place.town)
        )
        return (False, country, region, town, self._compute_sort_timestamp(media_file))

    def _format_place_label(self, media_file: MediaFile) -> str:
        location = self._locations.get(media_file.relative_path)
        if location is None or location.place is None:
            return NO_LOCATION_LABEL
        return location.place.path_label

    def _format_search_label(self, media_file: MediaFile) -> str:
        return self._search_labels.get(media_file.relative_path, OTHER_SEARCH_RESULTS_LABEL)

    def _format_date_label(self, media_file: MediaFile) -> str:
        return format_month_label(self._compute_sort_timestamp(media_file))

    def _build_duplicate_view(self) -> SectionedFiles:
        """All members of every group that has at least one member matching the rating filter."""
        files_by_path = {media_file.relative_path: media_file for media_file in self._scan_result.files}
        visible_files: list[MediaFile] = []
        sections: list[Section] = []
        for group in self._duplicate_groups:
            members = [
                files_by_path[relative_path]
                for relative_path in group.relative_paths
                if relative_path in files_by_path and self._is_eligible(files_by_path[relative_path])
            ]
            if len(members) < 2 or not any(
                self._rating_filter.matches(self._ratings.get(member.relative_path)) for member in members
            ):
                continue
            first_taken_at = datetime.fromtimestamp(self._compute_sort_timestamp(members[0]), UTC)
            sections.append(
                Section(
                    label=f"group {len(sections) + 1} · {first_taken_at:%Y-%m-%d %H:%M}",
                    start_index=len(visible_files),
                    end_index=len(visible_files) + len(members),
                )
            )
            visible_files.extend(members)
        return SectionedFiles(files=visible_files, sections=sections)

    def _restore_cursor(self, *, anchor_file: MediaFile | None, previous_index: int) -> None:
        if not self._visible_files:
            self._current_index = 0
            return
        if anchor_file is not None:
            for index, media_file in enumerate(self._visible_files):
                if media_file.relative_path == anchor_file.relative_path:
                    self._current_index = index
                    return
        if anchor_file is not None and self._sort_order is SortOrder.PATH and not self._is_duplicates_mode:
            # Keep the place: the next file in path order.
            sort_keys = [build_path_sort_key(media_file.relative_path) for media_file in self._visible_files]
            anchor_index = bisect.bisect_left(sort_keys, build_path_sort_key(anchor_file.relative_path))
            self._current_index = min(anchor_index, len(self._visible_files) - 1)
            return
        self._current_index = min(previous_index, len(self._visible_files) - 1)
