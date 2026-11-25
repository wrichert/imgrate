"""UI-independent rating session: visible file list, sorting, duplicate groups, navigation, rating and undo."""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from lazyrate.duplicates import DuplicateFinder, DuplicateGroup, HashedFile
from lazyrate.media import IMAGE_EXTENSIONS, MediaFile, ScanResult, build_path_sort_key
from lazyrate.ratings import ALL_FILTER, MAX_STARS, REJECTED, RatingBucket, RatingFilter, describe_bucket

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from lazyrate.fingerprints import Fingerprint
    from lazyrate.store import RatingStore


class SortOrder(StrEnum):
    PATH = "path"
    RATING = "rating"
    DATE = "date taken"


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

    def __init__(
        self, *, scan_result: ScanResult, store: RatingStore, duplicate_finder: DuplicateFinder | None = None
    ) -> None:
        self._scan_result = scan_result
        self._store = store
        self._duplicate_finder = duplicate_finder or DuplicateFinder()
        self._ratings = store.load_ratings()
        saved_extensions = store.load_enabled_extensions()
        self._enabled_extensions = (
            saved_extensions if saved_extensions is not None else set(scan_result.extension_counts) & IMAGE_EXTENSIONS
        )
        self._fingerprints = self._select_current_fingerprints(store.load_fingerprints())
        self._duplicate_groups: list[DuplicateGroup] = []
        self._is_duplicates_mode = False
        self._rating_filter = ALL_FILTER
        self._sort_order = SortOrder.PATH
        self._visible_files: list[MediaFile] = []
        self._sections: list[Section] = []
        self._section_counts_cache: dict[int, SectionCounts] | None = None
        self._section_starts: list[int] = []
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

    def resolve_path(self, media_file: MediaFile) -> Path:
        return self.root / media_file.relative_path

    def count_ratings(self) -> dict[RatingBucket, int]:
        """Count files of the enabled formats per rating bucket (ignores the rating filter)."""
        return dict(
            Counter(
                self._ratings.get(media_file.relative_path)
                for media_file in self._scan_result.files
                if media_file.extension in self._enabled_extensions
            )
        )

    def find_relative_paths_with_ratings(self, buckets: frozenset[RatingBucket]) -> list[str]:
        """Files of the enabled formats whose rating is in ``buckets`` (ignores the current view)."""
        return [
            media_file.relative_path
            for media_file in self._scan_result.files
            if self._is_eligible(media_file) and self.get_rating(media_file) in buckets
        ]

    def load_setting(self, key: str) -> str | None:
        return self._store.load_setting(key)

    def save_setting(self, *, key: str, value: str) -> None:
        self._store.save_setting(key=key, value=value)

    def find_files_missing_fingerprints(self) -> list[MediaFile]:
        """Image files (any enabled state) without an up-to-date fingerprint."""
        return [
            media_file
            for media_file in self._scan_result.files
            if media_file.extension in IMAGE_EXTENSIONS and media_file.relative_path not in self._fingerprints
        ]

    # -------------------------------------------------------------- navigation

    def move_to(self, index: int) -> None:
        if not self._visible_files:
            return
        self._current_index = max(0, min(index, len(self._visible_files) - 1))

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

    def set_duplicates_mode(self, *, is_duplicates_mode: bool) -> None:
        self._is_duplicates_mode = is_duplicates_mode
        self._rebuild_visible_files()

    def set_enabled_extensions(self, extensions: set[str]) -> None:
        self._enabled_extensions = set(extensions)
        self._store.save_enabled_extensions(self._enabled_extensions)
        self._rebuild_visible_files()

    def replace_scan(self, scan_result: ScanResult) -> None:
        if scan_result.root != self._scan_result.root:
            raise ValueError(f"Rescan root {scan_result.root} differs from {self._scan_result.root}")
        self._scan_result = scan_result
        self._fingerprints = self._select_current_fingerprints(self._fingerprints)
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
        self._store.set_rating(relative_path=relative_path, rating=rating)
        if rating is None:
            self._ratings.pop(relative_path, None)
        else:
            self._ratings[relative_path] = rating
        return change

    def _push_undo(self, changes: list[RatingChange]) -> None:
        if not changes:
            return
        self._undo_stack.append(changes)
        del self._undo_stack[: -self.MAX_UNDO_DEPTH]

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
        return media_file.extension in self._enabled_extensions

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
        if self._sort_order is SortOrder.RATING:
            # Stable sort: within one rating the path order of the scan is kept.
            matching_files.sort(key=lambda media_file: rank_rating_for_sort(self.get_rating(media_file)))
            build_label = self._format_rating_label
        elif self._sort_order is SortOrder.DATE:
            matching_files.sort(key=self._compute_sort_timestamp)
            build_label = self._format_date_label
        else:
            build_label = format_folder_label
        return SectionedFiles(
            files=matching_files, sections=split_into_sections(matching_files, build_label=build_label)
        )

    def _compute_section_counts(self) -> dict[int, SectionCounts]:
        """Keyed by ``Section.start_index``. In path / date sort a file only moves out by leaving the rating filter
        (its folder or month never changes); in rating sort it moves to the section labelled with its new rating."""
        if self._is_duplicates_mode:
            return {
                section.start_index: SectionCounts(member_count=section.size, moved_out_count=0, moved_in_count=0)
                for section in self._sections
            }
        is_rating_sort = self._sort_order is SortOrder.RATING
        moved_out_counts: Counter[str] = Counter()
        moved_in_counts: Counter[str] = Counter()
        for section in self._sections:
            for media_file in self._visible_files[section.start_index : section.end_index]:
                rating = self.get_rating(media_file)
                if not self._rating_filter.matches(rating):
                    moved_out_counts[section.label] += 1
                    continue
                current_label = describe_bucket(rating)
                if is_rating_sort and current_label != section.label:
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
