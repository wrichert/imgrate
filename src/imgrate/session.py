"""UI-independent rating session: visible file list, sorting, duplicate groups, navigation, rating and undo."""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from imgrate.duplicates import DuplicateFinder, DuplicateGroup, HashedFile
from imgrate.media import IMAGE_EXTENSIONS, MediaFile, ScanResult
from imgrate.ratings import ALL_FILTER, MAX_STARS, REJECTED, RatingBucket, RatingFilter

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from imgrate.fingerprints import Fingerprint
    from imgrate.store import RatingStore


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
class GroupPosition:
    """Where the cursor is in duplicates mode (1-based numbers for display)."""

    group_number: int
    group_count: int
    member_number: int
    member_count: int
    start_index: int
    """Index of the group's first member in the visible list."""


@dataclass(frozen=True, slots=True)
class DuplicateView:
    files: list[MediaFile]
    group_starts: list[int]
    """Visible-list index of each group's first member."""


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
        self._visible_group_starts: list[int] = []
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
    def current_group_position(self) -> GroupPosition | None:
        if not self._is_duplicates_mode or not self._visible_group_starts:
            return None
        group_offset = bisect.bisect_right(self._visible_group_starts, self._current_index) - 1
        start_index = self._visible_group_starts[group_offset]
        end_index = (
            self._visible_group_starts[group_offset + 1]
            if group_offset + 1 < len(self._visible_group_starts)
            else len(self._visible_files)
        )
        return GroupPosition(
            group_number=group_offset + 1,
            group_count=len(self._visible_group_starts),
            member_number=self._current_index - start_index + 1,
            member_count=end_index - start_index,
            start_index=start_index,
        )

    # ----------------------------------------------------------------- queries

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

    def move_by_group(self, delta: int) -> None:
        """Duplicates mode: jump to the first member of the next / previous group."""
        group_position = self.current_group_position
        if group_position is None:
            return
        target_offset = max(0, min(group_position.group_number - 1 + delta, len(self._visible_group_starts) - 1))
        self._current_index = self._visible_group_starts[target_offset]

    def move_to_next_unrated(self) -> bool:
        """Move to the next unrated file after the cursor (wrapping). Returns whether one was found."""
        file_count = len(self._visible_files)
        for offset in range(1, file_count + 1):
            candidate_index = (self._current_index + offset) % file_count
            if self._visible_files[candidate_index].relative_path not in self._ratings:
                self._current_index = candidate_index
                return True
        return False

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
        group_position = self.current_group_position
        if group_position is None:
            return []
        group_end_index = group_position.start_index + group_position.member_count
        changes = [
            self._apply_rating(relative_path=media_file.relative_path, rating=REJECTED)
            for index, media_file in enumerate(
                self._visible_files[group_position.start_index : group_end_index], start=group_position.start_index
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
        if self._is_duplicates_mode:
            duplicate_view = self._build_duplicate_view()
            self._visible_files = duplicate_view.files
            self._visible_group_starts = duplicate_view.group_starts
        else:
            self._visible_files = self._build_sorted_view()
            self._visible_group_starts = []
        self._restore_cursor(anchor_file=anchor_file, previous_index=previous_index)

    def _build_sorted_view(self) -> list[MediaFile]:
        matching_files = [
            media_file
            for media_file in self._scan_result.files
            if self._is_eligible(media_file) and self._rating_filter.matches(self.get_rating(media_file))
        ]
        if self._sort_order is SortOrder.RATING:
            # Stable sort: within one rating the path order of the scan is kept.
            matching_files.sort(key=lambda media_file: rank_rating_for_sort(self.get_rating(media_file)))
        elif self._sort_order is SortOrder.DATE:
            matching_files.sort(key=self._compute_sort_timestamp)
        return matching_files

    def _build_duplicate_view(self) -> DuplicateView:
        """All members of every group that has at least one member matching the rating filter."""
        files_by_path = {media_file.relative_path: media_file for media_file in self._scan_result.files}
        visible_files: list[MediaFile] = []
        group_starts: list[int] = []
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
            group_starts.append(len(visible_files))
            visible_files.extend(members)
        return DuplicateView(files=visible_files, group_starts=group_starts)

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
            sort_keys = [media_file.relative_path.lower() for media_file in self._visible_files]
            anchor_index = bisect.bisect_left(sort_keys, anchor_file.relative_path.lower())
            self._current_index = min(anchor_index, len(self._visible_files) - 1)
            return
        self._current_index = min(previous_index, len(self._visible_files) - 1)
