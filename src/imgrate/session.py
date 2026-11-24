"""UI-independent rating session: visible file list, navigation, rating and undo."""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from imgrate.media import IMAGE_EXTENSIONS, MediaFile, ScanResult
from imgrate.ratings import ALL_FILTER, RatingBucket, RatingFilter

if TYPE_CHECKING:
    from pathlib import Path

    from imgrate.store import RatingStore


@dataclass(frozen=True, slots=True)
class RatingChange:
    relative_path: str
    previous_rating: int | None
    new_rating: int | None


class RatingSession:
    """Holds the filtered, ordered view over all scanned files and the cursor into it.

    The visible list is only rebuilt when the filter, the formats or the scan change —
    not when a rating changes — so the cursor never jumps while rating through a filtered view.
    """

    MAX_UNDO_DEPTH = 1000

    def __init__(self, *, scan_result: ScanResult, store: RatingStore) -> None:
        self._scan_result = scan_result
        self._store = store
        self._ratings = store.load_ratings()
        saved_extensions = store.load_enabled_extensions()
        self._enabled_extensions = (
            saved_extensions if saved_extensions is not None else set(scan_result.extension_counts) & IMAGE_EXTENSIONS
        )
        self._rating_filter = ALL_FILTER
        self._visible_files: list[MediaFile] = []
        self._current_index = 0
        self._undo_stack: list[RatingChange] = []
        self._rebuild_visible_files(anchor_relative_path=None)

    def __repr__(self) -> str:
        return (
            f"RatingSession(root={self.root!s}, visible={len(self._visible_files)}, "
            f"filter={self._rating_filter.name!r})"
        )

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
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

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

    def move_to(self, index: int) -> None:
        if not self._visible_files:
            return
        self._current_index = max(0, min(index, len(self._visible_files) - 1))

    def move_by(self, delta: int) -> None:
        self.move_to(self._current_index + delta)

    def move_to_next_unrated(self) -> bool:
        """Move to the next unrated file after the cursor (wrapping). Returns whether one was found."""
        file_count = len(self._visible_files)
        for offset in range(1, file_count + 1):
            candidate_index = (self._current_index + offset) % file_count
            if self._visible_files[candidate_index].relative_path not in self._ratings:
                self._current_index = candidate_index
                return True
        return False

    def rate_current(self, rating: int | None) -> RatingChange | None:
        current_file = self.current_file
        if current_file is None:
            return None
        return self.rate(current_file, rating=rating)

    def rate(self, media_file: MediaFile, *, rating: int | None) -> RatingChange:
        change = RatingChange(
            relative_path=media_file.relative_path,
            previous_rating=self._ratings.get(media_file.relative_path),
            new_rating=rating,
        )
        self._apply_rating(relative_path=media_file.relative_path, rating=rating)
        self._undo_stack.append(change)
        del self._undo_stack[: -self.MAX_UNDO_DEPTH]
        return change

    def undo(self) -> RatingChange | None:
        """Revert the last rating change and move the cursor to that file if it is visible."""
        if not self._undo_stack:
            return None
        change = self._undo_stack.pop()
        self._apply_rating(relative_path=change.relative_path, rating=change.previous_rating)
        for index, media_file in enumerate(self._visible_files):
            if media_file.relative_path == change.relative_path:
                self._current_index = index
                break
        return change

    def set_rating_filter(self, rating_filter: RatingFilter) -> None:
        self._rating_filter = rating_filter
        self._rebuild_visible_files(anchor_relative_path=self._current_relative_path())

    def set_enabled_extensions(self, extensions: set[str]) -> None:
        self._enabled_extensions = set(extensions)
        self._store.save_enabled_extensions(self._enabled_extensions)
        self._rebuild_visible_files(anchor_relative_path=self._current_relative_path())

    def replace_scan(self, scan_result: ScanResult) -> None:
        if scan_result.root != self._scan_result.root:
            raise ValueError(f"Rescan root {scan_result.root} differs from {self._scan_result.root}")
        self._scan_result = scan_result
        self._rebuild_visible_files(anchor_relative_path=self._current_relative_path())

    def _apply_rating(self, *, relative_path: str, rating: int | None) -> None:
        self._store.set_rating(relative_path=relative_path, rating=rating)
        if rating is None:
            self._ratings.pop(relative_path, None)
        else:
            self._ratings[relative_path] = rating

    def _current_relative_path(self) -> str | None:
        current_file = self.current_file
        return None if current_file is None else current_file.relative_path

    def _rebuild_visible_files(self, *, anchor_relative_path: str | None) -> None:
        self._visible_files = [
            media_file
            for media_file in self._scan_result.files
            if media_file.extension in self._enabled_extensions
            and self._rating_filter.matches(self._ratings.get(media_file.relative_path))
        ]
        if anchor_relative_path is None or not self._visible_files:
            self._current_index = 0
            return
        # Keep the cursor on the same file, or on the next one in sort order if it got filtered out.
        sort_keys = [media_file.relative_path.lower() for media_file in self._visible_files]
        anchor_index = bisect.bisect_left(sort_keys, anchor_relative_path.lower())
        self._current_index = min(anchor_index, len(self._visible_files) - 1)
