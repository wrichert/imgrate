"""Photos moved or renamed outside lazyrate keep their rating and cached data; records of photos gone for good can be
pruned.

lazyrate addresses photos by their path relative to the rated folder. A photo that disappears from one path while a
photo with the same file size and modification time appears at another is the same photo, moved: ``mv``, Finder and
most photo tools keep the modification time. Its records (rating, fingerprint, metadata, AI description) follow it.
Only one-to-one matches count; if several vanished or new photos share size and time, none of them is moved.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lazyrate.media import MediaFile
    from lazyrate.store import RatingStore


@dataclass(frozen=True, slots=True)
class FileIdentity:
    size_bytes: int
    modified_timestamp_utc: float


@dataclass(frozen=True, slots=True)
class KnownFile:
    """A path the database has records for; ``identity`` is None if only a rating is stored (nothing to match)."""

    relative_path: str
    identity: FileIdentity | None


@dataclass(frozen=True, slots=True)
class FileMove:
    old_relative_path: str
    new_relative_path: str


@dataclass(frozen=True, slots=True)
class MatchKey:
    identity: FileIdentity
    extension: str


def build_match_key(*, relative_path: str, identity: FileIdentity) -> MatchKey:
    return MatchKey(identity=identity, extension=PurePosixPath(relative_path).suffix.lower())


def find_moved_files(*, known_files: Iterable[KnownFile], current_files: Iterable[MediaFile]) -> list[FileMove]:
    """Vanished known paths matched one-to-one to new paths of the same size, modification time and extension."""
    known_by_path = {known_file.relative_path: known_file for known_file in known_files}
    current_by_path = {media_file.relative_path: media_file for media_file in current_files}
    vanished_by_key: defaultdict[MatchKey, list[str]] = defaultdict(list)
    for relative_path, known_file in known_by_path.items():
        if relative_path in current_by_path or known_file.identity is None:
            continue
        vanished_by_key[build_match_key(relative_path=relative_path, identity=known_file.identity)].append(
            relative_path
        )
    appeared_by_key: defaultdict[MatchKey, list[str]] = defaultdict(list)
    for relative_path, media_file in current_by_path.items():
        if relative_path in known_by_path:
            continue
        identity = FileIdentity(
            size_bytes=media_file.size_bytes, modified_timestamp_utc=media_file.modified_timestamp_utc
        )
        appeared_by_key[build_match_key(relative_path=relative_path, identity=identity)].append(relative_path)
    return sorted(
        (
            FileMove(old_relative_path=vanished_paths[0], new_relative_path=appeared_by_key[key][0])
            for key, vanished_paths in vanished_by_key.items()
            if len(vanished_paths) == 1 and len(appeared_by_key.get(key, ())) == 1
        ),
        key=lambda file_move: file_move.new_relative_path,
    )


def find_vanished_paths(*, known_files: Iterable[KnownFile], current_files: Iterable[MediaFile]) -> list[str]:
    current_paths = {media_file.relative_path for media_file in current_files}
    return sorted(
        known_file.relative_path for known_file in known_files if known_file.relative_path not in current_paths
    )


def relocate_moved_files(*, store: RatingStore, current_files: list[MediaFile]) -> list[FileMove]:
    """Move the records of photos moved or renamed since the last scan to their new paths; returns the moves."""
    file_moves = find_moved_files(known_files=store.load_known_files(), current_files=current_files)
    if file_moves:
        store.move_file_records(file_moves)
    return file_moves
