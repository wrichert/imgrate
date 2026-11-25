"""Copying or linking selected files out of the photo folder (shared by CLI and TUI)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from lazyrate.ratings import MAX_STARS, REJECTED


@dataclass(frozen=True, slots=True)
class ExportOptions:
    destination: Path
    is_flat: bool
    is_symlink: bool
    is_dry_run: bool = False


@dataclass(frozen=True, slots=True)
class ExportSummary:
    exported_count: int
    skipped_existing_count: int
    missing_count: int
    target_paths: list[Path] = field(default_factory=list)
    """Targets of all exported (or, in a dry run, would-be exported) files."""


def select_export_ratings(*, min_rating: int | None, exact_ratings: list[int] | None) -> frozenset[int]:
    """Ratings to export; empty if the arguments are out of range. Default: 5 stars only."""
    if exact_ratings:
        if any(not REJECTED <= rating <= MAX_STARS for rating in exact_ratings):
            return frozenset()
        return frozenset(exact_ratings)
    effective_min_rating = MAX_STARS if min_rating is None else min_rating
    if not 1 <= effective_min_rating <= MAX_STARS:
        return frozenset()
    return frozenset(range(effective_min_rating, MAX_STARS + 1))


def validate_destination(*, root: Path, destination: Path) -> Path:
    """Resolve ``destination``; it must not lie inside the photo folder (copies would be rescanned and rated)."""
    resolved_destination = destination.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    if resolved_destination == resolved_root or resolved_destination.is_relative_to(resolved_root):
        raise ValueError(f"Destination {resolved_destination} is inside the photo folder {resolved_root}")
    if resolved_destination.exists() and not resolved_destination.is_dir():
        raise ValueError(f"Destination {resolved_destination} is not a directory")
    return resolved_destination


def build_unique_flat_name(*, relative_path: str, used_names: set[str]) -> str:
    """``2023/summer/IMG_1.jpg`` -> ``IMG_1.jpg``; on collision ``2023_summer_IMG_1.jpg``. Records the name."""
    file_name = Path(relative_path).name
    candidate_name = file_name if file_name not in used_names else relative_path.replace("/", "_")
    used_names.add(candidate_name)
    return candidate_name


def export_files(*, root: Path, relative_paths: list[str], options: ExportOptions) -> ExportSummary:
    """Copy (or symlink) files into ``options.destination``. Existing targets are never overwritten."""
    destination = validate_destination(root=root, destination=options.destination)
    exported_count = 0
    skipped_existing_count = 0
    missing_count = 0
    target_paths: list[Path] = []
    used_flat_names: set[str] = set()
    for relative_path in sorted(relative_paths):
        source_path = root / relative_path
        if not source_path.is_file():
            missing_count += 1
            continue
        target_relative_path = (
            build_unique_flat_name(relative_path=relative_path, used_names=used_flat_names)
            if options.is_flat
            else relative_path
        )
        target_path = destination / target_relative_path
        if target_path.exists() or target_path.is_symlink():
            skipped_existing_count += 1
            continue
        target_paths.append(target_path)
        exported_count += 1
        if options.is_dry_run:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if options.is_symlink:
            target_path.symlink_to(source_path)
        else:
            shutil.copy2(source_path, target_path)
    return ExportSummary(
        exported_count=exported_count,
        skipped_existing_count=skipped_existing_count,
        missing_count=missing_count,
        target_paths=target_paths,
    )
