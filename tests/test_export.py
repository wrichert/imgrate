from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from imgrate.export import ExportOptions, export_files, select_export_ratings, validate_destination

if TYPE_CHECKING:
    from pathlib import Path


def test_destination_inside_photo_folder_is_refused(photo_root: Path) -> None:
    with pytest.raises(ValueError, match="inside the photo folder"):
        validate_destination(root=photo_root, destination=photo_root / "selection")


def test_existing_targets_are_skipped_not_overwritten(photo_root: Path, tmp_path: Path) -> None:
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "a.jpg").write_bytes(b"keep me")
    options = ExportOptions(destination=destination, is_flat=True, is_symlink=False)

    summary = export_files(root=photo_root, relative_paths=["2023/a.jpg", "2024/d.jpg", "gone.jpg"], options=options)

    assert (summary.exported_count, summary.skipped_existing_count, summary.missing_count) == (1, 1, 1)
    assert (destination / "a.jpg").read_bytes() == b"keep me"
    assert (destination / "d.jpg").is_file()


def test_dry_run_writes_nothing(photo_root: Path, tmp_path: Path) -> None:
    destination = tmp_path / "dry"
    options = ExportOptions(destination=destination, is_flat=False, is_symlink=False, is_dry_run=True)

    summary = export_files(root=photo_root, relative_paths=["2023/a.jpg"], options=options)

    assert summary.target_paths == [destination.resolve() / "2023/a.jpg"]
    assert not destination.exists()


@pytest.mark.parametrize(
    ("min_rating", "exact_ratings", "expected"),
    [
        (None, None, frozenset({5})),
        (4, None, frozenset({4, 5})),
        (None, [0, 3], frozenset({0, 3})),
        (None, [6], frozenset()),
        (0, None, frozenset()),
    ],
)
def test_select_export_ratings(
    min_rating: int | None, exact_ratings: list[int] | None, expected: frozenset[int]
) -> None:
    assert select_export_ratings(min_rating=min_rating, exact_ratings=exact_ratings) == expected
