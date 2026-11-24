from __future__ import annotations

from typing import TYPE_CHECKING

from imgrate.cli import build_unique_flat_name, main
from imgrate.store import RatingStore

if TYPE_CHECKING:
    from pathlib import Path


def test_export_copies_only_files_at_or_above_min_rating(photo_root: Path, tmp_path: Path) -> None:
    store = RatingStore(photo_root / RatingStore.DEFAULT_FILE_NAME)
    store.set_rating(relative_path="2023/a.jpg", rating=5)
    store.set_rating(relative_path="2023/sub/c.png", rating=4)
    store.set_rating(relative_path="2024/d.jpg", rating=3)
    store.set_rating(relative_path="gone.jpg", rating=5)
    store.close()
    destination = tmp_path / "calendar"

    exit_code = main(["export", str(photo_root), str(destination), "--min-rating", "4"])

    assert exit_code == 0
    assert sorted(path.relative_to(destination).as_posix() for path in destination.rglob("*.*")) == [
        "2023/a.jpg",
        "2023/sub/c.png",
    ]


def test_export_flat_symlinks(photo_root: Path, tmp_path: Path) -> None:
    store = RatingStore(photo_root / RatingStore.DEFAULT_FILE_NAME)
    store.set_rating(relative_path="2023/a.jpg", rating=5)
    store.close()
    destination = tmp_path / "flat"

    main(["export", str(photo_root), str(destination), "--flat", "--symlink"])

    assert (destination / "a.jpg").is_symlink()


def test_flat_name_collision_uses_relative_path() -> None:
    used_names: set[str] = set()

    first_name = build_unique_flat_name(relative_path="2023/IMG_1.jpg", used_names=used_names)
    second_name = build_unique_flat_name(relative_path="2024/IMG_1.jpg", used_names=used_names)

    assert (first_name, second_name) == ("IMG_1.jpg", "2024_IMG_1.jpg")
