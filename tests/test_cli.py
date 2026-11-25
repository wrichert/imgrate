from __future__ import annotations

from typing import TYPE_CHECKING

from lazyrate.cli import main
from lazyrate.export import build_unique_flat_name
from lazyrate.store import RatingStore

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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


def test_export_exact_ratings(photo_root: Path, tmp_path: Path) -> None:
    store = RatingStore(photo_root / RatingStore.DEFAULT_FILE_NAME)
    store.set_rating(relative_path="2023/a.jpg", rating=5)
    store.set_rating(relative_path="2023/sub/c.png", rating=3)
    store.set_rating(relative_path="2024/d.jpg", rating=0)
    store.close()
    destination = tmp_path / "threes_and_rejects"

    exit_code = main(["export", str(photo_root), str(destination), "--rating", "3", "--rating", "0", "--flat"])

    assert exit_code == 0
    assert sorted(path.name for path in destination.iterdir()) == ["c.png", "d.jpg"]


def test_export_rejects_out_of_range_rating(photo_root: Path, tmp_path: Path) -> None:
    assert main(["export", str(photo_root), str(tmp_path / "x"), "--rating", "7"]) == 2


def test_missing_photo_folder_is_an_error_instead_of_scanning_the_current_directory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([])

    assert exit_code == 2
    assert "photo folder missing" in capsys.readouterr().err


def test_legacy_imgrate_database_is_renamed_and_keeps_ratings(photo_root: Path) -> None:
    legacy_store = RatingStore(photo_root / ".imgrate.sqlite")
    legacy_store.set_rating(relative_path="2023/a.jpg", rating=5)
    legacy_store.close()

    exit_code = main(["stats", str(photo_root)])

    assert exit_code == 0
    assert not (photo_root / ".imgrate.sqlite").exists()
    assert RatingStore(photo_root / RatingStore.DEFAULT_FILE_NAME).load_ratings() == {"2023/a.jpg": 5}
