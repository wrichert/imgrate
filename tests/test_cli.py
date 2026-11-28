from __future__ import annotations

from pathlib import Path

import pytest
from conftest import save_fake_raw, save_geotagged_photo

from lazyrate.cli import main, resolve_search_target
from lazyrate.export import build_unique_flat_name
from lazyrate.geo_map import BROWSER_MAP_FILE_NAME
from lazyrate.store import RatingStore


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


def test_without_a_folder_the_current_directory_is_used(
    geo_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(geo_root)

    stats_exit_code = main(["stats"])
    capsys.readouterr()
    search_exit_code = main(["search", "funchal"])  # "madeira" would be its sub folder
    search_output = capsys.readouterr().out

    assert stats_exit_code == 0
    assert (geo_root / RatingStore.DEFAULT_FILE_NAME).is_file()
    assert search_exit_code == 0
    assert search_output.count("madeira/funchal_") == 2


def test_plain_lazyrate_rates_the_current_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    opened_roots: list[Path] = []
    monkeypatch.setattr(
        "lazyrate.cli.run_tui",
        lambda *, root, database_path, is_ai_enabled: opened_roots.append(root) or 0,
    )

    exit_code = main([])

    assert exit_code == 0
    assert opened_roots == [tmp_path.resolve()]


@pytest.mark.parametrize(
    ("first_argument", "query", "expected_root", "expected_query"),
    [
        (None, None, Path(), None),
        ("madeira", None, Path(), "madeira"),
        ("beach", "madeira", Path(), "beach madeira"),
        ("~/Pictures/2025", "madeira", Path("~/Pictures/2025"), "madeira"),
        ("./trip", None, Path("./trip"), None),
    ],
)
def test_search_arguments_without_a_folder_are_the_query(
    first_argument: str | None, query: str | None, expected_root: Path, expected_query: str | None
) -> None:
    search_target = resolve_search_target(first_argument=first_argument, query=query)

    assert search_target.root == expected_root
    assert search_target.query == expected_query


def test_legacy_imgrate_database_is_renamed_and_keeps_ratings(photo_root: Path) -> None:
    legacy_store = RatingStore(photo_root / ".imgrate.sqlite")
    legacy_store.set_rating(relative_path="2023/a.jpg", rating=5)
    legacy_store.close()

    exit_code = main(["stats", str(photo_root)])

    assert exit_code == 0
    assert not (photo_root / ".imgrate.sqlite").exists()
    assert RatingStore(photo_root / RatingStore.DEFAULT_FILE_NAME).load_ratings() == {"2023/a.jpg": 5}


def test_search_by_place_works_without_ai(geo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["search", str(geo_root), "madeira"])

    output_lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert len(output_lines) == 3
    assert output_lines[0].startswith("Portugal › Madeira › Funchal  ")  # noqa: RUF001
    assert all("/madeira/" in line for line in output_lines)


def test_search_near_coordinates_prints_distances(geo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["search", str(geo_root), "--near=48.14,11.58", "--radius", "5"])

    output_lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert len(output_lines) == 1
    assert output_lines[0].strip().startswith("0.")
    assert output_lines[0].endswith("munich/marienplatz.jpg")


def test_stats_reports_gps_once_locations_were_read(geo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["search", str(geo_root), "lisbon"])
    capsys.readouterr()

    main(["stats", str(geo_root)])

    assert "GPS: 5 photos in 4 towns (Portugal 4, Germany 1)" in capsys.readouterr().out


def test_map_writes_a_page_with_all_photos_with_gps(geo_root: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "map.html"

    exit_code = main(["map", str(geo_root), "--output", str(output_path), "--no-open"])

    assert exit_code == 0
    assert output_path.read_text().count('"path": ') == 5


def test_map_of_a_search_contains_only_the_matches(geo_root: Path) -> None:
    exit_code = main(["map", str(geo_root), "lisbon", "--no-open"])

    page = (geo_root / BROWSER_MAP_FILE_NAME).read_text()
    assert exit_code == 0
    assert page.count('"path": ') == 1
    assert "lisbon/belem.jpg" in page


def test_search_by_camera_and_date_narrows_a_place_search(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    save_fake_raw(tmp_path / "DSC_0001.NEF", latitude_deg=32.6657, longitude_deg=-16.9255)
    save_geotagged_photo(tmp_path / "phone.jpg", latitude_deg=32.6657, longitude_deg=-16.9255, color=(1, 2, 3))

    nikon_exit_code = main(["search", str(tmp_path), "funchal camera:nikon"])
    nikon_output = capsys.readouterr().out
    date_exit_code = main(["search", str(tmp_path), "2024-08"])
    date_output = capsys.readouterr().out

    assert nikon_exit_code == 0
    assert "DSC_0001.NEF" in nikon_output
    assert "phone.jpg" not in nikon_output
    assert date_exit_code == 0
    assert "DSC_0001.NEF" in date_output  # taken 2024-08-15 (the fake RAW's EXIF date)
