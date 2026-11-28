from __future__ import annotations

from typing import TYPE_CHECKING

from conftest import encode_jpeg, save_fake_raw

from lazyrate.media import find_raw_companions, scan_media_files
from lazyrate.session import RatingSession
from lazyrate.store import RatingStore

if TYPE_CHECKING:
    from pathlib import Path


def build_raw_root(root: Path) -> Path:
    (root / "trip").mkdir()
    (root / "trip" / "DSC_0001.JPG").write_bytes(encode_jpeg((64, 48), (200, 0, 0)))
    save_fake_raw(root / "trip" / "DSC_0001.NEF")
    save_fake_raw(root / "trip" / "DSC_0002.NEF")  # RAW only
    return root


def open_session(root: Path) -> RatingSession:
    store = RatingStore(root / RatingStore.DEFAULT_FILE_NAME)
    return RatingSession(scan_result=scan_media_files(root), store=store)


def test_raw_twin_is_hidden_behind_its_jpeg_and_a_lone_raw_is_a_photo(tmp_path: Path) -> None:
    session = open_session(build_raw_root(tmp_path))

    visible_paths = [media_file.relative_path for media_file in session.visible_files]
    jpeg_file = next(media_file for media_file in session.visible_files if media_file.extension == ".jpg")

    assert sorted(visible_paths) == ["trip/DSC_0001.JPG", "trip/DSC_0002.NEF"]
    assert [companion.relative_path for companion in session.get_raw_companions(jpeg_file)] == ["trip/DSC_0001.NEF"]
    assert ".nef" in session.enabled_extensions


def test_rating_the_jpeg_rates_its_raw_and_exports_both(tmp_path: Path) -> None:
    root = build_raw_root(tmp_path)
    session = open_session(root)
    jpeg_file = next(media_file for media_file in session.visible_files if media_file.extension == ".jpg")

    session.rate(jpeg_file, rating=5)
    exported_paths = session.find_relative_paths_with_ratings(frozenset({5}))
    stored_ratings = RatingStore(root / RatingStore.DEFAULT_FILE_NAME).load_ratings()

    assert exported_paths == ["trip/DSC_0001.JPG", "trip/DSC_0001.NEF"]
    assert stored_ratings == {"trip/DSC_0001.JPG": 5, "trip/DSC_0001.NEF": 5}


def test_jpeg_takes_over_the_rating_its_raw_already_had(tmp_path: Path) -> None:
    root = build_raw_root(tmp_path)
    store = RatingStore(root / RatingStore.DEFAULT_FILE_NAME)
    store.set_rating(relative_path="trip/DSC_0001.NEF", rating=3)
    store.close()

    session = open_session(root)

    assert session.ratings["trip/DSC_0001.JPG"] == 3


def test_saved_format_selection_gets_raw_formats_once(tmp_path: Path) -> None:
    root = build_raw_root(tmp_path)
    store = RatingStore(root / RatingStore.DEFAULT_FILE_NAME)
    store.save_enabled_extensions({".jpg"})
    store.close()

    first_session = open_session(root)
    first_extensions = first_session.enabled_extensions
    first_session.set_enabled_extensions({".jpg"})  # switched off again on purpose
    second_session = open_session(root)

    assert ".nef" in first_extensions
    assert ".nef" not in second_session.enabled_extensions


def test_pairing_needs_both_formats_enabled(tmp_path: Path) -> None:
    scan_result = scan_media_files(build_raw_root(tmp_path))

    paired = find_raw_companions(scan_result.files, enabled_extensions={".jpg", ".nef"})
    raw_disabled = find_raw_companions(scan_result.files, enabled_extensions={".jpg"})

    assert list(paired) == ["trip/DSC_0001.JPG"]
    assert raw_disabled == {}
