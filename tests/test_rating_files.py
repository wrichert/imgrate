from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from conftest import encode_jpeg, save_fake_raw

from lazyrate.locations import LocationReader, PlaceIndex
from lazyrate.media import scan_media_files
from lazyrate.session import RatingSession
from lazyrate.store import RatingStore
from lazyrate.xmp import RatingFileMode, read_file_rating, write_rating_file

if TYPE_CHECKING:
    from pathlib import Path

    from lazyrate.locations import PhotoLocation


def open_session(root: Path) -> RatingSession:
    return RatingSession(scan_result=scan_media_files(root), store=RatingStore(root / RatingStore.DEFAULT_FILE_NAME))


def read_metadata(session: RatingSession) -> int:
    """The metadata pass as the app runs it: save each batch, then hand it to the session."""
    store = RatingStore(session.root / RatingStore.DEFAULT_FILE_NAME)
    imported_counts: list[int] = []
    LocationReader(root=session.root, place_index=PlaceIndex()).read_locations(
        session.find_files_missing_locations(),
        on_batch=functools.partial(save_and_register_batch, store, session, imported_counts),
        should_stop=lambda: False,
    )
    store.close()
    return sum(imported_counts)


def save_and_register_batch(
    store: RatingStore,
    session: RatingSession,
    imported_counts: list[int],
    locations: list[PhotoLocation],
    _done_count: int,
) -> None:
    store.save_locations(locations)
    imported_counts.append(session.add_locations(locations))


def test_switching_on_sidecars_writes_existing_ratings_and_every_new_one(tmp_path: Path) -> None:
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(encode_jpeg((16, 16), (0, 0, 0)))
    session = open_session(tmp_path)
    session.rate(session.visible_files[0], rating=4)

    written_count = session.set_rating_file_mode(RatingFileMode.SIDECAR)
    session.rate(session.visible_files[1], rating=0)
    session.wait_for_rating_files()

    assert written_count == 1
    assert read_file_rating(tmp_path / "a.jpg") == 4
    assert read_file_rating(tmp_path / "b.jpg") == 0
    assert open_session(tmp_path).rating_file_mode is RatingFileMode.SIDECAR


def test_ratings_stored_in_files_are_imported_for_unrated_photos_only(tmp_path: Path) -> None:
    for name in ("camera_rated.jpg", "rated_here.jpg"):
        (tmp_path / name).write_bytes(encode_jpeg((16, 16), (0, 0, 0)))
        write_rating_file(tmp_path / name, rating=5, mode=RatingFileMode.SIDECAR)
    session = open_session(tmp_path)
    rated_here = next(
        media_file for media_file in session.visible_files if media_file.relative_path == "rated_here.jpg"
    )
    session.rate(rated_here, rating=2)

    imported_count = read_metadata(session)

    assert imported_count == 1
    assert session.ratings == {"camera_rated.jpg": 5, "rated_here.jpg": 2}


def test_raw_rating_from_the_camera_reaches_its_jpeg_twin(tmp_path: Path) -> None:
    (tmp_path / "DSC_0001.JPG").write_bytes(encode_jpeg((16, 16), (0, 0, 0)))
    save_fake_raw(tmp_path / "DSC_0001.NEF")
    write_rating_file(tmp_path / "DSC_0001.NEF", rating=3, mode=RatingFileMode.SIDECAR)
    session = open_session(tmp_path)

    read_metadata(session)

    assert session.ratings["DSC_0001.JPG"] == 3


def test_embedded_rating_keeps_the_cached_metadata_valid(tmp_path: Path) -> None:
    photo_path = tmp_path / "photo.jpg"
    photo_path.write_bytes(encode_jpeg((16, 16), (0, 0, 0)))
    session = open_session(tmp_path)
    read_metadata(session)
    session.set_rating_file_mode(RatingFileMode.EMBEDDED)

    session.rate(session.visible_files[0], rating=5)
    session.wait_for_rating_files()
    reopened_session = open_session(tmp_path)

    assert read_file_rating(photo_path) == 5
    assert reopened_session.find_files_missing_locations() == []
