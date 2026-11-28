from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from lazyrate.cli import main
from lazyrate.embeddings import StoredEmbedding
from lazyrate.fingerprints import Fingerprint
from lazyrate.media import MediaFile, scan_media_files
from lazyrate.relocation import FileIdentity, FileMove, KnownFile, find_moved_files, find_vanished_paths
from lazyrate.session import RatingSession
from lazyrate.store import RatingStore

if TYPE_CHECKING:
    from pathlib import Path


def build_media_file(relative_path: str, *, size_bytes: int = 1000, modified_timestamp_utc: float = 1.5) -> MediaFile:
    return MediaFile(
        relative_path=relative_path,
        extension=relative_path.rsplit(".", 1)[-1].lower(),
        size_bytes=size_bytes,
        modified_timestamp_utc=modified_timestamp_utc,
    )


def build_known_file(relative_path: str, *, size_bytes: int = 1000, modified_timestamp_utc: float = 1.5) -> KnownFile:
    return KnownFile(
        relative_path=relative_path,
        identity=FileIdentity(size_bytes=size_bytes, modified_timestamp_utc=modified_timestamp_utc),
    )


def save_fingerprints_for_scan(store: RatingStore, root: Path) -> None:
    """What the fingerprint indexer stores for every photo (its size and modification time)."""
    store.save_fingerprints(
        Fingerprint(
            relative_path=media_file.relative_path,
            size_bytes=media_file.size_bytes,
            modified_timestamp_utc=media_file.modified_timestamp_utc,
            difference_hash=None,
            capture_timestamp=None,
        )
        for media_file in scan_media_files(root).files
    )


def test_renamed_file_is_matched_by_size_and_modification_time() -> None:
    known_files = [build_known_file("2023/a.jpg"), build_known_file("2023/b.jpg", size_bytes=2000)]
    current_files = [build_media_file("2023/b.jpg", size_bytes=2000), build_media_file("trip/renamed.JPG")]

    file_moves = find_moved_files(known_files=known_files, current_files=current_files)

    assert file_moves == [FileMove(old_relative_path="2023/a.jpg", new_relative_path="trip/renamed.JPG")]


def test_ambiguous_matches_are_not_moved() -> None:
    known_files = [build_known_file("a.jpg"), build_known_file("b.jpg")]
    current_files = [build_media_file("x/a.jpg"), build_media_file("x/b.jpg")]

    file_moves = find_moved_files(known_files=known_files, current_files=current_files)

    assert file_moves == []


@pytest.mark.parametrize(
    "current_file",
    [
        build_media_file("new.jpg", size_bytes=999),
        build_media_file("new.jpg", modified_timestamp_utc=2.5),
        build_media_file("new.png"),
    ],
)
def test_different_size_time_or_format_is_no_move(current_file: MediaFile) -> None:
    file_moves = find_moved_files(known_files=[build_known_file("old.jpg")], current_files=[current_file])

    assert file_moves == []


def test_rating_without_cached_identity_cannot_move() -> None:
    known_files = [KnownFile(relative_path="old.jpg", identity=None)]

    file_moves = find_moved_files(known_files=known_files, current_files=[build_media_file("new.jpg")])

    assert file_moves == []


def test_vanished_paths_are_known_paths_not_on_disk() -> None:
    known_files = [build_known_file("kept.jpg"), KnownFile(relative_path="gone.jpg", identity=None)]

    vanished_paths = find_vanished_paths(known_files=known_files, current_files=[build_media_file("kept.jpg")])

    assert vanished_paths == ["gone.jpg"]


def test_moving_records_rekeys_every_table(store: RatingStore) -> None:
    store.set_rating(relative_path="old.jpg", rating=4)
    store.save_fingerprints(
        [
            Fingerprint(
                relative_path="old.jpg",
                size_bytes=10,
                modified_timestamp_utc=1.0,
                difference_hash=7,
                capture_timestamp=None,
            )
        ]
    )
    store.save_embeddings(
        [
            StoredEmbedding(
                relative_path="old.jpg",
                size_bytes=10,
                modified_timestamp_utc=1.0,
                model_name="test-model",
                vector=np.ones(4, dtype=np.float32),
            )
        ]
    )

    store.move_file_records([FileMove(old_relative_path="old.jpg", new_relative_path="trip/new.jpg")])

    assert store.load_ratings() == {"trip/new.jpg": 4}
    assert list(store.load_fingerprints()) == ["trip/new.jpg"]
    assert list(store.load_embeddings("test-model")) == ["trip/new.jpg"]


def test_session_keeps_rating_of_photo_moved_between_runs(photo_root: Path, store: RatingStore) -> None:
    save_fingerprints_for_scan(store, photo_root)
    store.set_rating(relative_path="2023/a.jpg", rating=5)
    (photo_root / "madeira").mkdir()
    (photo_root / "2023" / "a.jpg").rename(photo_root / "madeira" / "sunset.jpg")

    session = RatingSession(scan_result=scan_media_files(photo_root), store=store)

    assert session.ratings["madeira/sunset.jpg"] == 5
    assert "2023/a.jpg" not in session.ratings
    assert [file_move.new_relative_path for file_move in session.recent_moves] == ["madeira/sunset.jpg"]


def test_rescan_keeps_rating_of_photo_moved_while_running(photo_root: Path, store: RatingStore) -> None:
    save_fingerprints_for_scan(store, photo_root)
    session = RatingSession(scan_result=scan_media_files(photo_root), store=store)
    session.rate(next(file for file in session.all_files if file.relative_path == "2024/d.jpg"), rating=3)
    (photo_root / "2024" / "d.jpg").rename(photo_root / "2024" / "renamed.jpg")

    session.replace_scan(scan_media_files(photo_root))

    assert session.ratings["2024/renamed.jpg"] == 3
    assert "2024/d.jpg" not in session.ratings


def test_prune_forgets_vanished_photos_but_keeps_moved_ones(photo_root: Path) -> None:
    store = RatingStore(photo_root / RatingStore.DEFAULT_FILE_NAME)
    save_fingerprints_for_scan(store, photo_root)
    store.set_rating(relative_path="2023/a.jpg", rating=5)
    store.set_rating(relative_path="2024/d.jpg", rating=0)
    store.close()
    (photo_root / "2023" / "a.jpg").rename(photo_root / "2023" / "moved.jpg")
    (photo_root / "2024" / "d.jpg").unlink()

    exit_code = main(["prune", str(photo_root)])

    reopened_store = RatingStore(photo_root / RatingStore.DEFAULT_FILE_NAME)
    assert exit_code == 0
    assert reopened_store.load_ratings() == {"2023/moved.jpg": 5}
    assert "2024/d.jpg" not in reopened_store.load_fingerprints()
    reopened_store.close()


def test_prune_dry_run_keeps_everything(photo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = RatingStore(photo_root / RatingStore.DEFAULT_FILE_NAME)
    store.set_rating(relative_path="gone.jpg", rating=2)
    store.close()

    exit_code = main(["prune", str(photo_root), "--dry-run"])

    reopened_store = RatingStore(photo_root / RatingStore.DEFAULT_FILE_NAME)
    assert exit_code == 0
    assert "gone.jpg" in capsys.readouterr().out
    assert reopened_store.load_ratings() == {"gone.jpg": 2}
    reopened_store.close()
