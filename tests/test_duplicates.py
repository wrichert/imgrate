from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from imgrate.duplicates import DuplicateFinder, HashedFile
from imgrate.fingerprints import Fingerprint, compute_image_signature, count_differing_bits

if TYPE_CHECKING:
    from pathlib import Path

    from imgrate.store import RatingStore


def test_near_duplicates_have_close_hashes_and_different_images_do_not(burst_root: Path) -> None:
    sunrise_hash = compute_image_signature(burst_root / "burst/sunrise_1.jpg").difference_hash
    resized_hash = compute_image_signature(burst_root / "burst/sunrise_2.jpg").difference_hash
    brighter_hash = compute_image_signature(burst_root / "burst/sunrise_3.jpg").difference_hash
    forest_hash = compute_image_signature(burst_root / "burst/forest_1.jpg").difference_hash
    assert None not in (sunrise_hash, resized_hash, brighter_hash, forest_hash)

    assert count_differing_bits(sunrise_hash, resized_hash) <= DuplicateFinder.DEFAULT_MAX_DIFFERING_BITS
    assert count_differing_bits(sunrise_hash, brighter_hash) <= DuplicateFinder.DEFAULT_MAX_DIFFERING_BITS
    assert count_differing_bits(sunrise_hash, forest_hash) > DuplicateFinder.DEFAULT_MAX_DIFFERING_BITS


def test_signature_reads_exif_capture_time(burst_root: Path) -> None:
    signature = compute_image_signature(burst_root / "burst/beach.jpg")

    assert signature.capture_timestamp == pytest.approx(1717243200.0)  # 2024-06-01 12:00:00 as UTC


def test_undecodable_file_has_no_hash(tmp_path: Path) -> None:
    broken_path = tmp_path / "broken.jpg"
    broken_path.write_bytes(b"not a jpeg")

    assert compute_image_signature(broken_path).difference_hash is None


def test_finder_groups_within_time_window_only() -> None:
    hashed_files = [
        HashedFile(relative_path="a1", difference_hash=0b1111, sort_timestamp=0.0),
        HashedFile(relative_path="a2", difference_hash=0b1110, sort_timestamp=5.0),
        HashedFile(relative_path="a3", difference_hash=0b1100, sort_timestamp=9.0),
        HashedFile(relative_path="other", difference_hash=(1 << 64) - 1, sort_timestamp=6.0),
        HashedFile(relative_path="a_next_day", difference_hash=0b1111, sort_timestamp=86_400.0),
    ]
    finder = DuplicateFinder(max_differing_bits=1, max_time_gap_seconds=60)

    groups = finder.find_groups(hashed_files)

    assert [group.relative_paths for group in groups] == [["a1", "a2", "a3"]]


def test_finder_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="max_differing_bits"):
        DuplicateFinder(max_differing_bits=-1)


def test_store_round_trips_unsigned_64_bit_hash(store: RatingStore) -> None:
    fingerprint = Fingerprint(
        relative_path="x.jpg",
        size_bytes=10,
        modified_timestamp_utc=1.5,
        difference_hash=(1 << 64) - 1,
        capture_timestamp=None,
    )

    store.save_fingerprints([fingerprint])

    assert store.load_fingerprints() == {"x.jpg": fingerprint}
