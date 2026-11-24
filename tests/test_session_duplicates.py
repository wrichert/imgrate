from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from imgrate.fingerprints import Fingerprint, compute_image_signature
from imgrate.media import scan_media_files
from imgrate.ratings import REJECTED, UNRATED_FILTER
from imgrate.session import RatingSession, SortOrder

if TYPE_CHECKING:
    from pathlib import Path

    from imgrate.store import RatingStore


def visible_paths(session: RatingSession) -> list[str]:
    return [media_file.relative_path for media_file in session.visible_files]


def index_all(session: RatingSession, store: RatingStore) -> None:
    """Same steps as the TUI's background indexer: compute, persist, register, regroup."""
    fingerprints = [
        Fingerprint(
            relative_path=media_file.relative_path,
            size_bytes=media_file.size_bytes,
            modified_timestamp_utc=media_file.modified_timestamp_utc,
            **_signature_fields(session.resolve_path(media_file)),
        )
        for media_file in session.find_files_missing_fingerprints()
    ]
    store.save_fingerprints(fingerprints)
    session.add_fingerprints(fingerprints)
    session.rebuild_duplicate_groups()


def _signature_fields(path: Path) -> dict[str, int | float | None]:
    signature = compute_image_signature(path)
    return {"difference_hash": signature.difference_hash, "capture_timestamp": signature.capture_timestamp}


@pytest.fixture
def burst_session(burst_root: Path, store: RatingStore) -> RatingSession:
    session = RatingSession(scan_result=scan_media_files(burst_root), store=store)
    index_all(session, store)
    return session


def test_duplicates_mode_shows_groups_in_time_order(burst_session: RatingSession) -> None:
    burst_session.set_duplicates_mode(is_duplicates_mode=True)

    assert burst_session.duplicate_group_count == 2
    assert visible_paths(burst_session) == [
        "burst/sunrise_1.jpg",
        "burst/sunrise_2.jpg",
        "burst/sunrise_3.jpg",
        "burst/forest_1.jpg",
        "burst/forest_2.jpg",
    ]
    section_position = burst_session.current_section_position
    assert section_position is not None
    assert (section_position.section_number, section_position.section_count, section_position.section.size) == (
        1,
        2,
        3,
    )
    assert section_position.section.label == "group 1 · 2024-06-01 06:00"


def test_keep_best_rejects_rest_of_group_and_undo_restores_all(burst_session: RatingSession) -> None:
    burst_session.set_duplicates_mode(is_duplicates_mode=True)
    burst_session.move_by(1)  # sunrise_2 is the keeper
    burst_session.rate_current(5)

    changes = burst_session.reject_others_in_group()

    assert {change.relative_path for change in changes} == {"burst/sunrise_1.jpg", "burst/sunrise_3.jpg"}
    assert [burst_session.get_rating(media_file) for media_file in burst_session.visible_files[:3]] == [
        REJECTED,
        5,
        REJECTED,
    ]
    burst_session.undo()
    assert [burst_session.get_rating(media_file) for media_file in burst_session.visible_files[:3]] == [None, 5, None]


def test_duplicates_mode_hides_groups_without_matching_member(burst_session: RatingSession) -> None:
    burst_session.set_duplicates_mode(is_duplicates_mode=True)
    burst_session.reject_others_in_group()
    burst_session.rate_current(4)

    burst_session.set_rating_filter(UNRATED_FILTER)

    assert visible_paths(burst_session) == ["burst/forest_1.jpg", "burst/forest_2.jpg"]


def test_move_by_section_between_groups(burst_session: RatingSession) -> None:
    burst_session.set_duplicates_mode(is_duplicates_mode=True)

    burst_session.move_by_section(1)
    assert burst_session.current_index == 3
    burst_session.move_by_section(5)
    assert burst_session.current_index == 3
    burst_session.move_by(1)
    burst_session.move_by_section(-1)  # from inside a section: back to its start
    assert burst_session.current_index == 3
    burst_session.move_by_section(-1)
    assert burst_session.current_index == 0


def test_rating_sort_has_one_section_per_rating(burst_session: RatingSession) -> None:
    for media_file, rating in zip(burst_session.visible_files[:3], (5, 5, REJECTED), strict=True):
        burst_session.rate(media_file, rating=rating)

    burst_session.set_sort_order(SortOrder.RATING)

    assert [(section.label, section.size) for section in burst_session.sections] == [
        ("★★★★★", 2),
        ("unrated", 4),
        ("rejected", 1),
    ]


def test_date_sort_has_month_sections_and_path_sort_has_folder_sections(burst_session: RatingSession) -> None:
    assert [(section.label, section.size) for section in burst_session.sections] == [("burst", 6), ("later", 1)]

    burst_session.set_sort_order(SortOrder.DATE)

    assert [(section.label, section.size) for section in burst_session.sections] == [("June 2024", 7)]


def test_sort_by_rating_puts_best_first_and_rejected_last(burst_session: RatingSession) -> None:
    ratings_by_path = {"burst/beach.jpg": REJECTED, "burst/forest_2.jpg": 5, "later/sunrise_next_day.jpg": 3}
    for media_file in burst_session.visible_files:
        if media_file.relative_path in ratings_by_path:
            burst_session.rate(media_file, rating=ratings_by_path[media_file.relative_path])

    burst_session.set_sort_order(SortOrder.RATING)

    paths = visible_paths(burst_session)
    assert paths[:2] == ["burst/forest_2.jpg", "later/sunrise_next_day.jpg"]
    assert paths[-1] == "burst/beach.jpg"


def test_sort_by_date_uses_capture_time(burst_session: RatingSession) -> None:
    burst_session.set_sort_order(SortOrder.DATE)

    assert visible_paths(burst_session) == [
        "burst/sunrise_1.jpg",
        "burst/sunrise_2.jpg",
        "burst/sunrise_3.jpg",
        "burst/forest_1.jpg",
        "burst/forest_2.jpg",
        "burst/beach.jpg",
        "later/sunrise_next_day.jpg",
    ]


def test_sort_change_keeps_cursor_on_same_file(burst_session: RatingSession) -> None:
    burst_session.move_to(2)
    current_path = visible_paths(burst_session)[2]

    burst_session.set_sort_order(SortOrder.DATE)

    assert burst_session.current_file is not None
    assert burst_session.current_file.relative_path == current_path


def test_regrouping_does_not_reshuffle_normal_view(burst_root: Path, store: RatingStore) -> None:
    session = RatingSession(scan_result=scan_media_files(burst_root), store=store)
    session.set_rating_filter(UNRATED_FILTER)
    session.rate_current(5)
    paths_before = visible_paths(session)

    index_all(session, store)

    assert visible_paths(session) == paths_before


def test_stale_fingerprints_are_ignored(burst_root: Path, store: RatingStore, burst_session: RatingSession) -> None:
    store.save_fingerprints(
        [
            Fingerprint(
                relative_path="burst/beach.jpg",
                size_bytes=1,
                modified_timestamp_utc=0.0,
                difference_hash=0,
                capture_timestamp=None,
            )
        ]
    )

    reopened_session = RatingSession(scan_result=scan_media_files(burst_root), store=store)

    assert [media_file.relative_path for media_file in reopened_session.find_files_missing_fingerprints()] == [
        "burst/beach.jpg"
    ]
