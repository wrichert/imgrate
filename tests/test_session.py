from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from lazyrate.locations import LocationReader, PlaceIndex
from lazyrate.media import scan_media_files
from lazyrate.ratings import ALL_FILTER, REJECTED, UNRATED_FILTER, build_minimum_stars_filter
from lazyrate.session import RatingSession, SortOrder

if TYPE_CHECKING:
    from pathlib import Path

    from lazyrate.locations import PhotoLocation
    from lazyrate.store import RatingStore


def visible_paths(session: RatingSession) -> list[str]:
    return [media_file.relative_path for media_file in session.visible_files]


def test_videos_are_disabled_by_default(session: RatingSession) -> None:
    assert ".mp4" not in session.enabled_extensions
    assert "2024/clip.mp4" not in visible_paths(session)
    assert session.visible_count == 5


def test_enabling_extension_is_persisted(session: RatingSession, photo_root: Path, store: RatingStore) -> None:
    session.set_enabled_extensions({".jpg", ".mp4"})

    reopened_session = RatingSession(scan_result=scan_media_files(photo_root), store=store)

    assert reopened_session.enabled_extensions == {".jpg", ".mp4"}
    assert visible_paths(reopened_session) == ["2023/a.jpg", "2023/b.JPG", "2024/clip.mp4", "2024/d.jpg"]


def test_rating_is_persisted(session: RatingSession, photo_root: Path, store: RatingStore) -> None:
    session.rate_current(5)

    reopened_session = RatingSession(scan_result=scan_media_files(photo_root), store=store)

    assert reopened_session.current_rating == 5
    assert store.load_ratings() == {"2023/a.jpg": 5}


def test_visible_list_is_stable_while_rating_in_filtered_view(session: RatingSession) -> None:
    session.set_rating_filter(UNRATED_FILTER)

    session.rate_current(3)
    session.move_by(1)
    session.rate_current(REJECTED)

    assert session.visible_count == 5
    assert session.current_index == 1


def test_minimum_stars_filter(session: RatingSession) -> None:
    for rating in (1, 4, 5):
        session.rate_current(rating)
        session.move_by(1)

    session.set_rating_filter(build_minimum_stars_filter(4))

    assert visible_paths(session) == ["2023/b.JPG", "2023/sub/c.png"]


def test_filter_change_keeps_cursor_on_next_file_in_order(session: RatingSession) -> None:
    session.move_to(2)  # 2023/sub/c.png
    session.rate_current(5)
    session.set_rating_filter(UNRATED_FILTER)

    assert session.current_file is not None
    assert session.current_file.relative_path == "2024/d.jpg"


def test_undo_restores_previous_rating_and_cursor(session: RatingSession) -> None:
    session.rate_current(2)
    session.rate_current(4)
    session.move_by(3)

    change = session.undo()

    assert change is not None
    assert session.current_index == 0
    assert session.current_rating == 2


def test_clear_rating_removes_row(session: RatingSession, store: RatingStore) -> None:
    session.rate_current(4)
    session.rate_current(None)

    assert store.load_ratings() == {}
    assert session.count_ratings() == {None: 5}


def test_move_to_next_unrated_wraps(session: RatingSession) -> None:
    session.rate_current(3)
    session.move_to(4)
    session.rate_current(3)

    is_found = session.move_to_next_unrated()

    assert is_found
    assert session.current_index == 1


def test_movement_is_clamped(session: RatingSession) -> None:
    session.move_by(-10)
    assert session.current_index == 0
    session.move_by(100)
    assert session.current_index == 4


def test_empty_view_has_no_current_file(session: RatingSession) -> None:
    session.set_rating_filter(build_minimum_stars_filter(5))

    assert session.current_file is None
    assert session.rate_current(5) is None
    session.set_rating_filter(ALL_FILTER)
    assert session.visible_count == 5


def test_invalid_rating_is_rejected(store: RatingStore) -> None:
    with pytest.raises(ValueError, match="Rating must be"):
        store.set_rating(relative_path="a.jpg", rating=7)


def test_rating_moves_counts_between_rating_sections_until_refresh(session: RatingSession) -> None:
    session.set_sort_order(SortOrder.RATING)
    session.rate(session.visible_files[0], rating=REJECTED)
    session.refresh_view()
    unrated_section, rejected_section = session.sections

    session.rate(session.visible_files[0], rating=REJECTED)
    unrated_counts = session.count_section_members(unrated_section)
    rejected_counts = session.count_section_members(rejected_section)
    session.refresh_view()
    counts_after_refresh = [session.count_section_members(section) for section in session.sections]

    assert (unrated_counts.member_count, unrated_counts.moved_out_count) == (unrated_section.size - 1, 1)
    assert (rejected_counts.member_count, rejected_counts.moved_in_count) == (rejected_section.size + 1, 1)
    assert not any(counts.is_pending_resort for counts in counts_after_refresh)


def test_leaving_the_rating_filter_counts_as_moved_out(session: RatingSession) -> None:
    session.set_rating_filter(UNRATED_FILTER)
    folder_section = session.sections[0]

    session.rate(session.visible_files[0], rating=REJECTED)
    folder_counts = session.count_section_members(folder_section)

    assert (folder_counts.member_count, folder_counts.moved_out_count) == (folder_section.size - 1, 1)


def test_move_to_first_unrated_skips_rated_files(session: RatingSession) -> None:
    session.rate(session.visible_files[0], rating=5)
    session.rate(session.visible_files[1], rating=REJECTED)
    session.move_to(3)

    session.move_to_first_unrated()

    assert session.current_index == 2


def test_move_to_first_unrated_falls_back_to_first_file(session: RatingSession) -> None:
    for media_file in session.visible_files:
        session.rate(media_file, rating=3)
    session.move_to(2)

    session.move_to_first_unrated()

    assert session.current_index == 0


def build_located_session(geo_root: Path, store: RatingStore) -> RatingSession:
    collected_locations: list[PhotoLocation] = []
    scan_result = scan_media_files(geo_root)
    LocationReader(root=geo_root, place_index=PlaceIndex()).read_locations(
        scan_result.files,
        on_batch=lambda locations, _done_count: collected_locations.extend(locations),
        should_stop=lambda: False,
    )
    store.save_locations(collected_locations)
    return RatingSession(scan_result=scan_result, store=store)


def test_place_sort_groups_by_country_region_town_with_unlocated_last(geo_root: Path, store: RatingStore) -> None:
    session = build_located_session(geo_root, store)

    session.set_sort_order(SortOrder.PLACE)

    assert SortOrder.PLACE in session.available_sort_orders
    assert [section.label for section in session.sections] == [
        "Germany › Bavaria › Munich",  # noqa: RUF001
        "Portugal › Lisbon › Belém",  # noqa: RUF001
        "Portugal › Madeira › Funchal",  # noqa: RUF001
        "Portugal › Madeira › São Jorge",  # noqa: RUF001
        "no location",
    ]


def test_changed_files_lose_their_stored_location(geo_root: Path, store: RatingStore) -> None:
    build_located_session(geo_root, store)
    os.utime(geo_root / "munich" / "marienplatz.jpg", (1_000_000, 1_000_000))

    session = RatingSession(scan_result=scan_media_files(geo_root), store=store)

    assert [media_file.relative_path for media_file in session.find_files_missing_locations()] == [
        "munich/marienplatz.jpg"
    ]


def test_search_with_section_labels_lists_unmatched_photos_last(session: RatingSession) -> None:
    session.set_search(
        query="madeira",
        scores={"2024/d.jpg": 0.0, "2023/a.jpg": -1.0},
        section_labels={"2024/d.jpg": "Funchal", "2023/a.jpg": "Santana"},
    )

    assert [section.label for section in session.sections] == ["Funchal", "Santana", "other photos"]
    assert session.visible_files[0].relative_path == "2024/d.jpg"
