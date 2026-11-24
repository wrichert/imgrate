from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from imgrate.media import scan_media_files
from imgrate.ratings import ALL_FILTER, REJECTED, UNRATED_FILTER, build_minimum_stars_filter
from imgrate.session import RatingSession

if TYPE_CHECKING:
    from pathlib import Path

    from imgrate.store import RatingStore


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
