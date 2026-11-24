from __future__ import annotations

import pytest

from imgrate.grid_layout import (
    build_grid_rows,
    find_index_in_adjacent_row,
    find_page_index,
    paginate_grid_rows,
)
from imgrate.session import Section

SECTIONS = [
    Section(label="★★★★★", start_index=0, end_index=5),
    Section(label="★★★★", start_index=5, end_index=6),
    Section(label="unrated", start_index=6, end_index=13),
]


def describe_rows(column_count: int) -> list[str]:
    return [
        f"H:{row.section.label}" if row.is_header else f"{row.start_index}-{row.end_index}"
        for row in build_grid_rows(SECTIONS, column_count=column_count)
    ]


def test_each_section_gets_header_and_starts_on_new_row() -> None:
    assert describe_rows(3) == ["H:★★★★★", "0-3", "3-5", "H:★★★★", "5-6", "H:unrated", "6-9", "9-12", "12-13"]


def test_pages_fit_height_and_never_end_with_header() -> None:
    rows = build_grid_rows(SECTIONS, column_count=3)

    pages = paginate_grid_rows(rows, available_height=30, header_height=1, thumbnail_row_height=13)

    assert [[row.is_header for row in page.rows] for page in pages] == [
        [True, False, False],
        [True, False, True, False],
        [False, False],
    ]
    assert find_page_index(pages, 6) == 1
    assert find_page_index(pages, 12) == 2


def test_tiny_screen_still_shows_one_thumbnail_row_per_page() -> None:
    rows = build_grid_rows(SECTIONS[:1], column_count=5)

    pages = paginate_grid_rows(rows, available_height=3, header_height=1, thumbnail_row_height=13)

    assert len(pages) == 1
    assert (pages[0].start_index, pages[0].end_index) == (0, 5)


@pytest.mark.parametrize(
    ("file_index", "direction", "expected_index"),
    [
        (1, 1, 4),  # 0-3 -> 3-5, same column
        (2, 1, 4),  # clamped to shorter row
        (4, 1, 5),  # skips header into next section
        (5, -1, 3),
        (12, 1, 12),  # last row stays
    ],
)
def test_vertical_movement_keeps_column(file_index: int, direction: int, expected_index: int) -> None:
    rows = build_grid_rows(SECTIONS, column_count=3)

    assert find_index_in_adjacent_row(rows, file_index=file_index, direction=direction) == expected_index


def test_invalid_column_count() -> None:
    with pytest.raises(ValueError, match="column_count"):
        build_grid_rows(SECTIONS, column_count=0)
