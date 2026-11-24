"""Layout of the sectioned thumbnail grid: section header rows, thumbnail rows, and pages that fit the screen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imgrate.session import Section


@dataclass(frozen=True, slots=True)
class GridRow:
    """Either a section header or one row of thumbnails ``[start_index, end_index)``."""

    section: Section
    start_index: int
    end_index: int
    is_header: bool


@dataclass(frozen=True, slots=True)
class GridPage:
    rows: list[GridRow]

    @property
    def start_index(self) -> int:
        return min(row.start_index for row in self.rows if not row.is_header)

    @property
    def end_index(self) -> int:
        return max(row.end_index for row in self.rows if not row.is_header)


def build_grid_rows(sections: list[Section], *, column_count: int) -> list[GridRow]:
    """Every section starts with a header and on a fresh thumbnail row."""
    if column_count < 1:
        raise ValueError(f"column_count must be >= 1, got {column_count}")
    rows: list[GridRow] = []
    for section in sections:
        rows.append(
            GridRow(section=section, start_index=section.start_index, end_index=section.start_index, is_header=True)
        )
        for row_start_index in range(section.start_index, section.end_index, column_count):
            rows.append(
                GridRow(
                    section=section,
                    start_index=row_start_index,
                    end_index=min(row_start_index + column_count, section.end_index),
                    is_header=False,
                )
            )
    return rows


def paginate_grid_rows(
    rows: list[GridRow], *, available_height: int, header_height: int, thumbnail_row_height: int
) -> list[GridPage]:
    """Greedy page fill. A header never ends a page; it moves to the next page together with its first row.

    A page always holds at least one thumbnail row, even if the screen is too small for it.
    """
    pages: list[GridPage] = []
    page_rows: list[GridRow] = []
    used_height = 0
    for row_index, row in enumerate(rows):
        row_height = header_height if row.is_header else thumbnail_row_height
        # A header needs room for itself and the thumbnail row following it.
        required_height = row_height + (thumbnail_row_height if row.is_header else 0)
        has_thumbnails = any(not page_row.is_header for page_row in page_rows)
        if has_thumbnails and used_height + required_height > available_height:
            pages.append(GridPage(rows=page_rows))
            page_rows = []
            used_height = 0
        page_rows.append(row)
        used_height += row_height
        if row_index == len(rows) - 1:
            pages.append(GridPage(rows=page_rows))
    return pages


def find_page_index(pages: list[GridPage], file_index: int) -> int:
    for page_index, page in enumerate(pages):
        if page.start_index <= file_index < page.end_index:
            return page_index
    return 0


def find_thumbnail_row_index(rows: list[GridRow], file_index: int) -> int | None:
    for row_index, row in enumerate(rows):
        if not row.is_header and row.start_index <= file_index < row.end_index:
            return row_index
    return None


def find_index_in_adjacent_row(rows: list[GridRow], *, file_index: int, direction: int) -> int:
    """Same column in the next / previous thumbnail row (skipping headers), clamped to that row's length."""
    row_index = find_thumbnail_row_index(rows, file_index)
    if row_index is None:
        return file_index
    column = file_index - rows[row_index].start_index
    target_row_index = row_index + direction
    while 0 <= target_row_index < len(rows) and rows[target_row_index].is_header:
        target_row_index += direction
    if not 0 <= target_row_index < len(rows):
        return file_index
    target_row = rows[target_row_index]
    return min(target_row.start_index + column, target_row.end_index - 1)
