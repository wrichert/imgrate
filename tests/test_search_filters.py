from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lazyrate.geo_search import SearchResult
from lazyrate.search_filters import (
    PhotoFilterFacts,
    build_filter_result,
    parse_search_filters,
    restrict_search_result,
    select_matching_paths,
)


def timestamp(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp()


PHOTOS = [
    PhotoFilterFacts("nikon_aug.nef", "NIKON Z 6_2", "NIKKOR Z 24-70mm f/4 S", timestamp("2024-08-15T10:00")),
    PhotoFilterFacts("nikon_dec.nef", "NIKON Z 6_2", "NIKKOR Z 50mm f/1.8 S", timestamp("2023-12-31T23:59")),
    PhotoFilterFacts("phone_aug.heic", "Apple iPhone 15 Pro", "back camera", timestamp("2024-08-01T08:00")),
    PhotoFilterFacts("scan.jpg", None, None, timestamp("2024-02-29T12:00")),
]


def matching(query: str) -> list[str]:
    return [facts.relative_path for facts in select_matching_paths(parse_search_filters(query), PHOTOS)]


def test_filters_are_split_from_the_rest_of_the_query() -> None:
    filters = parse_search_filters('beach madeira camera:nikon lens:"24-70" 2024-08')

    assert filters.camera_terms == ("nikon",)
    assert filters.lens_terms == ("24-70",)
    assert filters.date_range is not None
    assert filters.date_range.label == "2024-08"
    assert filters.remaining_text == "beach madeira"


def test_camera_lens_and_dates_select_photos() -> None:
    assert matching("camera:nikon") == ["nikon_aug.nef", "nikon_dec.nef"]
    assert matching("camera:phone") == ["phone_aug.heic"]
    assert matching("lens:50mm") == ["nikon_dec.nef"]
    assert matching("2024") == ["nikon_aug.nef", "phone_aug.heic", "scan.jpg"]
    assert matching("2024-08") == ["nikon_aug.nef", "phone_aug.heic"]
    assert matching("2024-02-29") == ["scan.jpg"]
    assert matching("2023-12..2024-02") == ["nikon_dec.nef", "scan.jpg"]
    assert matching("camera:nikon 2024") == ["nikon_aug.nef"]


def test_coordinates_and_plain_words_are_no_filters() -> None:
    filters = parse_search_filters("@32.75,-16.95 50km")

    assert not filters.is_active
    assert filters.remaining_text == "@32.75,-16.95 50km"


@pytest.mark.parametrize("query", ["2024-13", "2024-02-30", "2024..2023"])
def test_impossible_dates_are_rejected(query: str) -> None:
    with pytest.raises(ValueError, match=r"date|Date"):
        parse_search_filters(query)


def test_filters_alone_list_photos_by_month_oldest_first() -> None:
    filters = parse_search_filters("camera:nikon")

    result = build_filter_result(filters, select_matching_paths(filters, PHOTOS))

    assert sorted(result.scores, key=lambda path: -result.scores[path]) == ["nikon_dec.nef", "nikon_aug.nef"]
    assert result.section_labels == {"nikon_dec.nef": "2023-12", "nikon_aug.nef": "2024-08"}


def test_filters_narrow_another_search() -> None:
    place_result = SearchResult(
        scores={"nikon_aug.nef": 0.0, "phone_aug.heic": 0.0},
        section_labels={"nikon_aug.nef": "Funchal", "phone_aug.heic": "Funchal"},
        summary="2 photos taken in madeira",
    )

    narrowed_result = restrict_search_result(
        place_result, filters=parse_search_filters("camera:phone"), allowed_paths={"phone_aug.heic"}
    )

    assert narrowed_result.scores == {"phone_aug.heic": 0.0}
    assert narrowed_result.section_labels == {"phone_aug.heic": "Funchal"}
    assert "camera:phone" in narrowed_result.summary
