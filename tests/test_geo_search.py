from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from lazyrate.geo_search import (
    DEFAULT_RADIUS_KM,
    describe_distance_band,
    find_place_match,
    parse_radius_query,
    rank_within_place,
    search_by_place,
    search_by_radius,
)
from lazyrate.locations import GeoCoordinates, LocationReader, PlaceIndex
from lazyrate.media import scan_media_files

if TYPE_CHECKING:
    from pathlib import Path

    from lazyrate.locations import PhotoLocation


@pytest.fixture
def geo_locations(geo_root: Path) -> dict[str, PhotoLocation]:
    collected_locations: dict[str, PhotoLocation] = {}
    LocationReader(root=geo_root, place_index=PlaceIndex()).read_locations(
        scan_media_files(geo_root).files,
        on_batch=lambda locations, _done_count: collected_locations.update(
            (location.relative_path, location) for location in locations
        ),
        should_stop=lambda: False,
    )
    return collected_locations


@pytest.mark.parametrize(
    ("query_text", "expected_latitude_deg", "expected_longitude_deg", "expected_radius_km"),
    [
        ("32.75,-16.95", 32.75, -16.95, DEFAULT_RADIUS_KM),
        ("@32.75 -16.95 100km", 32.75, -16.95, 100.0),
        ("near:32.75, -16.95 r:5km", 32.75, -16.95, 5.0),
        ("32.75,-16.95 +-100 km", 32.75, -16.95, 100.0),
        ("-33.9,18.4 ±2.5km", -33.9, 18.4, 2.5),
    ],
)
def test_radius_queries_are_parsed(
    query_text: str, expected_latitude_deg: float, expected_longitude_deg: float, expected_radius_km: float
) -> None:
    radius_query = parse_radius_query(query_text)

    assert radius_query is not None
    assert radius_query.center == GeoCoordinates(
        latitude_deg=expected_latitude_deg, longitude_deg=expected_longitude_deg
    )
    assert radius_query.radius_km == expected_radius_km


def test_text_is_not_a_radius_query_and_bad_coordinates_fail() -> None:
    assert parse_radius_query("beach sunset") is None
    assert parse_radius_query("madeira 2024") is None
    with pytest.raises(ValueError, match="out of range"):
        parse_radius_query("132.7,-16.9")


def test_distance_bands_end_at_the_radius() -> None:
    assert describe_distance_band(0.4, radius_km=100) == "< 1 km"
    assert describe_distance_band(7.0, radius_km=100) == "5-10 km"
    assert describe_distance_band(80.0, radius_km=100) == "50-100 km"
    assert describe_distance_band(3.0, radius_km=3) == "1-3 km"


def test_radius_search_finds_only_nearby_photos_nearest_first(geo_locations: dict[str, PhotoLocation]) -> None:
    radius_query = parse_radius_query("@32.6657,-16.9255 100km")
    assert radius_query is not None

    result = search_by_radius(query=radius_query, locations=geo_locations.values())

    ranked_paths = sorted(result.scores, key=lambda relative_path: -result.scores[relative_path])
    assert ranked_paths == ["madeira/funchal_1.jpg", "madeira/funchal_2.jpg", "madeira/north_coast.jpg"]
    assert result.section_labels["madeira/funchal_1.jpg"] == "< 1 km"
    assert result.section_labels["madeira/north_coast.jpg"] == "10-25 km"


def test_place_search_matches_region_and_groups_by_town(geo_locations: dict[str, PhotoLocation]) -> None:
    place_match = find_place_match(query_text="Madeira", locations=geo_locations)
    assert place_match is not None

    result = search_by_place(match=place_match, locations=geo_locations)

    assert set(result.scores) == {"madeira/funchal_1.jpg", "madeira/funchal_2.jpg", "madeira/north_coast.jpg"}
    assert result.section_labels["madeira/funchal_1.jpg"] == "Portugal › Madeira › Funchal"  # noqa: RUF001
    assert result.scores["madeira/funchal_1.jpg"] > result.scores["madeira/north_coast.jpg"]  # bigger group first


def test_place_match_splits_off_the_rest_of_the_query(geo_locations: dict[str, PhotoLocation]) -> None:
    place_match = find_place_match(query_text="white beach in portugal", locations=geo_locations)
    unknown_place_match = find_place_match(query_text="atlantis", locations=geo_locations)

    assert place_match is not None
    assert place_match.place_name == "portugal"
    assert place_match.remaining_text == "white beach in"
    assert len(place_match.matching_paths) == 4
    assert unknown_place_match is None


def test_text_ranking_within_a_place_drops_other_photos(geo_locations: dict[str, PhotoLocation]) -> None:
    place_match = find_place_match(query_text="white madeira", locations=geo_locations)
    assert place_match is not None
    text_scores = {relative_path: 0.1 for relative_path in geo_locations} | {"madeira/funchal_1.jpg": 0.9}

    result = rank_within_place(match=place_match, text_scores=text_scores)

    assert max(result.scores, key=lambda relative_path: result.scores[relative_path]) == "madeira/funchal_1.jpg"
    assert "munich/marienplatz.jpg" not in result.scores
    assert result.section_labels == {}
