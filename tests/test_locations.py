from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from lazyrate.locations import (
    GeoCoordinates,
    LocationReader,
    PlaceIndex,
    compute_distances_km,
    normalize_place_name,
    read_gps_coordinates,
)
from lazyrate.media import scan_media_files

if TYPE_CHECKING:
    from pathlib import Path

    from lazyrate.locations import PhotoLocation
    from lazyrate.store import RatingStore


def read_all_locations(root: Path) -> list[PhotoLocation]:
    collected_locations: list[PhotoLocation] = []
    LocationReader(root=root, place_index=PlaceIndex()).read_locations(
        scan_media_files(root).files,
        on_batch=lambda locations, _done_count: collected_locations.extend(locations),
        should_stop=lambda: False,
    )
    return collected_locations


def test_gps_coordinates_are_read_with_hemisphere_signs(geo_root: Path) -> None:
    coordinates = read_gps_coordinates(geo_root / "madeira" / "funchal_1.jpg")

    assert coordinates is not None
    assert coordinates.latitude_deg == pytest.approx(32.6657, abs=1e-6)
    assert coordinates.longitude_deg == pytest.approx(-16.9255, abs=1e-6)


def test_missing_and_null_island_gps_mean_no_location(geo_root: Path) -> None:
    missing_coordinates = read_gps_coordinates(geo_root / "misc" / "no_gps.jpg")
    null_island_coordinates = read_gps_coordinates(geo_root / "misc" / "null_island.jpg")

    assert missing_coordinates is None
    assert null_island_coordinates is None


def test_nearest_town_prefers_the_city_over_its_suburbs() -> None:
    photo_positions = [
        GeoCoordinates(latitude_deg=32.65, longitude_deg=-16.91),  # Funchal, next to the parish Santa Luzia
        GeoCoordinates(latitude_deg=40.758, longitude_deg=-73.9855),  # Times Square, across from Weehawken
        GeoCoordinates(latitude_deg=40.7357, longitude_deg=-74.1724),  # Newark, not swallowed by New York
    ]

    places = PlaceIndex().find_nearest_places(photo_positions)

    assert [place.town for place in places] == ["Funchal", "New York City", "Newark"]
    assert places[0].path_label == "Portugal › Madeira › Funchal"  # noqa: RUF001


def test_reader_names_places_and_remembers_photos_without_gps(geo_root: Path) -> None:
    locations = {location.relative_path: location for location in read_all_locations(geo_root)}

    assert locations["madeira/funchal_2.jpg"].place is not None
    assert locations["madeira/funchal_2.jpg"].place.region == "Madeira"
    assert locations["munich/marienplatz.jpg"].place is not None
    assert locations["munich/marienplatz.jpg"].place.town == "Munich"
    assert locations["misc/no_gps.jpg"].coordinates is None
    assert len(locations) == 7


def test_locations_round_trip_through_the_store(geo_root: Path, store: RatingStore) -> None:
    locations = read_all_locations(geo_root)

    store.save_locations(locations)

    assert store.load_locations() == {location.relative_path: location for location in locations}


def test_distances_are_great_circle_kilometres() -> None:
    lisbon = GeoCoordinates(latitude_deg=38.7223, longitude_deg=-9.1393)

    distances_km = compute_distances_km(
        origin=lisbon, latitudes_deg=np.array([38.7223, 40.4168]), longitudes_deg=np.array([-9.1393, -3.7038])
    )

    assert distances_km[0] == pytest.approx(0.0, abs=1e-6)
    assert distances_km[1] == pytest.approx(503, abs=3)  # Lisbon - Madrid


def test_place_names_match_without_case_and_accents() -> None:
    assert normalize_place_name("São Vicente") == normalize_place_name("sao  VICENTE")
    assert normalize_place_name("Baden-Württemberg") == "baden wurttemberg"


def test_impossible_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        GeoCoordinates(latitude_deg=91.0, longitude_deg=0.0)
