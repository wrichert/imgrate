"""Search by where photos were taken: ``@32.75,-16.95 100km`` (radius) or ``madeira`` (town, region or country)."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lazyrate.locations import GeoCoordinates, compute_distances_km, normalize_place_name

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lazyrate.locations import PhotoLocation

DEFAULT_RADIUS_KM = 10.0
DISTANCE_BAND_LIMITS_KM = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?"
RADIUS_QUERY_PATTERN = re.compile(
    rf"""^\s*(?:near:|@)?\s*
    (?P<latitude>{NUMBER_PATTERN})\s*[,\s]\s*(?P<longitude>{NUMBER_PATTERN})
    (?:\s*(?:r:|±|\+-|\+/-)?\s*(?P<radius>\d+(?:\.\d+)?)\s*km)?\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class RadiusQuery:
    center: GeoCoordinates
    radius_km: float

    def __str__(self) -> str:
        return f"within {self.radius_km:g} km of {self.center}"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Ranking for the session: higher score first; files without a score are listed last as "other photos".

    ``section_labels`` names the section of each scored file; empty means "label by rank" (text search).
    """

    scores: dict[str, float]
    section_labels: dict[str, str]
    summary: str

    @property
    def match_count(self) -> int:
        return len(self.scores)


@dataclass(frozen=True, slots=True)
class PlaceMatch:
    """A place name found in the query, the photos taken there, and the rest of the query (e.g. ``beach``)."""

    place_name: str
    matching_paths: list[str]
    remaining_text: str


def parse_radius_query(text: str) -> RadiusQuery | None:
    """``32.75,-16.95``, ``@32.75 -16.95 100km``, ``near:32.75,-16.95 r:5km``, ``32.75,-16.95 +-100km``.

    Returns ``None`` if ``text`` is not a coordinate query; raises ``ValueError`` for impossible coordinates.
    """
    match = RADIUS_QUERY_PATTERN.match(text)
    if match is None:
        return None
    center = GeoCoordinates(latitude_deg=float(match["latitude"]), longitude_deg=float(match["longitude"]))
    radius_km = float(match["radius"]) if match["radius"] else DEFAULT_RADIUS_KM
    if radius_km <= 0:
        raise ValueError(f"Radius must be positive, got {radius_km:g} km")
    return RadiusQuery(center=center, radius_km=radius_km)


def describe_distance_band(distance_km: float, *, radius_km: float) -> str:
    """``< 1 km``, ``1-5 km``, … with the radius itself as the last limit."""
    band_limits_km = [limit for limit in DISTANCE_BAND_LIMITS_KM if limit < radius_km] + [radius_km]
    lower_limit_km = 0.0
    upper_limit_km = band_limits_km[0]
    for upper_limit_km in band_limits_km:
        # The radius itself belongs to the last band: radius search includes photos exactly at the radius.
        if distance_km < upper_limit_km or upper_limit_km == radius_km:
            break
        lower_limit_km = upper_limit_km
    if lower_limit_km == 0.0:
        return f"< {upper_limit_km:g} km"
    return f"{lower_limit_km:g}-{upper_limit_km:g} km"


def search_by_radius(*, query: RadiusQuery, locations: Iterable[PhotoLocation]) -> SearchResult:
    coordinates_by_path = {
        location.relative_path: location.coordinates for location in locations if location.coordinates is not None
    }
    if not coordinates_by_path:
        return SearchResult(scores={}, section_labels={}, summary=f"No photos with GPS data {query}")
    distances_km = compute_distances_km(
        origin=query.center,
        latitudes_deg=np.array([point.latitude_deg for point in coordinates_by_path.values()]),
        longitudes_deg=np.array([point.longitude_deg for point in coordinates_by_path.values()]),
    )
    scores: dict[str, float] = {}
    section_labels: dict[str, str] = {}
    for relative_path, distance_km in zip(coordinates_by_path, distances_km.tolist(), strict=True):
        if distance_km <= query.radius_km:
            scores[relative_path] = -distance_km
            section_labels[relative_path] = describe_distance_band(distance_km, radius_km=query.radius_km)
    return SearchResult(
        scores=scores, section_labels=section_labels, summary=f"{len(scores):,} photos {query}, nearest first"
    )


def find_place_match(*, query_text: str, locations: dict[str, PhotoLocation]) -> PlaceMatch | None:
    """Longest run of query words that names the town, region or country of at least one photo."""
    paths_by_name: defaultdict[str, list[str]] = defaultdict(list)
    for location in locations.values():
        if location.place is not None:
            for name in set(location.place.normalized_names):
                paths_by_name[name].append(location.relative_path)
    query_words = query_text.split()
    normalized_words = [normalize_place_name(word) for word in query_words]
    for word_count in range(len(query_words), 0, -1):
        for start_index in range(len(query_words) - word_count + 1):
            end_index = start_index + word_count
            matching_paths = paths_by_name.get(" ".join(normalized_words[start_index:end_index]))
            if matching_paths:
                return PlaceMatch(
                    place_name=" ".join(query_words[start_index:end_index]),
                    matching_paths=matching_paths,
                    remaining_text=" ".join(query_words[:start_index] + query_words[end_index:]),
                )
    return None


def search_by_place(*, match: PlaceMatch, locations: dict[str, PhotoLocation]) -> SearchResult:
    """Photos taken in the place, one section per town (most photos first); scan order within a town."""
    paths_by_label: defaultdict[str, list[str]] = defaultdict(list)
    for relative_path in match.matching_paths:
        place = locations[relative_path].place
        if place is not None:
            paths_by_label[place.path_label].append(relative_path)
    ordered_labels = sorted(paths_by_label, key=lambda label: (-len(paths_by_label[label]), label))
    scores: dict[str, float] = {}
    section_labels: dict[str, str] = {}
    for group_index, label in enumerate(ordered_labels):
        for relative_path in paths_by_label[label]:
            scores[relative_path] = -float(group_index)
            section_labels[relative_path] = label
    return SearchResult(
        scores=scores,
        section_labels=section_labels,
        summary=f"{len(scores):,} photos taken in {match.place_name}, by town",
    )


def rank_within_place(*, match: PlaceMatch, text_scores: dict[str, float]) -> SearchResult:
    """``beach madeira``: only photos taken in Madeira, ranked by how well they match ``beach``."""
    scores = {
        relative_path: text_scores[relative_path]
        for relative_path in match.matching_paths
        if relative_path in text_scores
    }
    return SearchResult(
        scores=scores,
        section_labels={},
        summary=f"{len(scores):,} photos taken in {match.place_name}, best '{match.remaining_text}' matches first",
    )
