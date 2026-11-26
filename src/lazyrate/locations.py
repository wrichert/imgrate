"""Where photos were taken: EXIF GPS coordinates plus the nearest town, looked up offline.

The town table (``data/places.tsv.gz``) is built from GeoNames (CC BY 4.0) by ``scripts/build_places.py``:
every town with at least 1,000 inhabitants, with its region (state, province, island group) and country.
Nothing is sent over the network.
"""

from __future__ import annotations

import gzip
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

from lazyrate.media import IMAGE_EXTENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    from lazyrate.media import MediaFile

register_heif_opener()

EARTH_RADIUS_KM = 6371.0088
PLACE_PATH_SEPARATOR = " › "  # noqa: RUF001
NO_LOCATION_LABEL = "no location"
TOWN_RADIUS_KM_PER_SQRT_POPULATION = 0.01
MAX_TOWN_RADIUS_KM = 15.0


@dataclass(frozen=True, slots=True)
class GeoCoordinates:
    latitude_deg: float
    longitude_deg: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.latitude_deg <= 90.0 and -180.0 <= self.longitude_deg <= 180.0):
            raise ValueError(f"Coordinates out of range: {self.latitude_deg}, {self.longitude_deg}")

    def __str__(self) -> str:
        return f"{self.latitude_deg:.5f}, {self.longitude_deg:.5f}"


@dataclass(frozen=True, slots=True)
class Place:
    """Nearest town of a photo, with its region and country (any of them may be empty)."""

    town: str
    region: str
    country: str

    @property
    def path_label(self) -> str:
        """Country, region, town (coarse to fine, like a folder path)."""
        return PLACE_PATH_SEPARATOR.join(name for name in (self.country, self.region, self.town) if name)

    @property
    def normalized_names(self) -> tuple[str, ...]:
        return tuple(normalize_place_name(name) for name in (self.town, self.region, self.country) if name)


@dataclass(frozen=True, slots=True)
class PhotoLocation:
    """GPS position of one file, valid as long as size and modification time are unchanged.

    ``coordinates`` and ``place`` are ``None`` for photos without GPS data, so they are not read again.
    """

    relative_path: str
    size_bytes: int
    modified_timestamp_utc: float
    coordinates: GeoCoordinates | None
    place: Place | None


def normalize_place_name(name: str) -> str:
    """Case- and accent-insensitive form: ``São Vicente`` -> ``sao vicente``."""
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(without_accents.casefold().replace("-", " ").split())


def read_gps_coordinates(path: Path) -> GeoCoordinates | None:
    """EXIF GPS position, or ``None`` if the file has none (or a 0/0 placeholder) or cannot be read."""
    try:
        with Image.open(path) as image:
            gps_tags = image.getexif().get_ifd(ExifTags.IFD.GPSInfo)
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        return None
    try:
        latitude_deg = convert_to_degrees(
            gps_tags[ExifTags.GPS.GPSLatitude], reference=gps_tags.get(ExifTags.GPS.GPSLatitudeRef, "N")
        )
        longitude_deg = convert_to_degrees(
            gps_tags[ExifTags.GPS.GPSLongitude], reference=gps_tags.get(ExifTags.GPS.GPSLongitudeRef, "E")
        )
        coordinates = GeoCoordinates(latitude_deg=latitude_deg, longitude_deg=longitude_deg)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if latitude_deg == 0.0 and longitude_deg == 0.0:
        return None
    return coordinates


def convert_to_degrees(degrees_minutes_seconds: Sequence[float], *, reference: str | bytes) -> float:
    """EXIF stores (degrees, minutes, seconds) plus a hemisphere letter; south and west are negative."""
    degrees, minutes, seconds = (float(value) for value in degrees_minutes_seconds)
    decimal_degrees = degrees + minutes / 60.0 + seconds / 3600.0
    reference_text = reference.decode("ascii", "ignore") if isinstance(reference, bytes) else str(reference)
    return -decimal_degrees if reference_text.strip("\x00 ").upper() in {"S", "W"} else decimal_degrees


def convert_to_unit_vectors(*, latitudes_deg: np.ndarray, longitudes_deg: np.ndarray) -> np.ndarray:
    latitudes_rad = np.radians(latitudes_deg)
    longitudes_rad = np.radians(longitudes_deg)
    return np.column_stack(
        (
            np.cos(latitudes_rad) * np.cos(longitudes_rad),
            np.cos(latitudes_rad) * np.sin(longitudes_rad),
            np.sin(latitudes_rad),
        )
    ).astype(np.float32)


def compute_distances_km(
    *, origin: GeoCoordinates, latitudes_deg: np.ndarray, longitudes_deg: np.ndarray
) -> np.ndarray:
    """Great-circle (haversine) distances from ``origin``."""
    origin_latitude_rad = np.radians(origin.latitude_deg)
    latitudes_rad = np.radians(latitudes_deg)
    half_latitude_delta = (latitudes_rad - origin_latitude_rad) / 2.0
    half_longitude_delta = np.radians(longitudes_deg - origin.longitude_deg) / 2.0
    haversine = (
        np.sin(half_latitude_delta) ** 2
        + np.cos(origin_latitude_rad) * np.cos(latitudes_rad) * np.sin(half_longitude_delta) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


@dataclass(frozen=True, slots=True)
class PlaceTable:
    """All towns as unit vectors on the sphere, with their approximate radius."""

    town_vectors: np.ndarray
    town_radii_km: np.ndarray
    places: list[Place]


class PlaceIndex:
    """Nearest-town lookup over the bundled GeoNames table; loaded on first use (~0.3 s), then thread-safe.

    Pure nearest-point lookup would name the small suburb next to a city centre, so every town gets a radius
    that grows with its population and a photo belongs to the town whose edge is nearest (or that it lies in).
    """

    DATA_PACKAGE = "lazyrate"
    DATA_FILE_PATH = ("data", "places.tsv.gz")
    QUERY_CHUNK_SIZE = 64
    """Photos compared against all ~160k towns at once: 64 x 160k float32 = 42 MB of scores."""
    CANDIDATE_COUNT = 64
    """Nearest town centres considered per photo before the radius correction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._table: PlaceTable | None = None

    def __repr__(self) -> str:
        return f"PlaceIndex(is_loaded={self._table is not None})"

    def find_nearest_places(self, coordinates: Sequence[GeoCoordinates]) -> list[Place]:
        if not coordinates:
            return []
        table = self._load()
        photo_vectors = convert_to_unit_vectors(
            latitudes_deg=np.array([point.latitude_deg for point in coordinates]),
            longitudes_deg=np.array([point.longitude_deg for point in coordinates]),
        )
        nearest_indices: list[int] = []
        for chunk_start in range(0, len(photo_vectors), self.QUERY_CHUNK_SIZE):
            chunk = photo_vectors[chunk_start : chunk_start + self.QUERY_CHUNK_SIZE]
            nearest_indices.extend(select_nearest_towns(chunk, table=table, candidate_count=self.CANDIDATE_COUNT))
        return [table.places[index] for index in nearest_indices]

    def _load(self) -> PlaceTable:
        with self._lock:
            if self._table is None:
                self._table = load_place_table(
                    resources.files(self.DATA_PACKAGE).joinpath(*self.DATA_FILE_PATH).read_bytes()
                )
            return self._table


def select_nearest_towns(photo_vectors: np.ndarray, *, table: PlaceTable, candidate_count: int) -> list[int]:
    """Per photo: among the nearest town centres, the one with the smallest distance to its edge."""
    # On the unit sphere a larger dot product means a smaller great-circle distance.
    similarities = photo_vectors @ table.town_vectors.T
    candidate_count = min(candidate_count, similarities.shape[1])
    candidate_indices = np.argpartition(-similarities, candidate_count - 1, axis=1)[:, :candidate_count]
    candidate_similarities = np.take_along_axis(similarities, candidate_indices, axis=1)
    candidate_distances_km = EARTH_RADIUS_KM * np.arccos(np.clip(candidate_similarities, -1.0, 1.0))
    edge_distances_km = candidate_distances_km - table.town_radii_km[candidate_indices]
    best_columns = np.argmin(edge_distances_km, axis=1)
    return candidate_indices[np.arange(len(candidate_indices)), best_columns].tolist()


def estimate_town_radius_km(populations: np.ndarray) -> np.ndarray:
    """Rough built-up radius: ~3 km for 100k inhabitants, capped so a metropolis does not swallow its neighbours."""
    return np.minimum(TOWN_RADIUS_KM_PER_SQRT_POPULATION * np.sqrt(populations), MAX_TOWN_RADIUS_KM)


def load_place_table(compressed_table: bytes) -> PlaceTable:
    """Parse ``latitude  longitude  population  town  region  country`` rows."""
    rows = [line.split("\t") for line in gzip.decompress(compressed_table).decode("utf-8").splitlines()]
    latitudes_deg = np.array([float(row[0]) for row in rows])
    longitudes_deg = np.array([float(row[1]) for row in rows])
    populations = np.array([float(row[2]) for row in rows])
    return PlaceTable(
        town_vectors=convert_to_unit_vectors(latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg),
        town_radii_km=estimate_town_radius_km(populations).astype(np.float32),
        places=[Place(town=row[3], region=row[4], country=row[5]) for row in rows],
    )


def find_files_missing_locations(*, files: Iterable[MediaFile], locations: dict[str, PhotoLocation]) -> list[MediaFile]:
    """Image files whose GPS data was never read or changed since."""
    pending_files: list[MediaFile] = []
    for media_file in files:
        if media_file.extension not in IMAGE_EXTENSIONS:
            continue
        location = locations.get(media_file.relative_path)
        if (
            location is None
            or location.size_bytes != media_file.size_bytes
            or location.modified_timestamp_utc != media_file.modified_timestamp_utc
        ):
            pending_files.append(media_file)
    return pending_files


class LocationReader:
    """Reads GPS data of many files on a thread pool and names the nearest towns, in batches."""

    WORKER_COUNT = 8
    BATCH_SIZE = 256

    def __init__(self, *, root: Path, place_index: PlaceIndex) -> None:
        self._root = root
        self._place_index = place_index

    def __repr__(self) -> str:
        return f"LocationReader({self._root!s})"

    def read_locations(
        self,
        files: Sequence[MediaFile],
        *,
        on_batch: Callable[[list[PhotoLocation], int], None],
        should_stop: Callable[[], bool],
    ) -> None:
        """Calls ``on_batch(locations, done_count)`` after every batch; stops early when ``should_stop()``."""
        with ThreadPoolExecutor(self.WORKER_COUNT, thread_name_prefix="location") as executor:
            for batch_start in range(0, len(files), self.BATCH_SIZE):
                if should_stop():
                    return
                batch_files = files[batch_start : batch_start + self.BATCH_SIZE]
                batch_paths = [self._root / media_file.relative_path for media_file in batch_files]
                coordinates = list(executor.map(read_gps_coordinates, batch_paths))
                on_batch(self._build_locations(batch_files, coordinates), batch_start + len(batch_files))

    def _build_locations(
        self, batch_files: Sequence[MediaFile], coordinates: Sequence[GeoCoordinates | None]
    ) -> list[PhotoLocation]:
        known_coordinates = [point for point in coordinates if point is not None]
        nearest_places = iter(self._place_index.find_nearest_places(known_coordinates))
        return [
            PhotoLocation(
                relative_path=media_file.relative_path,
                size_bytes=media_file.size_bytes,
                modified_timestamp_utc=media_file.modified_timestamp_utc,
                coordinates=point,
                place=None if point is None else next(nearest_places),
            )
            for media_file, point in zip(batch_files, coordinates, strict=True)
        ]
