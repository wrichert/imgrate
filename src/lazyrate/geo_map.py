"""Maps of where photos were taken: an offline image for the side pane and an HTML page for the browser.

The offline map draws Natural Earth land outlines and country borders (public domain, ``data/basemap.npz``, built by
``scripts/build_basemap.py``). The browser map uses Leaflet with online map tiles; the photos, thumbnails and
coordinates are embedded in the page and never uploaded, but the tile servers see which map area is viewed.
"""

from __future__ import annotations

import base64
import functools
import html
import io
import json
import math
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from importlib import resources
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lazyrate.photo_files import PHOTO_READ_ERRORS, open_photo, orient_photo
from lazyrate.ratings import describe_bucket

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from lazyrate.locations import GeoCoordinates, Place

MAX_LATITUDE_DEG = 85.0
WORLD_CENTER_LONGITUDE_DEG = 10.0
"""The world view (no photo with GPS) is centered on Europe / Africa when the pane is too narrow for all of it."""
LABEL_FONT_CANDIDATES = (
    "Arial.ttf",
    "DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
"""System fonts with accented letters; Pillow's built-in font only covers ASCII well."""


@dataclass(frozen=True, slots=True)
class LabelFont:
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    is_ascii_only: bool

    def prepare_text(self, text: str) -> str:
        """``São Jorge`` -> ``Sao Jorge`` when only the built-in font is available."""
        if not self.is_ascii_only:
            return text
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(character for character in decomposed if character.isascii())


@functools.cache
def load_label_font(size_px: int) -> LabelFont:
    for font_name in LABEL_FONT_CANDIDATES:
        try:
            return LabelFont(font=ImageFont.truetype(font_name, size_px), is_ascii_only=False)
        except OSError:
            continue
    return LabelFont(font=ImageFont.load_default(size=size_px), is_ascii_only=True)


@dataclass(frozen=True, slots=True)
class MapPoint:
    relative_path: str
    coordinates: GeoCoordinates
    rating: int | None
    place: Place | None


@dataclass(frozen=True, slots=True)
class MapLabel:
    """A town name drawn next to its photos."""

    coordinates: GeoCoordinates
    text: str


@dataclass(frozen=True, slots=True)
class MapExtent:
    """Visible rectangle in degrees (equirectangular projection, longitudes scaled by cos(mid latitude))."""

    min_longitude_deg: float
    max_longitude_deg: float
    min_latitude_deg: float
    max_latitude_deg: float

    @property
    def longitude_span_deg(self) -> float:
        return self.max_longitude_deg - self.min_longitude_deg

    @property
    def latitude_span_deg(self) -> float:
        return self.max_latitude_deg - self.min_latitude_deg

    @property
    def width_km(self) -> float:
        mid_latitude_rad = math.radians((self.min_latitude_deg + self.max_latitude_deg) / 2)
        return self.longitude_span_deg * KM_PER_DEGREE * math.cos(mid_latitude_rad)

    def project(
        self, *, longitudes_deg: np.ndarray, latitudes_deg: np.ndarray, width_px: int, height_px: int
    ) -> np.ndarray:
        """``(N, 2)`` pixel positions."""
        x_px = (longitudes_deg - self.min_longitude_deg) / self.longitude_span_deg * width_px
        y_px = (self.max_latitude_deg - latitudes_deg) / self.latitude_span_deg * height_px
        return np.column_stack((x_px, y_px))


KM_PER_DEGREE = 111.195


def fit_extent(
    points: Sequence[GeoCoordinates], *, aspect_ratio: float, padding_ratio: float = 0.12, min_span_km: float = 25.0
) -> MapExtent:
    """Smallest rectangle around ``points`` (plus padding, at least ``min_span_km``) with ``aspect_ratio`` = w / h."""
    if not points:
        return fit_world_extent(aspect_ratio=aspect_ratio)
    longitudes_deg = [point.longitude_deg for point in points]
    latitudes_deg = [point.latitude_deg for point in points]
    center_latitude_deg = (min(latitudes_deg) + max(latitudes_deg)) / 2
    center_longitude_deg = (min(longitudes_deg) + max(longitudes_deg)) / 2
    longitude_scale = max(math.cos(math.radians(center_latitude_deg)), 0.05)
    min_span_deg = min_span_km / KM_PER_DEGREE
    # Work in "ground degrees" (longitudes shrunk by cos(latitude)) so the aspect ratio is geographic.
    ground_width_deg = max((max(longitudes_deg) - min(longitudes_deg)) * longitude_scale, min_span_deg)
    ground_height_deg = max(max(latitudes_deg) - min(latitudes_deg), min_span_deg)
    ground_width_deg *= 1 + 2 * padding_ratio
    ground_height_deg *= 1 + 2 * padding_ratio
    if ground_width_deg / ground_height_deg < aspect_ratio:
        ground_width_deg = ground_height_deg * aspect_ratio
    else:
        ground_height_deg = ground_width_deg / aspect_ratio
    half_longitude_span_deg = min(ground_width_deg / longitude_scale / 2, 180.0)
    half_latitude_span_deg = min(ground_height_deg / 2, MAX_LATITUDE_DEG)
    return MapExtent(
        min_longitude_deg=center_longitude_deg - half_longitude_span_deg,
        max_longitude_deg=center_longitude_deg + half_longitude_span_deg,
        min_latitude_deg=center_latitude_deg - half_latitude_span_deg,
        max_latitude_deg=center_latitude_deg + half_latitude_span_deg,
    )


def fit_world_extent(*, aspect_ratio: float) -> MapExtent:
    """The whole world if the pane is wide enough, else its middle; undistorted either way."""
    height_deg = min(360.0 / aspect_ratio, 2 * MAX_LATITUDE_DEG)
    half_width_deg = height_deg * aspect_ratio / 2
    return MapExtent(
        min_longitude_deg=WORLD_CENTER_LONGITUDE_DEG - half_width_deg,
        max_longitude_deg=WORLD_CENTER_LONGITUDE_DEG + half_width_deg,
        min_latitude_deg=-height_deg / 2,
        max_latitude_deg=height_deg / 2,
    )


@dataclass(frozen=True, slots=True)
class MapViewport:
    """Zoom and pan on top of the extent fitted to the photos.

    The center is measured in widths / heights of the fitted extent from its middle; ``x`` grows east, ``y`` south
    (like pixels).
    """

    ZOOM_STEP_FACTOR: ClassVar[float] = 2.0
    MIN_ZOOM_LEVEL: ClassVar[int] = -4
    MAX_ZOOM_LEVEL: ClassVar[int] = 14
    PAN_STEP_RATIO: ClassVar[float] = 0.25
    """One pan step moves the map by this share of the visible width or height."""

    zoom_level: int = 0
    center_x_ratio: float = 0.0
    center_y_ratio: float = 0.0

    @property
    def zoom_factor(self) -> float:
        return self.ZOOM_STEP_FACTOR**self.zoom_level

    @property
    def is_fitted(self) -> bool:
        return self == MapViewport()

    def zoom(self, *, steps: int, anchor_x_ratio: float = 0.5, anchor_y_ratio: float = 0.5) -> MapViewport:
        """Zoom in (``steps`` > 0) or out, keeping the map position under the anchor (0..1 of the pane) in place."""
        zoom_level = max(self.MIN_ZOOM_LEVEL, min(self.MAX_ZOOM_LEVEL, self.zoom_level + steps))
        new_zoom_factor = self.ZOOM_STEP_FACTOR**zoom_level
        anchor_x_offset = anchor_x_ratio - 0.5
        anchor_y_offset = anchor_y_ratio - 0.5
        return MapViewport(
            zoom_level=zoom_level,
            center_x_ratio=self.center_x_ratio + anchor_x_offset / self.zoom_factor - anchor_x_offset / new_zoom_factor,
            center_y_ratio=self.center_y_ratio + anchor_y_offset / self.zoom_factor - anchor_y_offset / new_zoom_factor,
        )

    def pan(self, *, x_steps: int, y_steps: int) -> MapViewport:
        step_ratio = self.PAN_STEP_RATIO / self.zoom_factor
        return replace(
            self,
            center_x_ratio=self.center_x_ratio + x_steps * step_ratio,
            center_y_ratio=self.center_y_ratio + y_steps * step_ratio,
        )

    def drag(self, *, x_ratio_delta: float, y_ratio_delta: float) -> MapViewport:
        """The map follows the mouse: dragging up shows what lies further south."""
        return replace(
            self,
            center_x_ratio=self.center_x_ratio - x_ratio_delta / self.zoom_factor,
            center_y_ratio=self.center_y_ratio - y_ratio_delta / self.zoom_factor,
        )

    def center_on(self, *, x_ratio: float, y_ratio: float) -> MapViewport:
        """Move the map position at ``x_ratio`` / ``y_ratio`` (0..1 of the pane) to the middle."""
        return replace(
            self,
            center_x_ratio=self.center_x_ratio + (x_ratio - 0.5) / self.zoom_factor,
            center_y_ratio=self.center_y_ratio + (y_ratio - 0.5) / self.zoom_factor,
        )

    def apply(self, fitted_extent: MapExtent) -> MapExtent:
        center_longitude_deg = (
            fitted_extent.min_longitude_deg + (0.5 + self.center_x_ratio) * fitted_extent.longitude_span_deg
        )
        center_latitude_deg = (
            fitted_extent.max_latitude_deg - (0.5 + self.center_y_ratio) * fitted_extent.latitude_span_deg
        )
        half_longitude_span_deg = fitted_extent.longitude_span_deg / self.zoom_factor / 2
        half_latitude_span_deg = fitted_extent.latitude_span_deg / self.zoom_factor / 2
        return MapExtent(
            min_longitude_deg=center_longitude_deg - half_longitude_span_deg,
            max_longitude_deg=center_longitude_deg + half_longitude_span_deg,
            min_latitude_deg=center_latitude_deg - half_latitude_span_deg,
            max_latitude_deg=center_latitude_deg + half_latitude_span_deg,
        )


def find_points_near(
    points: Sequence[MapPoint],
    *,
    extent: MapExtent,
    image_size: tuple[int, int],
    x_ratio: float,
    y_ratio: float,
    max_distance_px: float,
) -> list[str]:
    """Paths of the photos drawn within ``max_distance_px`` of a position (0..1 of the map), nearest first."""
    if not points:
        return []
    width_px, height_px = image_size
    pixel_positions = extent.project(
        longitudes_deg=np.array([point.coordinates.longitude_deg for point in points]),
        latitudes_deg=np.array([point.coordinates.latitude_deg for point in points]),
        width_px=width_px,
        height_px=height_px,
    )
    distances_px = np.hypot(pixel_positions[:, 0] - x_ratio * width_px, pixel_positions[:, 1] - y_ratio * height_px)
    near_indices = np.flatnonzero(distances_px <= max_distance_px)
    near_indices = near_indices[np.argsort(distances_px[near_indices], kind="stable")]
    return [points[index].relative_path for index in near_indices.tolist()]


@dataclass(frozen=True, slots=True)
class RenderedMap:
    image: Image.Image
    extent: MapExtent


@dataclass(frozen=True, slots=True)
class OutlineSet:
    """Many polylines in one array: ``points[offsets[i]:offsets[i + 1]]`` is line ``i`` (lon, lat in degrees)."""

    points_deg: np.ndarray
    offsets: np.ndarray
    bounding_boxes_deg: np.ndarray
    """``(lines, 4)``: min lon, min lat, max lon, max lat."""
    hole_flags: np.ndarray

    def select_visible_lines(self, extent: MapExtent) -> np.ndarray:
        boxes = self.bounding_boxes_deg
        return np.flatnonzero(
            (boxes[:, 2] >= extent.min_longitude_deg)
            & (boxes[:, 0] <= extent.max_longitude_deg)
            & (boxes[:, 3] >= extent.min_latitude_deg)
            & (boxes[:, 1] <= extent.max_latitude_deg)
        )


def build_outline_set(*, points: np.ndarray, offsets: np.ndarray, hole_flags: np.ndarray | None) -> OutlineSet:
    points_deg = points.astype(np.float64) / BASEMAP_COORDINATE_SCALE
    starts = offsets[:-1]
    bounding_boxes_deg = np.column_stack(
        (
            np.minimum.reduceat(points_deg[:, 0], starts),
            np.minimum.reduceat(points_deg[:, 1], starts),
            np.maximum.reduceat(points_deg[:, 0], starts),
            np.maximum.reduceat(points_deg[:, 1], starts),
        )
    )
    return OutlineSet(
        points_deg=points_deg,
        offsets=offsets,
        bounding_boxes_deg=bounding_boxes_deg,
        hole_flags=np.zeros(len(starts), dtype=bool) if hole_flags is None else hole_flags,
    )


BASEMAP_COORDINATE_SCALE = 1000


@dataclass(frozen=True, slots=True)
class MapColors:
    water: tuple[int, int, int] = (24, 38, 56)
    land: tuple[int, int, int] = (58, 74, 66)
    coast: tuple[int, int, int] = (96, 118, 106)
    border: tuple[int, int, int] = (140, 150, 140)
    point: tuple[int, int, int] = (255, 196, 64)
    point_outline: tuple[int, int, int] = (60, 40, 0)
    current: tuple[int, int, int] = (255, 72, 72)
    current_outline: tuple[int, int, int] = (255, 255, 255)
    text: tuple[int, int, int] = (220, 220, 220)


class Basemap:
    """Offline world outlines; loaded on first use (~0.2 s), then thread-safe."""

    DATA_PACKAGE = "lazyrate"
    DATA_FILE_PATH = ("data", "basemap.npz")
    COLORS = MapColors()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._land: OutlineSet | None = None
        self._borders: OutlineSet | None = None

    def __repr__(self) -> str:
        return f"Basemap(is_loaded={self._land is not None})"

    def render(
        self,
        *,
        points: Sequence[GeoCoordinates],
        highlighted: GeoCoordinates | None,
        labels: Sequence[MapLabel] = (),
        width_px: int,
        height_px: int,
    ) -> Image.Image:
        """Land, borders, one dot per photo position, town labels, the highlighted photo as a ringed red dot."""
        base_map = self.render_base(points=points, labels=labels, width_px=width_px, height_px=height_px)
        return self.draw_highlight(base_map, highlighted=highlighted)

    def render_base(
        self,
        *,
        points: Sequence[GeoCoordinates],
        labels: Sequence[MapLabel],
        width_px: int,
        height_px: int,
        viewport: MapViewport | None = None,
        notice: str | None = None,
    ) -> RenderedMap:
        """Everything except the highlighted photo, framed around ``points`` (then zoomed / panned by ``viewport``);
        cache it while only the cursor moves. ``notice`` is written across the middle (e.g. "no GPS data")."""
        land, borders = self._load()
        extent = fit_extent(points, aspect_ratio=width_px / height_px)
        if viewport is not None:
            extent = viewport.apply(extent)
        image = Image.new("RGB", (width_px, height_px), self.COLORS.water)
        draw = ImageDraw.Draw(image)
        self._draw_land(draw, land=land, extent=extent, width_px=width_px, height_px=height_px)
        self._draw_borders(draw, borders=borders, extent=extent, width_px=width_px, height_px=height_px)
        self._draw_points(draw, points=points, highlighted=None, extent=extent, image_size=image.size)
        self._draw_labels(draw, labels=labels, extent=extent, image_size=image.size)
        draw_scale_bar(draw, extent=extent, image_size=image.size, color=self.COLORS.text)
        if notice is not None:
            self._draw_notice(draw, notice=notice, image_size=image.size)
        return RenderedMap(image=image, extent=extent)

    def draw_highlight(self, base_map: RenderedMap, *, highlighted: GeoCoordinates | None) -> Image.Image:
        """Copy of the base map with the highlighted photo marked (if it lies inside the map)."""
        image = base_map.image.copy()
        if highlighted is not None:
            self._draw_points(
                ImageDraw.Draw(image), points=[], highlighted=highlighted, extent=base_map.extent, image_size=image.size
            )
        return image

    def _load(self) -> tuple[OutlineSet, OutlineSet]:
        with self._lock:
            if self._land is None or self._borders is None:
                with resources.files(self.DATA_PACKAGE).joinpath(*self.DATA_FILE_PATH).open("rb") as data_file:
                    arrays = dict(np.load(data_file))
                self._land = build_outline_set(
                    points=arrays["land_points"], offsets=arrays["land_offsets"], hole_flags=arrays["land_hole_flags"]
                )
                self._borders = build_outline_set(
                    points=arrays["border_points"], offsets=arrays["border_offsets"], hole_flags=None
                )
            return self._land, self._borders

    def _draw_land(
        self, draw: ImageDraw.ImageDraw, *, land: OutlineSet, extent: MapExtent, width_px: int, height_px: int
    ) -> None:
        for line_index in land.select_visible_lines(extent):
            pixel_points = project_line(land, line_index, extent=extent, width_px=width_px, height_px=height_px)
            if pixel_points is None:
                continue
            is_hole = bool(land.hole_flags[line_index])
            fill_color = self.COLORS.water if is_hole else self.COLORS.land
            draw.polygon(pixel_points, fill=fill_color, outline=self.COLORS.coast)

    def _draw_borders(
        self, draw: ImageDraw.ImageDraw, *, borders: OutlineSet, extent: MapExtent, width_px: int, height_px: int
    ) -> None:
        for line_index in borders.select_visible_lines(extent):
            pixel_points = project_line(borders, line_index, extent=extent, width_px=width_px, height_px=height_px)
            if pixel_points is not None:
                draw.line(pixel_points, fill=self.COLORS.border, width=1)

    def _draw_points(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        points: Sequence[GeoCoordinates],
        highlighted: GeoCoordinates | None,
        extent: MapExtent,
        image_size: tuple[int, int],
    ) -> None:
        width_px, height_px = image_size
        dot_radius_px = max(2, min(width_px, height_px) // 120)
        if points:
            pixel_positions = extent.project(
                longitudes_deg=np.array([point.longitude_deg for point in points]),
                latitudes_deg=np.array([point.latitude_deg for point in points]),
                width_px=width_px,
                height_px=height_px,
            )
            # One dot per pixel position is enough; thousands of photos at one spot would only overdraw.
            for x_px, y_px in np.unique(np.round(pixel_positions).astype(np.int32), axis=0).tolist():
                draw.ellipse(
                    (x_px - dot_radius_px, y_px - dot_radius_px, x_px + dot_radius_px, y_px + dot_radius_px),
                    fill=self.COLORS.point,
                    outline=self.COLORS.point_outline,
                )
        if highlighted is not None:
            ((x_px, y_px),) = extent.project(
                longitudes_deg=np.array([highlighted.longitude_deg]),
                latitudes_deg=np.array([highlighted.latitude_deg]),
                width_px=width_px,
                height_px=height_px,
            ).tolist()
            ring_radius_px = dot_radius_px * 2 + 1
            draw.ellipse(
                (x_px - ring_radius_px, y_px - ring_radius_px, x_px + ring_radius_px, y_px + ring_radius_px),
                fill=self.COLORS.current,
                outline=self.COLORS.current_outline,
                width=2,
            )

    def _draw_notice(self, draw: ImageDraw.ImageDraw, *, notice: str, image_size: tuple[int, int]) -> None:
        width_px, height_px = image_size
        font_size_px = max(12, min(width_px, height_px) // 24)
        label_font = load_label_font(font_size_px)
        center_px = (width_px / 2, height_px / 2)
        text = label_font.prepare_text(notice)
        text_box = draw.textbbox(center_px, text, font=label_font.font, anchor="mm")
        padding_px = font_size_px // 2
        draw.rectangle(
            (text_box[0] - padding_px, text_box[1] - padding_px, text_box[2] + padding_px, text_box[3] + padding_px),
            fill=self.COLORS.water,
            outline=self.COLORS.coast,
        )
        draw.text(center_px, text, fill=self.COLORS.text, font=label_font.font, anchor="mm")

    def _draw_labels(
        self, draw: ImageDraw.ImageDraw, *, labels: Sequence[MapLabel], extent: MapExtent, image_size: tuple[int, int]
    ) -> None:
        """Labels in the given order (most important first); a label overlapping an earlier one is skipped."""
        width_px, height_px = image_size
        font_size_px = max(10, min(width_px, height_px) // 32)
        label_font = load_label_font(font_size_px)
        offset_px = max(4, min(width_px, height_px) // 100)
        occupied_boxes: list[tuple[float, float, float, float]] = []
        for label in labels:
            ((x_px, y_px),) = extent.project(
                longitudes_deg=np.array([label.coordinates.longitude_deg]),
                latitudes_deg=np.array([label.coordinates.latitude_deg]),
                width_px=width_px,
                height_px=height_px,
            ).tolist()
            text = label_font.prepare_text(label.text)
            right_box = draw.textbbox((0, 0), text, font=label_font.font, stroke_width=2)
            text_width_px = right_box[2] - right_box[0]
            candidate_positions = (
                (x_px + offset_px, y_px - font_size_px / 2),  # right of the dot, else left of it
                (x_px - offset_px - text_width_px, y_px - font_size_px / 2),
            )
            for candidate_position in candidate_positions:
                text_box = draw.textbbox(candidate_position, text, font=label_font.font, stroke_width=2)
                is_inside = (
                    text_box[0] >= 0 and text_box[1] >= 0 and text_box[2] <= width_px and text_box[3] <= height_px
                )
                if is_inside and not any(boxes_overlap(text_box, occupied_box) for occupied_box in occupied_boxes):
                    break
            else:
                continue
            text_position = candidate_position
            occupied_boxes.append(text_box)
            draw.text(
                text_position,
                text,
                fill=self.COLORS.text,
                font=label_font.font,
                stroke_width=2,
                stroke_fill=self.COLORS.water,
            )


def boxes_overlap(first_box: tuple[float, float, float, float], second_box: tuple[float, float, float, float]) -> bool:
    return not (
        first_box[2] < second_box[0]
        or second_box[2] < first_box[0]
        or first_box[3] < second_box[1]
        or second_box[3] < first_box[1]
    )


def build_town_labels(points: Sequence[MapPoint], *, max_label_count: int = 12) -> list[MapLabel]:
    """The towns with the most photos first, each at the position of its first photo."""
    first_coordinates_by_town: dict[str, GeoCoordinates] = {}
    photo_counts_by_town: dict[str, int] = {}
    for point in points:
        if point.place is None or not point.place.town:
            continue
        first_coordinates_by_town.setdefault(point.place.town, point.coordinates)
        photo_counts_by_town[point.place.town] = photo_counts_by_town.get(point.place.town, 0) + 1
    busiest_towns = sorted(photo_counts_by_town, key=lambda town: -photo_counts_by_town[town])[:max_label_count]
    return [MapLabel(coordinates=first_coordinates_by_town[town], text=town) for town in busiest_towns]


def project_line(
    outlines: OutlineSet, line_index: int, *, extent: MapExtent, width_px: int, height_px: int
) -> list[tuple[int, int]] | None:
    """Pixel polyline of one outline, consecutive duplicate pixels removed; ``None`` if it collapses to a dot."""
    line_points = outlines.points_deg[outlines.offsets[line_index] : outlines.offsets[line_index + 1]]
    pixel_points = np.round(
        extent.project(
            longitudes_deg=line_points[:, 0], latitudes_deg=line_points[:, 1], width_px=width_px, height_px=height_px
        )
    ).astype(np.int32)
    is_new_pixel = np.ones(len(pixel_points), dtype=bool)
    is_new_pixel[1:] = np.any(pixel_points[1:] != pixel_points[:-1], axis=1)
    pixel_points = pixel_points[is_new_pixel]
    if len(pixel_points) < 2:
        return None
    return [(int(x_px), int(y_px)) for x_px, y_px in pixel_points]


SCALE_BAR_STEPS_KM = (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def choose_scale_bar_km(map_width_km: float) -> float:
    """Largest round distance that is at most a quarter of the map width."""
    fitting_steps_km = [step_km for step_km in SCALE_BAR_STEPS_KM if step_km <= map_width_km / 4]
    return fitting_steps_km[-1] if fitting_steps_km else SCALE_BAR_STEPS_KM[0]


def draw_scale_bar(
    draw: ImageDraw.ImageDraw, *, extent: MapExtent, image_size: tuple[int, int], color: tuple[int, int, int]
) -> None:
    width_px, height_px = image_size
    scale_km = choose_scale_bar_km(extent.width_km)
    bar_width_px = round(scale_km / extent.width_km * width_px)
    margin_px = max(6, width_px // 60)
    font_size_px = max(10, min(width_px, height_px) // 28)
    font = load_label_font(font_size_px).font
    bar_y_px = height_px - margin_px
    draw.line((margin_px, bar_y_px, margin_px + bar_width_px, bar_y_px), fill=color, width=2)
    for x_px in (margin_px, margin_px + bar_width_px):
        draw.line((x_px, bar_y_px - 4, x_px, bar_y_px), fill=color, width=2)
    draw.text((margin_px, bar_y_px - 6 - font_size_px), f"{scale_km:g} km", fill=color, font=font)


# ----------------------------------------------------------------------------- browser map

LEAFLET_URL = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4"
MARKER_CLUSTER_URL = "https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.5.3"
THUMBNAIL_EDGE_PX = 240
BROWSER_MAP_FILE_NAME = ".lazyrate-map.html"
"""Written next to the database: a folder lazyrate may write to (the photo folder may be read-only)."""
THUMBNAIL_JPEG_QUALITY = 72


@dataclass(frozen=True, slots=True)
class BrowserMapPhoto:
    point: MapPoint
    file_uri: str
    thumbnail_data_uri: str | None


def create_thumbnail_data_uri(path: Path, *, edge_px: int = THUMBNAIL_EDGE_PX) -> str | None:
    """Small JPEG as a ``data:`` URI (browsers cannot show HEIC, and the page must work without a server)."""
    try:
        with open_photo(path) as source_image:
            source_image.draft("RGB", (edge_px, edge_px))
            thumbnail = orient_photo(source_image).convert("RGB")
            thumbnail.thumbnail((edge_px, edge_px))
    except PHOTO_READ_ERRORS:
        return None
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class BrowserMapWriter:
    """Writes the browser map page: thumbnails decoded on a thread pool, everything embedded in one HTML file."""

    WORKER_COUNT = 6
    MAX_THUMBNAIL_COUNT = 3000
    """Beyond this the page would get too heavy (~10 KB per thumbnail); further markers show text only."""

    def __init__(self, *, root: Path) -> None:
        self._root = root

    def __repr__(self) -> str:
        return f"BrowserMapWriter({self._root!s})"

    def write(
        self,
        points: Sequence[MapPoint],
        *,
        output_path: Path,
        title: str,
        on_progress: Callable[[int, int], None],
        should_stop: Callable[[], bool],
    ) -> bool:
        """Write the page; ``False`` if stopped early. ``on_progress(done, total)`` after each thumbnail."""
        thumbnail_points = points[: self.MAX_THUMBNAIL_COUNT]
        thumbnails: list[str | None] = []
        with ThreadPoolExecutor(self.WORKER_COUNT, thread_name_prefix="map-thumbnail") as executor:
            paths = [self._root / point.relative_path for point in thumbnail_points]
            for done_count, thumbnail in enumerate(executor.map(create_thumbnail_data_uri, paths), start=1):
                if should_stop():
                    executor.shutdown(cancel_futures=True)
                    return False
                thumbnails.append(thumbnail)
                on_progress(done_count, len(thumbnail_points))
        thumbnails.extend(None for _ in points[len(thumbnail_points) :])
        photos = [
            BrowserMapPhoto(
                point=point, file_uri=(self._root / point.relative_path).as_uri(), thumbnail_data_uri=thumbnail
            )
            for point, thumbnail in zip(points, thumbnails, strict=True)
        ]
        output_path.write_text(build_map_html(photos=photos, title=title), encoding="utf-8")
        return True


def build_map_html(*, photos: Sequence[BrowserMapPhoto], title: str) -> str:
    """Self-contained page: clustered markers, a popup per photo with thumbnail, rating, place and path."""
    photo_rows = [
        {
            "lat": photo.point.coordinates.latitude_deg,
            "lon": photo.point.coordinates.longitude_deg,
            "path": photo.point.relative_path,
            "uri": photo.file_uri,
            "rating": describe_bucket(photo.point.rating),
            "place": "" if photo.point.place is None else photo.point.place.path_label,
            "thumb": photo.thumbnail_data_uri,
        }
        for photo in photos
    ]
    # "</" must not appear inside the inline script; JSON escapes are valid JavaScript string escapes.
    photos_json = json.dumps(photo_rows, ensure_ascii=False).replace("</", "<\\/")
    return (
        MAP_PAGE_TEMPLATE.replace("{{TITLE}}", html.escape(title))
        .replace("{{PHOTOS_JSON}}", photos_json)
        .replace("{{LEAFLET_URL}}", LEAFLET_URL)
        .replace("{{MARKER_CLUSTER_URL}}", MARKER_CLUSTER_URL)
    )


MAP_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<link rel="stylesheet" href="{{LEAFLET_URL}}/leaflet.min.css">
<link rel="stylesheet" href="{{MARKER_CLUSTER_URL}}/MarkerCluster.min.css">
<link rel="stylesheet" href="{{MARKER_CLUSTER_URL}}/MarkerCluster.Default.min.css">
<style>
  :root { color-scheme: light dark; --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; }
  @media (prefers-color-scheme: dark) { :root { --bg: #1c1c1e; --fg: #f2f2f7; --muted: #a1a1a6; } }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--fg);
               font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  #map { position: absolute; inset: 0; }
  #title { position: absolute; z-index: 1000; top: 10px; left: 56px; right: 56px; pointer-events: none; }
  #title span { display: inline-block; max-width: 100%; padding: 6px 10px; border-radius: 6px; background: var(--bg);
                box-shadow: 0 1px 4px rgba(0,0,0,.3); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .photo-popup img { display: block; max-width: 240px; max-height: 240px; border-radius: 4px; margin-bottom: 6px; }
  .photo-popup .rating { font-weight: 600; }
  .photo-popup .place, .photo-popup .path { color: #555; word-break: break-all; }
</style>
</head>
<body>
<div id="map"></div>
<div id="title"><span>{{TITLE}}</span></div>
<script src="{{LEAFLET_URL}}/leaflet.min.js"></script>
<script src="{{MARKER_CLUSTER_URL}}/leaflet.markercluster.min.js"></script>
<script>
const photos = {{PHOTOS_JSON}};
const map = L.map("map", { worldCopyJump: true });
const baseLayers = {
  "Map": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' }),
  "Satellite": L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19, attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics" }),
  "Terrain": L.tileLayer("https://tile.opentopomap.org/{z}/{x}/{y}.png", {
    maxZoom: 17,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, '
      + 'SRTM | &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)' }),
};
baseLayers["Map"].addTo(map);
L.control.layers(baseLayers).addTo(map);
L.control.scale().addTo(map);
const clusters = L.markerClusterGroup({ maxClusterRadius: 40 });
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}
function buildPopup(photo) {
  const popup = element("div", "photo-popup");
  if (photo.thumb) {
    const link = element("a");
    link.href = photo.uri;
    const image = element("img");
    image.src = photo.thumb;
    image.alt = photo.path;
    link.appendChild(image);
    popup.appendChild(link);
  }
  popup.appendChild(element("div", "rating", photo.rating));
  if (photo.place) popup.appendChild(element("div", "place", photo.place));
  popup.appendChild(element("div", "path", photo.path));
  return popup;
}
for (const photo of photos) {
  const marker = L.marker([photo.lat, photo.lon], { title: photo.path });
  marker.bindPopup(() => buildPopup(photo), { maxWidth: 260 });
  clusters.addLayer(marker);
}
map.addLayer(clusters);
if (photos.length) {
  map.fitBounds(clusters.getBounds(), { padding: [40, 40], maxZoom: 15 });
} else {
  map.setView([20, 0], 2);
}
</script>
</body>
</html>
"""
