from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest

from lazyrate.geo_map import (
    Basemap,
    BrowserMapWriter,
    MapPoint,
    build_town_labels,
    choose_scale_bar_km,
    fit_extent,
)
from lazyrate.locations import GeoCoordinates, Place

if TYPE_CHECKING:
    from pathlib import Path

FUNCHAL = GeoCoordinates(latitude_deg=32.6657, longitude_deg=-16.9255)
MUNICH = GeoCoordinates(latitude_deg=48.1374, longitude_deg=11.5755)
WATER_COLOR = Basemap.COLORS.water
LAND_COLOR = Basemap.COLORS.land


def test_extent_contains_all_points_with_the_requested_aspect_ratio() -> None:
    extent = fit_extent([FUNCHAL, MUNICH], aspect_ratio=1.5)

    assert extent.min_longitude_deg < FUNCHAL.longitude_deg < MUNICH.longitude_deg < extent.max_longitude_deg
    assert extent.min_latitude_deg < FUNCHAL.latitude_deg < MUNICH.latitude_deg < extent.max_latitude_deg
    assert extent.width_km / (extent.latitude_span_deg * 111.195) == pytest.approx(1.5, rel=0.02)


def test_a_single_photo_still_shows_its_surroundings() -> None:
    extent = fit_extent([FUNCHAL], aspect_ratio=1.0)

    assert extent.width_km >= 25


def test_rendered_map_shows_land_water_and_the_highlighted_photo() -> None:
    image = Basemap().render(
        points=[FUNCHAL, GeoCoordinates(latitude_deg=32.80, longitude_deg=-16.88)],
        highlighted=FUNCHAL,
        width_px=400,
        height_px=300,
    )

    colors = {color for _count, color in image.getcolors(maxcolors=100_000)}
    assert (image.width, image.height) == (400, 300)
    assert WATER_COLOR in colors  # the Atlantic around Madeira
    assert LAND_COLOR in colors  # Madeira itself
    assert Basemap.COLORS.current in colors


def test_highlight_is_drawn_on_a_copy_of_the_cached_base_map() -> None:
    basemap = Basemap()
    base_map = basemap.render_base(points=[FUNCHAL], labels=[], width_px=200, height_px=200)

    highlighted_image = basemap.draw_highlight(base_map, highlighted=FUNCHAL)

    base_colors = {color for _count, color in base_map.image.getcolors(maxcolors=100_000)}
    highlighted_colors = {color for _count, color in highlighted_image.getcolors(maxcolors=100_000)}
    assert Basemap.COLORS.current not in base_colors
    assert Basemap.COLORS.current in highlighted_colors


def test_town_labels_put_the_busiest_towns_first() -> None:
    funchal = Place(town="Funchal", region="Madeira", country="Portugal")
    santana = Place(town="Santana", region="Madeira", country="Portugal")
    points = [
        MapPoint(relative_path="a.jpg", coordinates=FUNCHAL, rating=None, place=santana),
        MapPoint(relative_path="b.jpg", coordinates=FUNCHAL, rating=None, place=funchal),
        MapPoint(relative_path="c.jpg", coordinates=FUNCHAL, rating=None, place=funchal),
        MapPoint(relative_path="d.jpg", coordinates=FUNCHAL, rating=None, place=None),
    ]

    labels = build_town_labels(points)

    assert [label.text for label in labels] == ["Funchal", "Santana"]


def test_scale_bar_uses_round_distances() -> None:
    assert choose_scale_bar_km(90.0) == 20
    assert choose_scale_bar_km(1.0) == 0.5


def test_browser_map_embeds_photos_thumbnails_and_escapes_script_tags(geo_root: Path, tmp_path: Path) -> None:
    points = [
        MapPoint(
            relative_path="madeira/funchal_1.jpg",
            coordinates=FUNCHAL,
            rating=5,
            place=Place(town="Funchal", region="Madeira", country="Portugal"),
        )
    ]
    output_path = tmp_path / "map.html"
    progress_reports: list[tuple[int, int]] = []

    is_written = BrowserMapWriter(root=geo_root).write(
        points,
        output_path=output_path,
        title="trip </script><b>",
        on_progress=lambda done_count, total_count: progress_reports.append((done_count, total_count)),
        should_stop=lambda: False,
    )

    page = output_path.read_text()
    photos_json = re.search(r"const photos = (.*);\n", page)
    assert is_written
    assert progress_reports == [(1, 1)]
    assert photos_json is not None
    (photo,) = json.loads(photos_json.group(1).replace("<\\/", "</"))
    assert photo["thumb"].startswith("data:image/jpeg;base64,")
    assert photo["rating"] == "★★★★★"
    assert photo["uri"].startswith("file://")
    assert "trip &lt;/script&gt;&lt;b&gt;" in page
    assert page.count("</script>") == 3  # the two library tags and the inline script, nothing injected
