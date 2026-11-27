from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

from lazyrate.geo_map import (
    Basemap,
    BrowserMapWriter,
    MapPoint,
    MapViewport,
    build_town_labels,
    choose_scale_bar_km,
    find_points_near,
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


def test_world_view_without_photos_keeps_the_aspect_ratio_of_the_pane() -> None:
    wide_extent = fit_extent([], aspect_ratio=2.5)
    narrow_extent = fit_extent([], aspect_ratio=1.4)

    assert wide_extent.longitude_span_deg / wide_extent.latitude_span_deg == pytest.approx(2.5)
    assert narrow_extent.longitude_span_deg / narrow_extent.latitude_span_deg == pytest.approx(1.4)
    assert narrow_extent.latitude_span_deg <= 170


def test_zooming_in_keeps_the_point_under_the_mouse_in_place() -> None:
    fitted_extent = fit_extent([FUNCHAL, MUNICH], aspect_ratio=1.5)
    viewport = MapViewport()
    anchor_x_ratio, anchor_y_ratio = 0.8, 0.3
    anchor_longitude_deg = fitted_extent.min_longitude_deg + anchor_x_ratio * fitted_extent.longitude_span_deg
    anchor_latitude_deg = fitted_extent.max_latitude_deg - anchor_y_ratio * fitted_extent.latitude_span_deg

    zoomed_extent = viewport.zoom(steps=2, anchor_x_ratio=anchor_x_ratio, anchor_y_ratio=anchor_y_ratio).apply(
        fitted_extent
    )

    assert zoomed_extent.longitude_span_deg == pytest.approx(fitted_extent.longitude_span_deg / 4)
    assert zoomed_extent.min_longitude_deg + anchor_x_ratio * zoomed_extent.longitude_span_deg == pytest.approx(
        anchor_longitude_deg
    )
    assert zoomed_extent.max_latitude_deg - anchor_y_ratio * zoomed_extent.latitude_span_deg == pytest.approx(
        anchor_latitude_deg
    )


def test_panning_and_centering_move_the_visible_area() -> None:
    fitted_extent = fit_extent([FUNCHAL, MUNICH], aspect_ratio=1.5)
    zoomed_viewport = MapViewport().zoom(steps=1)

    east_extent = zoomed_viewport.pan(x_steps=1, y_steps=0).apply(fitted_extent)
    north_extent = zoomed_viewport.pan(x_steps=0, y_steps=-1).apply(fitted_extent)
    centered_extent = MapViewport().center_on(x_ratio=1.0, y_ratio=0.5).apply(fitted_extent)

    visible_width_deg = fitted_extent.longitude_span_deg / 2
    assert east_extent.min_longitude_deg - zoomed_viewport.apply(fitted_extent).min_longitude_deg == pytest.approx(
        visible_width_deg * MapViewport.PAN_STEP_RATIO
    )
    assert north_extent.max_latitude_deg > zoomed_viewport.apply(fitted_extent).max_latitude_deg
    assert (centered_extent.min_longitude_deg + centered_extent.max_longitude_deg) / 2 == pytest.approx(
        fitted_extent.max_longitude_deg
    )
    assert MapViewport().zoom(steps=1).zoom(steps=-1).is_fitted


def test_dragging_moves_the_map_with_the_mouse() -> None:
    fitted_extent = fit_extent([FUNCHAL, MUNICH], aspect_ratio=1.5)

    dragged_up_extent = MapViewport().drag(x_ratio_delta=0.0, y_ratio_delta=-0.25).apply(fitted_extent)

    # Grabbing the map and pulling it up brings the south into view.
    assert dragged_up_extent.min_latitude_deg < fitted_extent.min_latitude_deg
    assert dragged_up_extent.max_latitude_deg - fitted_extent.max_latitude_deg == pytest.approx(
        -0.25 * fitted_extent.latitude_span_deg
    )


def test_points_near_a_click_are_found_nearest_first() -> None:
    extent = fit_extent([FUNCHAL, MUNICH], aspect_ratio=1.0)
    points = [
        MapPoint(relative_path="munich.jpg", coordinates=MUNICH, rating=None, place=None),
        MapPoint(relative_path="funchal_far.jpg", coordinates=FUNCHAL, rating=None, place=None),
        MapPoint(relative_path="funchal_near.jpg", coordinates=FUNCHAL, rating=None, place=None),
    ]
    ((funchal_x_px, funchal_y_px),) = extent.project(
        longitudes_deg=np.array([FUNCHAL.longitude_deg]),
        latitudes_deg=np.array([FUNCHAL.latitude_deg]),
        width_px=400,
        height_px=400,
    ).tolist()

    near_paths = find_points_near(
        points,
        extent=extent,
        image_size=(400, 400),
        x_ratio=(funchal_x_px + 3) / 400,
        y_ratio=funchal_y_px / 400,
        max_distance_px=10,
    )
    missed_paths = find_points_near(
        points, extent=extent, image_size=(400, 400), x_ratio=0.5, y_ratio=0.5, max_distance_px=10
    )

    assert near_paths == ["funchal_far.jpg", "funchal_near.jpg"]
    assert missed_paths == []


def test_zoom_level_is_limited() -> None:
    viewport = MapViewport().zoom(steps=100)

    assert viewport.zoom_level == MapViewport.MAX_ZOOM_LEVEL


def test_a_notice_is_written_across_an_empty_map() -> None:
    basemap = Basemap()

    plain_map = basemap.render_base(points=[], labels=[], width_px=400, height_px=300)
    noticed_map = basemap.render_base(points=[], labels=[], width_px=400, height_px=300, notice="No GPS data")

    assert plain_map.image.tobytes() != noticed_map.image.tobytes()


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
