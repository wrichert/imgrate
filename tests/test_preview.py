from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from conftest import save_photo_with_camera
from PIL import Image

from lazyrate.preview import ZoomView, compute_zoom_crop, read_photo_details

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("maker", "model", "lens", "expected_camera", "expected_lens"),
    [
        ("NIKON CORPORATION", "NIKON Z 6_2", "NIKKOR Z 24-70mm f/4 S", "NIKON Z 6_2", "NIKKOR Z 24-70mm f/4 S"),
        (
            "Apple",
            "iPhone 15 Pro",
            "iPhone 15 Pro back triple camera 6.86mm f/1.78",
            "Apple iPhone 15 Pro",
            "back triple camera 6.86mm f/1.78",
        ),
        ("OLYMPUS IMAGING CORP.", "E-M5", None, "OLYMPUS E-M5", None),
    ],
)
def test_details_name_camera_and_lens_without_repeating_the_maker(
    tmp_path: Path, maker: str, model: str, lens: str | None, expected_camera: str, expected_lens: str | None
) -> None:
    photo_path = tmp_path / "photo.jpg"
    save_photo_with_camera(photo_path, maker=maker, model=model, lens=lens)

    details = read_photo_details(photo_path)

    assert details is not None
    assert details.camera == expected_camera
    assert details.lens == expected_lens


def test_photo_without_exif_has_no_camera(tmp_path: Path) -> None:
    photo_path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(photo_path)

    details = read_photo_details(photo_path)

    assert details is not None
    assert (details.width_px, details.height_px) == (8, 8)
    assert details.camera is None
    assert details.lens is None


def test_zoom_crop_shows_one_photo_pixel_per_screen_pixel_inside_the_photo() -> None:
    centered_crop = compute_zoom_crop(
        image_width_px=6000, image_height_px=4000, view_width_px=1000, view_height_px=800, zoom_view=ZoomView()
    )
    corner_crop = compute_zoom_crop(
        image_width_px=6000,
        image_height_px=4000,
        view_width_px=1000,
        view_height_px=800,
        zoom_view=ZoomView(center_x_ratio=1.0, center_y_ratio=0.0),
    )
    small_photo_crop = compute_zoom_crop(
        image_width_px=640, image_height_px=480, view_width_px=1000, view_height_px=800, zoom_view=ZoomView()
    )

    assert (centered_crop.left_px, centered_crop.top_px, centered_crop.width_px, centered_crop.height_px) == (
        2500,
        1600,
        1000,
        800,
    )
    assert (corner_crop.right_px, corner_crop.top_px) == (6000, 0)
    assert (small_photo_crop.width_px, small_photo_crop.height_px) == (640, 480)


def test_zoom_pans_by_a_quarter_of_the_view_and_stops_at_the_edge() -> None:
    zoom_view = ZoomView()

    panned_view = zoom_view.pan(x_steps=1, y_steps=0, visible_width_ratio=0.2, visible_height_ratio=0.2)
    edge_view = zoom_view.pan(x_steps=100, y_steps=-100, visible_width_ratio=0.2, visible_height_ratio=0.2)

    assert panned_view.center_x_ratio == pytest.approx(0.55)
    assert (edge_view.center_x_ratio, edge_view.center_y_ratio) == pytest.approx((0.9, 0.1))
