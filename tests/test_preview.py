from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from conftest import save_photo_with_camera
from PIL import Image

from lazyrate.preview import read_photo_details

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
