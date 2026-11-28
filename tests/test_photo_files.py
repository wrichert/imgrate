from __future__ import annotations

import io
import struct
from typing import TYPE_CHECKING

import pytest
from conftest import encode_jpeg, save_fake_raw
from PIL import ExifTags, Image

from lazyrate.locations import read_gps_coordinates
from lazyrate.photo_files import PHOTO_READ_ERRORS, open_photo, orient_photo, read_photo_exif
from lazyrate.preview import load_preview, read_photo_details

if TYPE_CHECKING:
    from pathlib import Path


def test_raw_file_opens_as_its_largest_embedded_jpeg(tmp_path: Path) -> None:
    raw_path = tmp_path / "DSC_0001.NEF"
    save_fake_raw(raw_path, preview_size=(600, 400))

    with open_photo(raw_path) as image:
        size = image.size

    assert size == (600, 400)  # the preview, not the 160x120 thumbnail


def test_raw_preview_is_turned_upright_by_the_raw_orientation(tmp_path: Path) -> None:
    raw_path = tmp_path / "DSC_0002.NEF"
    save_fake_raw(raw_path, preview_size=(600, 400), orientation=6)  # camera held upright

    with open_photo(raw_path) as image:
        upright_size = orient_photo(image).size
    preview = load_preview(raw_path, max_edge_px=300)

    assert upright_size == (400, 600)
    assert preview.image is not None
    assert preview.image.size == (200, 300)


def test_raw_metadata_gives_camera_lens_date_and_gps(tmp_path: Path) -> None:
    raw_path = tmp_path / "DSC_0003.NEF"
    save_fake_raw(raw_path, latitude_deg=32.6657, longitude_deg=-16.9255)

    exif = read_photo_exif(raw_path)
    details = read_photo_details(raw_path)
    coordinates = read_gps_coordinates(raw_path)

    assert exif.get(ExifTags.Base.Make) == "NIKON CORPORATION"
    assert details is not None
    assert (details.camera, details.lens) == ("NIKON Z 6_2", "NIKKOR Z 24-70mm f/4 S")
    assert details.captured_at == "2024:08:15 10:30:00"
    assert (details.width_px, details.height_px) == (600, 400)
    assert coordinates is not None
    assert coordinates.latitude_deg == pytest.approx(32.6657, abs=1e-4)
    assert coordinates.longitude_deg == pytest.approx(-16.9255, abs=1e-4)


def test_fujifilm_raf_preview_is_found_through_its_header(tmp_path: Path) -> None:
    preview_jpeg = encode_jpeg((300, 200), (200, 50, 50))
    header = bytearray(b"FUJIFILMCCD-RAW 0201FF383501".ljust(100, b"\x00"))
    struct.pack_into(">II", header, 84, len(header), len(preview_jpeg))
    raf_path = tmp_path / "DSCF0001.RAF"
    raf_path.write_bytes(bytes(header) + preview_jpeg)

    with open_photo(raf_path) as image:
        size = image.size

    assert size == (300, 200)


def test_unreadable_raw_raises_a_photo_read_error(tmp_path: Path) -> None:
    broken_path = tmp_path / "broken.nef"
    broken_path.write_bytes(b"II*\x00" + b"\x00" * 20)

    with pytest.raises(PHOTO_READ_ERRORS):
        open_photo(broken_path)


def test_regular_images_are_opened_unchanged(tmp_path: Path) -> None:
    jpeg_path = tmp_path / "photo.jpg"
    Image.open(io.BytesIO(encode_jpeg((64, 48), (1, 2, 3)))).save(jpeg_path)

    with open_photo(jpeg_path) as image:
        size = image.size

    assert size == (64, 48)
