from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from conftest import encode_jpeg, save_fake_raw
from PIL import Image

from lazyrate.ratings import REJECTED
from lazyrate.xmp import (
    RatingFileMode,
    RatingFileWriter,
    from_xmp_rating,
    read_file_rating,
    set_xmp_rating,
    to_xmp_rating,
    write_rating_file,
)

if TYPE_CHECKING:
    from pathlib import Path

LIGHTROOM_SIDECAR = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   xmp:Rating="2"
   crs:Exposure2012="+0.35">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""


def test_ratings_map_to_xmp_and_back() -> None:
    assert [to_xmp_rating(rating) for rating in (None, REJECTED, 1, 5)] == [0, -1, 1, 5]
    assert [from_xmp_rating(xmp_rating) for xmp_rating in (-1, 0, 3, 9)] == [REJECTED, None, 3, None]


def test_existing_sidecar_keeps_everything_but_the_rating(tmp_path: Path) -> None:
    raw_path = tmp_path / "DSC_0001.NEF"
    save_fake_raw(raw_path)
    sidecar_path = tmp_path / "DSC_0001.xmp"
    sidecar_path.write_text(LIGHTROOM_SIDECAR)

    write_rating_file(raw_path, rating=REJECTED, mode=RatingFileMode.SIDECAR)

    sidecar_text = sidecar_path.read_text()
    assert 'xmp:Rating="-1"' in sidecar_text
    assert 'crs:Exposure2012="+0.35"' in sidecar_text
    assert read_file_rating(raw_path) == REJECTED


def test_rating_is_added_to_a_description_without_one() -> None:
    xml = '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>'

    updated_xml = set_xmp_rating(xml, xmp_rating=4)

    assert 'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="4"' in updated_xml


def test_element_style_rating_is_replaced() -> None:
    assert set_xmp_rating("<xmp:Rating>2</xmp:Rating>", xmp_rating=5) == "<xmp:Rating>5</xmp:Rating>"


def test_embedded_rating_changes_only_the_xmp_block_and_keeps_the_modification_time(tmp_path: Path) -> None:
    jpeg_path = tmp_path / "photo.jpg"
    Image.new("RGB", (64, 48), (10, 200, 30)).save(jpeg_path, quality=90)
    os.utime(jpeg_path, (1_700_000_000, 1_700_000_000))
    original_pixels = Image.open(jpeg_path).tobytes()

    new_size_bytes = write_rating_file(jpeg_path, rating=4, mode=RatingFileMode.EMBEDDED)
    write_rating_file(jpeg_path, rating=2, mode=RatingFileMode.EMBEDDED)

    assert new_size_bytes is not None
    assert Image.open(jpeg_path).tobytes() == original_pixels
    assert jpeg_path.stat().st_mtime == 1_700_000_000
    assert jpeg_path.read_bytes().count(b"http://ns.adobe.com/xap/1.0/\x00") == 1
    assert read_file_rating(jpeg_path) == 2
    assert not (tmp_path / "photo.xmp").exists()


def test_embedded_mode_writes_sidecars_for_raw_files(tmp_path: Path) -> None:
    raw_path = tmp_path / "DSC_0002.NEF"
    save_fake_raw(raw_path)
    raw_bytes = raw_path.read_bytes()

    write_rating_file(raw_path, rating=5, mode=RatingFileMode.EMBEDDED)

    assert raw_path.read_bytes() == raw_bytes
    assert read_file_rating(raw_path) == 5


def test_unrated_photo_gets_no_new_sidecar(tmp_path: Path) -> None:
    jpeg_path = tmp_path / "photo.jpg"
    jpeg_path.write_bytes(encode_jpeg((8, 8), (0, 0, 0)))

    write_rating_file(jpeg_path, rating=None, mode=RatingFileMode.SIDECAR)

    assert not (tmp_path / "photo.xmp").exists()


def test_writer_reports_resized_photos_and_errors(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").write_bytes(encode_jpeg((8, 8), (0, 0, 0)))
    resized: list[tuple[str, int]] = []
    errors: list[str] = []
    writer = RatingFileWriter(
        root=tmp_path, on_resized=lambda path, size: resized.append((path, size)), on_error=errors.append
    )
    writer.set_mode(RatingFileMode.EMBEDDED)

    writer.write(relative_path="photo.jpg", rating=3)
    writer.write(relative_path="missing.jpg", rating=3)
    writer.wait_until_written()

    assert [path for path, _size in resized] == ["photo.jpg"]
    assert resized[0][1] == (tmp_path / "photo.jpg").stat().st_size
    assert len(errors) == 1
    assert "missing.jpg" in errors[0]


@pytest.mark.parametrize("sidecar_name", ["DSC_0003.xmp", "DSC_0003.NEF.xmp"])
def test_both_sidecar_naming_conventions_are_read(tmp_path: Path, sidecar_name: str) -> None:
    raw_path = tmp_path / "DSC_0003.NEF"
    save_fake_raw(raw_path)
    (tmp_path / sidecar_name).write_text(LIGHTROOM_SIDECAR)

    assert read_file_rating(raw_path) == 2
