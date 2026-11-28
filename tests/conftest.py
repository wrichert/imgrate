from __future__ import annotations

import io
import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import ExifTags, Image, ImageStat

from lazyrate.media import scan_media_files
from lazyrate.session import RatingSession
from lazyrate.store import RatingStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

PHOTO_RELATIVE_PATHS = (
    "2023/a.jpg",
    "2023/b.JPG",
    "2023/sub/c.png",
    "2024/d.jpg",
    "2024/e.heic",
)


@pytest.fixture
def photo_root(tmp_path: Path) -> Path:
    root = tmp_path / "photos"
    for index, relative_path in enumerate(PHOTO_RELATIVE_PATHS):
        target_path = root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        image_format = "PNG" if target_path.suffix.lower() in {".png", ".heic"} else "JPEG"
        Image.new("RGB", (64 + index, 48), color=(index * 40, 100, 200)).save(target_path, format=image_format)
    (root / "2024" / "clip.mp4").write_bytes(b"not a real video")
    (root / "notes.txt").write_text("ignored")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "x.jpg").write_bytes(b"")
    return root


@pytest.fixture
def store(tmp_path: Path) -> Iterator[RatingStore]:
    rating_store = RatingStore(tmp_path / "ratings.sqlite")
    yield rating_store
    rating_store.close()


@pytest.fixture
def session(photo_root: Path, store: RatingStore) -> RatingSession:
    return RatingSession(scan_result=scan_media_files(photo_root), store=store)


def save_pattern_variants(root: Path) -> None:
    """Two bursts of near-identical shots plus one unrelated image, all with distinct EXIF capture times."""
    sunrise = Image.radial_gradient("L").resize((640, 480)).convert("RGB")
    forest = Image.effect_noise((80, 60), 90).resize((640, 480), Image.Resampling.BICUBIC).convert("RGB")
    beach = Image.linear_gradient("L").rotate(90).resize((640, 480)).convert("RGB")
    variants = {
        "burst/sunrise_1.jpg": (sunrise, "2024:06:01 06:00:00"),
        "burst/sunrise_2.jpg": (sunrise.resize((320, 240)), "2024:06:01 06:00:02"),
        "burst/sunrise_3.jpg": (sunrise.point(lambda value: min(255, value + 12)), "2024:06:01 06:00:05"),
        "burst/forest_1.jpg": (forest, "2024:06:01 09:00:00"),
        "burst/forest_2.jpg": (forest.resize((600, 450)), "2024:06:01 09:01:00"),
        "burst/beach.jpg": (beach, "2024:06:01 12:00:00"),
        "later/sunrise_next_day.jpg": (sunrise, "2024:06:02 06:00:00"),
    }
    for relative_path, (image, captured_at) in variants.items():
        target_path = root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        exif = Image.Exif()
        exif[0x0132] = captured_at  # DateTime
        image.save(target_path, format="JPEG", quality=90, exif=exif)


@pytest.fixture
def burst_root(tmp_path: Path) -> Path:
    root = tmp_path / "bursts"
    save_pattern_variants(root)
    return root


class ColorEmbedder:
    """Deterministic stand-in for CLIP: mean and spread of each colour channel (no torch, no download).

    ``embed_text("white")`` points at bright, grey photos (like receipts); any other text at strongly red ones.
    """

    model_name = "test/color-statistics"

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        rows = [self._describe(image) for image in images]
        return np.array([row / np.linalg.norm(row) for row in rows], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        vector = np.array([1, 1, 1, 0, 0, 0] if "white" in text else [1, 0, 0, 0, 0, 0], dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def _describe(self, image: Image.Image) -> np.ndarray:
        statistics = ImageStat.Stat(image.convert("RGB"))
        return np.array([*statistics.mean, *statistics.stddev], dtype=np.float64) / 255 + 1e-3


JUNK_PHOTO_COUNT = 24
KEEPER_PHOTO_COUNT = 24


def save_junk_and_keeper_photos(root: Path) -> None:
    """Near-white "receipts" and colourful "photos" with some variation, as training data for the junk model."""
    rng = np.random.default_rng(seed=7)
    for index in range(JUNK_PHOTO_COUNT):
        gray_level = int(rng.integers(225, 255))
        pixels = np.clip(rng.normal(gray_level, 6, size=(48, 64, 3)), 0, 255).astype(np.uint8)
        target_path = root / "receipts" / f"r{index:02d}.jpg"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(target_path)
    for index in range(KEEPER_PHOTO_COUNT):
        base_color = rng.integers(20, 200, size=3)
        pixels = np.clip(rng.normal(base_color, 40, size=(48, 64, 3)), 0, 255).astype(np.uint8)
        target_path = root / "trips" / f"p{index:02d}.jpg"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(target_path)


@pytest.fixture
def mixed_root(tmp_path: Path) -> Path:
    root = tmp_path / "mixed"
    save_junk_and_keeper_photos(root)
    return root


@pytest.fixture
def color_embedder() -> ColorEmbedder:
    return ColorEmbedder()


GEOTAGGED_PHOTOS = {
    # relative path: (latitude_deg, longitude_deg, colour); None = no GPS data
    "madeira/funchal_1.jpg": (32.6657, -16.9255, (250, 250, 250)),
    "madeira/funchal_2.jpg": (32.6500, -16.9100, (30, 160, 40)),
    "madeira/north_coast.jpg": (32.8045, -16.8817, (200, 30, 30)),
    "lisbon/belem.jpg": (38.6916, -9.2160, (40, 90, 200)),
    "munich/marienplatz.jpg": (48.1374, 11.5755, (120, 120, 40)),
    "misc/no_gps.jpg": None,
    "misc/null_island.jpg": (0.0, 0.0, (10, 10, 10)),
}


def convert_to_degrees_minutes_seconds(value_deg: float) -> tuple[float, float, float]:
    absolute_deg = abs(value_deg)
    degrees = int(absolute_deg)
    minutes = int((absolute_deg - degrees) * 60)
    seconds = (absolute_deg - degrees - minutes / 60) * 3600
    return (float(degrees), float(minutes), round(seconds, 4))


def save_geotagged_photo(path: Path, *, latitude_deg: float, longitude_deg: float, color: tuple[int, int, int]) -> None:
    exif = Image.Exif()
    gps_tags = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps_tags[ExifTags.GPS.GPSLatitudeRef] = "N" if latitude_deg >= 0 else "S"
    gps_tags[ExifTags.GPS.GPSLatitude] = convert_to_degrees_minutes_seconds(latitude_deg)
    gps_tags[ExifTags.GPS.GPSLongitudeRef] = "E" if longitude_deg >= 0 else "W"
    gps_tags[ExifTags.GPS.GPSLongitude] = convert_to_degrees_minutes_seconds(longitude_deg)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=color).save(path, format="JPEG", exif=exif)


@pytest.fixture
def geo_root(tmp_path: Path) -> Path:
    root = tmp_path / "travels"
    for relative_path, gps_data in GEOTAGGED_PHOTOS.items():
        target_path = root / relative_path
        if gps_data is None:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), color=(90, 90, 90)).save(target_path, format="JPEG")
            continue
        latitude_deg, longitude_deg, color = gps_data
        save_geotagged_photo(target_path, latitude_deg=latitude_deg, longitude_deg=longitude_deg, color=color)
    return root


def save_photo_with_camera(path: Path, *, maker: str, model: str, lens: str | None) -> None:
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = maker
    exif[ExifTags.Base.Model] = model
    if lens is not None:
        exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.LensModel] = lens
    Image.new("RGB", (64, 48), (90, 120, 150)).save(path, exif=exif)


TIFF_ASCII = 2
TIFF_SHORT = 3
TIFF_LONG = 4
TIFF_RATIONAL = 5


class TiffBuilder:
    """Minimal little-endian TIFF writer for fake RAW files: directories, strings, rationals and embedded blobs."""

    def __init__(self) -> None:
        self.data = bytearray(b"II*\x00\x00\x00\x00\x00")

    def add_blob(self, blob: bytes) -> int:
        if len(self.data) % 2:
            self.data.append(0)
        offset = len(self.data)
        self.data.extend(blob)
        return offset

    def add_directory(self, entries: list[tuple[int, int, int, bytes]]) -> int:
        """``entries``: (tag, type, count, value bytes); returns the directory offset."""
        packed_entries = []
        for tag, value_type, count, value_bytes in sorted(entries):
            if len(value_bytes) <= 4:
                value_field = value_bytes.ljust(4, b"\x00")
            else:
                value_field = struct.pack("<I", self.add_blob(value_bytes))
            packed_entries.append(struct.pack("<HHI", tag, value_type, count) + value_field)
        directory = struct.pack("<H", len(packed_entries)) + b"".join(packed_entries) + struct.pack("<I", 0)
        return self.add_blob(directory)

    def finish(self, first_directory_offset: int) -> bytes:
        self.data[4:8] = struct.pack("<I", first_directory_offset)
        return bytes(self.data)


def ascii_entry(tag: int, text: str) -> tuple[int, int, int, bytes]:
    encoded = text.encode() + b"\x00"
    return (tag, TIFF_ASCII, len(encoded), encoded)


def long_entry(tag: int, value: int) -> tuple[int, int, int, bytes]:
    return (tag, TIFF_LONG, 1, struct.pack("<I", value))


def degrees_entry(tag: int, degrees: float) -> tuple[int, int, int, bytes]:
    dms = convert_to_degrees_minutes_seconds(abs(degrees))
    values = b"".join(struct.pack("<II", round(part * 1000), 1000) for part in dms)
    return (tag, TIFF_RATIONAL, 3, values)


def encode_jpeg(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def save_fake_raw(
    path: Path,
    *,
    preview_size: tuple[int, int] = (600, 400),
    color: tuple[int, int, int] = (40, 120, 200),
    orientation: int = 1,
    latitude_deg: float | None = None,
    longitude_deg: float | None = None,
) -> None:
    """A NEF-like file: IFD0 with camera and a small thumbnail, a sub-IFD with the large JPEG preview, EXIF and GPS
    directories."""
    builder = TiffBuilder()
    thumbnail_jpeg = encode_jpeg((160, 120), (0, 0, 0))
    preview_jpeg = encode_jpeg(preview_size, color)
    thumbnail_offset = builder.add_blob(thumbnail_jpeg)
    preview_offset = builder.add_blob(preview_jpeg)
    preview_directory = builder.add_directory(
        [long_entry(254, 0), long_entry(513, preview_offset), long_entry(514, len(preview_jpeg))]
    )
    exif_directory = builder.add_directory(
        [ascii_entry(36867, "2024:08:15 10:30:00"), ascii_entry(42036, "NIKKOR Z 24-70mm f/4 S")]
    )
    ifd0_entries = [
        ascii_entry(271, "NIKON CORPORATION"),
        ascii_entry(272, "NIKON Z 6_2"),
        (274, TIFF_SHORT, 1, struct.pack("<H", orientation)),
        (330, TIFF_LONG, 1, struct.pack("<I", preview_directory)),
        long_entry(513, thumbnail_offset),
        long_entry(514, len(thumbnail_jpeg)),
        long_entry(34665, exif_directory),
    ]
    if latitude_deg is not None and longitude_deg is not None:
        gps_directory = builder.add_directory(
            [
                ascii_entry(1, "N" if latitude_deg >= 0 else "S"),
                degrees_entry(2, latitude_deg),
                ascii_entry(3, "E" if longitude_deg >= 0 else "W"),
                degrees_entry(4, longitude_deg),
            ]
        )
        ifd0_entries.append(long_entry(34853, gps_directory))
    ifd0 = builder.add_directory(ifd0_entries)
    path.write_bytes(builder.finish(ifd0))
