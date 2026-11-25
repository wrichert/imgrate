"""Perceptual fingerprints (difference hash) and capture times, used for near-duplicate detection and date sorting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

if TYPE_CHECKING:
    from pathlib import Path

register_heif_opener()

HASH_WIDTH = 8
HASH_HEIGHT = 8
DRAFT_EDGE_PX = 128
EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """Fingerprint of one file, valid as long as size and modification time are unchanged.

    ``capture_timestamp`` is the EXIF capture time interpreted as UTC (EXIF has no zone); it is only
    used for ordering and for "taken within N minutes" comparisons, where a constant offset does not matter.
    ``difference_hash`` is ``None`` if the image could not be decoded.
    """

    relative_path: str
    size_bytes: int
    modified_timestamp_utc: float
    difference_hash: int | None
    capture_timestamp: float | None


@dataclass(frozen=True, slots=True)
class ImageSignature:
    difference_hash: int | None
    capture_timestamp: float | None


def compute_image_signature(path: Path) -> ImageSignature:
    """64-bit dHash: grayscale 9x8 thumbnail, one bit per "left pixel brighter than right pixel"."""
    try:
        with Image.open(path) as source_image:
            capture_timestamp = read_capture_timestamp(source_image)
            source_image.draft("L", (DRAFT_EDGE_PX, DRAFT_EDGE_PX))
            oriented_image = ImageOps.exif_transpose(source_image)
            small_image = oriented_image.convert("L").resize((HASH_WIDTH + 1, HASH_HEIGHT), Image.Resampling.LANCZOS)
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        return ImageSignature(difference_hash=None, capture_timestamp=None)
    pixels = small_image.tobytes()  # mode "L": one byte per pixel, row-major
    difference_hash = 0
    for row_index in range(HASH_HEIGHT):
        row_offset = row_index * (HASH_WIDTH + 1)
        for column_index in range(HASH_WIDTH):
            is_brighter = pixels[row_offset + column_index] > pixels[row_offset + column_index + 1]
            difference_hash = (difference_hash << 1) | int(is_brighter)
    return ImageSignature(difference_hash=difference_hash, capture_timestamp=capture_timestamp)


def read_capture_timestamp(image: Image.Image) -> float | None:
    exif = image.getexif()
    captured_at_text = exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.DateTimeOriginal) or exif.get(
        ExifTags.Base.DateTime
    )
    if not captured_at_text:
        return None
    try:
        captured_at = datetime.strptime(str(captured_at_text).strip("\x00 "), EXIF_DATETIME_FORMAT)
    except ValueError:
        return None
    return captured_at.replace(tzinfo=UTC).timestamp()


def count_differing_bits(first_hash: int, second_hash: int) -> int:
    return (first_hash ^ second_hash).bit_count()
