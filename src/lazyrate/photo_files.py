"""Opening photos of every supported kind: regular images through Pillow, RAW files through their embedded JPEG.

Every RAW format stores a JPEG preview next to the sensor data, for Nikon, Canon, Sony and Fujifilm usually at (or
close to) full size. Reading it is fast and needs no RAW decoder; it shows the photo as the camera rendered it, which
is what rating needs. Metadata (camera, lens, date, GPS) comes from the RAW file's own TIFF structure.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

from PIL import ExifTags, Image, ImageOps, TiffImagePlugin, UnidentifiedImageError
from pillow_heif import register_heif_opener

from lazyrate.media import RAW_EXTENSIONS

if TYPE_CHECKING:
    from pathlib import Path

register_heif_opener()

PHOTO_READ_ERRORS = (
    OSError,
    UnidentifiedImageError,
    ValueError,
    SyntaxError,
    struct.error,
    Image.DecompressionBombError,
)
"""What reading a broken, truncated or unsupported file can raise (Pillow's TIFF parser raises ``SyntaxError``)."""

RAW_ORIENTATION_INFO_KEY = "lazyrate_raw_orientation"
"""``Image.info`` key with the RAW file's EXIF orientation, for previews whose JPEG carries none."""
RAW_METADATA_READ_BYTES = 512 * 1024
"""The TIFF directories with camera, lens, date and GPS sit at the start of a RAW file."""
JPEG_SCAN_READ_BYTES = 64 * 1024
"""Enough to find the frame header (size) of an embedded JPEG."""
MAX_TIFF_DIRECTORIES = 32
TIFF_PREFIXES = (b"II*\x00", b"MM\x00*")
TIFF_LIKE_PREFIXES = {
    b"IIRO": b"II*\x00",  # Olympus ORF
    b"IIRS": b"II*\x00",  # Olympus ORF (older)
    b"MMOR": b"MM\x00*",  # Olympus ORF (big-endian)
    b"IIU\x00": b"II*\x00",  # Panasonic RW2
}
FUJIFILM_PREFIX = b"FUJIFILMCCD-RAW"
FUJIFILM_JPEG_OFFSET_POSITION = 84
CANON_CR3_PREVIEW_BOX = b"PRVW"
JPEG_START = b"\xff\xd8"
JPEG_FRAME_MARKERS = frozenset({0xC0, 0xC1, 0xC2})
"""Baseline, extended and progressive JPEG; lossless JPEG (``0xC3``, DNG sensor data) cannot be shown."""
TIFF_TAG_SUBFILE_TYPE = 254
TIFF_TAG_COMPRESSION = 259
TIFF_TAG_STRIP_OFFSETS = 273
TIFF_TAG_STRIP_BYTE_COUNTS = 279
TIFF_TAG_SUB_IFDS = 330
TIFF_TAG_JPEG_OFFSET = 513
TIFF_TAG_JPEG_LENGTH = 514
TIFF_JPEG_COMPRESSIONS = frozenset({6, 7})
ORIENTATION_TRANSPOSES = {
    2: (Image.Transpose.FLIP_LEFT_RIGHT,),
    3: (Image.Transpose.ROTATE_180,),
    4: (Image.Transpose.FLIP_TOP_BOTTOM,),
    5: (Image.Transpose.TRANSPOSE,),
    6: (Image.Transpose.ROTATE_270,),
    7: (Image.Transpose.TRANSVERSE,),
    8: (Image.Transpose.ROTATE_90,),
}
"""EXIF orientation -> the transposition that shows the photo upright (as ``ImageOps.exif_transpose``)."""


@dataclass(frozen=True, slots=True)
class JpegLocation:
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class EmbeddedJpeg:
    offset: int
    length: int
    width_px: int
    height_px: int

    @property
    def pixel_count(self) -> int:
        return self.width_px * self.height_px


def is_raw_file(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def open_photo(path: Path) -> Image.Image:
    """Lazily opened image (use it in ``with``); a RAW file opens as its embedded JPEG.

    Raises one of :data:`PHOTO_READ_ERRORS` if the file cannot be read.
    """
    if not is_raw_file(path):
        return Image.open(path)
    preview_image = Image.open(io.BytesIO(read_raw_preview_bytes(path)))
    orientation = read_photo_exif(path).get(ExifTags.Base.Orientation)
    if isinstance(orientation, int):
        preview_image.info[RAW_ORIENTATION_INFO_KEY] = orientation
    return preview_image


def orient_photo(image: Image.Image) -> Image.Image:
    """Upright copy: the image's own EXIF orientation, else (RAW previews) the RAW file's."""
    own_orientation = image.getexif().get(ExifTags.Base.Orientation)
    raw_orientation = image.info.get(RAW_ORIENTATION_INFO_KEY)
    if own_orientation is not None or raw_orientation is None:
        return ImageOps.exif_transpose(image)
    oriented_image = image.copy()
    for transpose in ORIENTATION_TRANSPOSES.get(raw_orientation, ()):
        oriented_image = oriented_image.transpose(transpose)
    return oriented_image


def read_photo_exif(path: Path) -> Image.Exif:
    """EXIF of any photo; empty if it has none. Raises one of :data:`PHOTO_READ_ERRORS` if unreadable."""
    if not is_raw_file(path):
        with Image.open(path) as image:
            return image.getexif()
    with path.open("rb") as raw_file:
        head = raw_file.read(RAW_METADATA_READ_BYTES)
    tiff_head = normalize_tiff_prefix(head)
    if tiff_head is not None:
        exif = Image.Exif()
        exif.load(tiff_head)
        return exif
    # Fujifilm and Canon CR3 are no TIFF files; their embedded JPEG carries the EXIF.
    with Image.open(io.BytesIO(read_raw_preview_bytes(path))) as preview_image:
        return preview_image.getexif()


def normalize_tiff_prefix(head: bytes) -> bytes | None:
    """``head`` with Olympus / Panasonic magic numbers replaced by the TIFF one, or ``None`` if it is no TIFF."""
    prefix = head[:4]
    if prefix in TIFF_PREFIXES:
        return head
    if prefix in TIFF_LIKE_PREFIXES:
        return TIFF_LIKE_PREFIXES[prefix] + head[4:]
    return None


def read_raw_preview_bytes(path: Path) -> bytes:
    """The largest JPEG embedded in a RAW file. Raises ``OSError`` if there is none."""
    with path.open("rb") as raw_file:
        head = raw_file.read(RAW_METADATA_READ_BYTES)
        if head.startswith(FUJIFILM_PREFIX):
            offset, length = struct.unpack_from(">II", head, FUJIFILM_JPEG_OFFSET_POSITION)
            return read_exactly(raw_file, offset=offset, length=length)
        if normalize_tiff_prefix(head) is None:
            return find_canon_cr3_preview(head)
        embedded_jpegs = [
            embedded_jpeg
            for location in find_tiff_jpeg_locations(raw_file, head=head)
            if (embedded_jpeg := describe_embedded_jpeg(raw_file, location=location)) is not None
        ]
        if not embedded_jpegs:
            raise OSError(f"No embedded JPEG preview in {path.name}")
        largest_jpeg = max(embedded_jpegs, key=lambda embedded_jpeg: embedded_jpeg.pixel_count)
        return read_exactly(raw_file, offset=largest_jpeg.offset, length=largest_jpeg.length)


def find_tiff_jpeg_locations(raw_file: BinaryIO, *, head: bytes) -> list[JpegLocation]:
    """Every JPEG referenced by the TIFF directories (IFD chain and sub-IFDs)."""
    tiff_header = (normalize_tiff_prefix(head) or head)[:8]
    first_directory = TiffImagePlugin.ImageFileDirectory_v2(tiff_header)
    pending_offsets = [first_directory.next]
    visited_offsets: set[int] = set()
    locations: list[JpegLocation] = []
    while pending_offsets and len(visited_offsets) < MAX_TIFF_DIRECTORIES:
        directory_offset = pending_offsets.pop()
        if directory_offset == 0 or directory_offset in visited_offsets:
            continue
        visited_offsets.add(directory_offset)
        directory = TiffImagePlugin.ImageFileDirectory_v2(tiff_header)
        raw_file.seek(directory_offset)
        directory.load(raw_file)
        pending_offsets.append(directory.next)
        pending_offsets.extend(as_integer_list(directory.get(TIFF_TAG_SUB_IFDS)))
        locations.extend(list_directory_jpegs(directory))
    return locations


def list_directory_jpegs(directory: TiffImagePlugin.ImageFileDirectory_v2) -> list[JpegLocation]:
    jpeg_locations: list[JpegLocation] = []
    jpeg_offsets = as_integer_list(directory.get(TIFF_TAG_JPEG_OFFSET))
    jpeg_lengths = as_integer_list(directory.get(TIFF_TAG_JPEG_LENGTH))
    if jpeg_offsets and jpeg_lengths:
        jpeg_locations.append(JpegLocation(offset=jpeg_offsets[0], length=jpeg_lengths[0]))
    strip_offsets = as_integer_list(directory.get(TIFF_TAG_STRIP_OFFSETS))
    strip_byte_counts = as_integer_list(directory.get(TIFF_TAG_STRIP_BYTE_COUNTS))
    is_single_jpeg_strip = len(strip_offsets) == 1 and len(strip_byte_counts) == 1
    if directory.get(TIFF_TAG_COMPRESSION) in TIFF_JPEG_COMPRESSIONS and is_single_jpeg_strip:
        jpeg_locations.append(JpegLocation(offset=strip_offsets[0], length=strip_byte_counts[0]))
    return jpeg_locations


def as_integer_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, tuple | list):
        return [item for item in value if isinstance(item, int)]
    return []


def describe_embedded_jpeg(raw_file: BinaryIO, *, location: JpegLocation) -> EmbeddedJpeg | None:
    """Size of the JPEG at ``location``; ``None`` if it is none, or one Pillow cannot decode (lossless JPEG)."""
    if location.length <= 0:
        return None
    raw_file.seek(location.offset)
    jpeg_head = raw_file.read(min(location.length, JPEG_SCAN_READ_BYTES))
    if not jpeg_head.startswith(JPEG_START):
        return None
    position = 2
    while position + 9 <= len(jpeg_head):
        if jpeg_head[position] != 0xFF:
            return None
        marker = jpeg_head[position + 1]
        segment_length = struct.unpack_from(">H", jpeg_head, position + 2)[0]
        if marker in JPEG_FRAME_MARKERS:
            height_px, width_px = struct.unpack_from(">HH", jpeg_head, position + 5)
            return EmbeddedJpeg(offset=location.offset, length=location.length, width_px=width_px, height_px=height_px)
        if 0xC3 <= marker <= 0xCF and marker not in {0xC4, 0xC8, 0xCC}:
            return None  # another frame type (lossless, hierarchical, arithmetic)
        position += 2 + segment_length
    return None


def find_canon_cr3_preview(head: bytes) -> bytes:
    """Canon CR3 (ISO media file): the ``PRVW`` box holds a 1620 px JPEG."""
    box_position = head.find(CANON_CR3_PREVIEW_BOX)
    if box_position < 4:
        raise OSError("No embedded JPEG preview found")
    box_size = struct.unpack_from(">I", head, box_position - 4)[0]
    jpeg_start = head.find(JPEG_START, box_position)
    box_end = box_position - 4 + box_size
    if jpeg_start < 0 or box_end > len(head):
        raise OSError("No embedded JPEG preview found")
    return head[jpeg_start:box_end]


def read_exactly(raw_file: BinaryIO, *, offset: int, length: int) -> bytes:
    raw_file.seek(offset)
    data = raw_file.read(length)
    if len(data) != length:
        raise OSError("Truncated embedded JPEG preview")
    return data
