"""Ratings in XMP, the format Lightroom, darktable, digiKam, Capture One, Bridge and cameras share.

``xmp:Rating`` is ``1``..``5`` for stars, ``0`` for unrated and ``-1`` for rejected. lazyrate can write it

* to a sidecar next to the photo (``DSC_1234.xmp``, as Lightroom and Capture One name it), or
* into JPEG files themselves (lossless: only the XMP block is replaced, the image data stays byte for byte);
  RAW, HEIC and other formats always get a sidecar, RAW files are never modified.

An existing sidecar (e.g. with Lightroom develop settings) is edited in place: only the rating changes.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import struct
import tempfile
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from lazyrate.ratings import REJECTED

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class SegmentSpan:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class RatingFileJob:
    relative_path: str
    rating: int | None
    mode: RatingFileMode


class RatingFileMode(StrEnum):
    OFF = "off"
    SIDECAR = "sidecar"
    EMBEDDED = "embedded"

    @property
    def description(self) -> str:
        return RATING_FILE_MODE_DESCRIPTIONS[self]


RATING_FILE_MODE_DESCRIPTIONS = {
    RatingFileMode.OFF: "only in lazyrate's database",
    RatingFileMode.SIDECAR: "also in XMP sidecars (DSC_1234.xmp)",
    RatingFileMode.EMBEDDED: "also inside JPEGs (others: XMP sidecars)",
}

XMP_REJECTED = -1
XMP_UNRATED = 0
XMP_NAMESPACE = "http://ns.adobe.com/xap/1.0/"
JPEG_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
JPEG_START = b"\xff\xd8"
JPEG_APP0 = 0xE0
JPEG_APP1 = 0xE1
JPEG_START_OF_SCAN = 0xDA
JPEG_MAX_SEGMENT_BYTES = 0xFFFF
JPEG_EXTENSIONS = frozenset({".jpg", ".jpeg"})
EMBEDDED_RATING_SCAN_BYTES = 256 * 1024
"""XMP blocks sit near the start of JPEG and RAW files."""
RATING_ATTRIBUTE_PATTERN = re.compile(
    r"""(?P<prefix>\b(?:xmp|xap):Rating\s*=\s*)(?P<quote>["'])(?P<value>-?\d+)(?P=quote)"""
)
RATING_ELEMENT_PATTERN = re.compile(
    r"(?P<prefix><(?:xmp|xap):Rating>\s*)(?P<value>-?\d+)(?P<suffix>\s*</(?:xmp|xap):Rating>)"
)
RATING_PATTERN_BYTES = re.compile(rb"""(?:xmp|xap):Rating(?:\s*=\s*["']|>\s*)(-?\d+)""")
DESCRIPTION_START_PATTERN = re.compile(r"<rdf:Description\b")
XMP_PACKET_TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="lazyrate">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="{rating}"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def to_xmp_rating(rating: int | None) -> int:
    if rating is None:
        return XMP_UNRATED
    return XMP_REJECTED if rating == REJECTED else rating


def from_xmp_rating(xmp_rating: int) -> int | None:
    """lazyrate rating of an ``xmp:Rating``; ``None`` for unrated and values outside -1..5."""
    if xmp_rating == XMP_REJECTED:
        return REJECTED
    if 1 <= xmp_rating <= 5:
        return xmp_rating
    return None


def find_sidecar_path(path: Path) -> Path:
    """The sidecar to read or write: an existing ``DSC_1234.xmp`` or darktable's ``DSC_1234.NEF.xmp``, else the
    first."""
    basename_sidecar = path.with_suffix(".xmp")
    if basename_sidecar.exists():
        return basename_sidecar
    full_name_sidecar = path.with_name(path.name + ".xmp")
    return full_name_sidecar if full_name_sidecar.exists() else basename_sidecar


def read_file_rating(path: Path) -> int | None:
    """Rating stored with the photo: its sidecar first, else the XMP block inside the file. ``None`` if neither has
    one (or the file cannot be read)."""
    sidecar_path = find_sidecar_path(path)
    try:
        if sidecar_path.exists():
            sidecar_rating = parse_xmp_rating(sidecar_path.read_text(encoding="utf-8", errors="replace"))
            if sidecar_rating is not None:
                return from_xmp_rating(sidecar_rating)
        with path.open("rb") as photo_file:
            head = photo_file.read(EMBEDDED_RATING_SCAN_BYTES)
    except OSError:
        return None
    match = RATING_PATTERN_BYTES.search(head)
    return None if match is None else from_xmp_rating(int(match.group(1)))


def parse_xmp_rating(xml: str) -> int | None:
    match = RATING_ATTRIBUTE_PATTERN.search(xml) or RATING_ELEMENT_PATTERN.search(xml)
    return None if match is None else int(match["value"])


def set_xmp_rating(xml: str, *, xmp_rating: int) -> str:
    """``xml`` with its rating replaced, or added to the first ``rdf:Description``; everything else unchanged."""
    if RATING_ATTRIBUTE_PATTERN.search(xml):
        return RATING_ATTRIBUTE_PATTERN.sub(rf"\g<prefix>\g<quote>{xmp_rating}\g<quote>", xml, count=1)
    if RATING_ELEMENT_PATTERN.search(xml):
        return RATING_ELEMENT_PATTERN.sub(rf"\g<prefix>{xmp_rating}\g<suffix>", xml, count=1)
    description_match = DESCRIPTION_START_PATTERN.search(xml)
    if description_match is None:
        return build_xmp_packet(xmp_rating=xmp_rating)
    namespace_declaration = "" if 'xmlns:xmp="' in xml else f' xmlns:xmp="{XMP_NAMESPACE}"'
    insert_position = description_match.end()
    return f'{xml[:insert_position]}{namespace_declaration} xmp:Rating="{xmp_rating}"{xml[insert_position:]}'


def build_xmp_packet(*, xmp_rating: int) -> str:
    return XMP_PACKET_TEMPLATE.format(rating=xmp_rating)


def write_sidecar_rating(path: Path, *, rating: int | None) -> Path | None:
    """Write the rating to the photo's sidecar (created if needed; not created just to say "unrated")."""
    sidecar_path = find_sidecar_path(path)
    xmp_rating = to_xmp_rating(rating)
    if sidecar_path.exists():
        xml = set_xmp_rating(sidecar_path.read_text(encoding="utf-8"), xmp_rating=xmp_rating)
    elif rating is None:
        return None
    else:
        xml = build_xmp_packet(xmp_rating=xmp_rating)
    replace_file_atomically(sidecar_path, xml.encode("utf-8"), keep_times_of=None)
    return sidecar_path


def write_embedded_jpeg_rating(path: Path, *, rating: int | None) -> int:
    """Put the rating into the JPEG's XMP block, losslessly; the modification time is kept. Returns the new size.

    Raises ``ValueError`` for files that are no JPEG or whose XMP block would grow too large.
    """
    data = path.read_bytes()
    updated_data = replace_jpeg_xmp(data, xmp_rating=to_xmp_rating(rating))
    replace_file_atomically(path, updated_data, keep_times_of=path)
    return len(updated_data)


def replace_jpeg_xmp(data: bytes, *, xmp_rating: int) -> bytes:
    """JPEG bytes with the XMP block's rating set (block added after the leading APP0 / APP1 if missing)."""
    if not data.startswith(JPEG_START):
        raise ValueError("Not a JPEG file")
    position = 2
    insert_position = 2
    xmp_segment: SegmentSpan | None = None
    while position + 4 <= len(data) and data[position] == 0xFF:
        marker = data[position + 1]
        if marker == JPEG_START_OF_SCAN:
            break
        segment_length = struct.unpack_from(">H", data, position + 2)[0]
        segment_end = position + 2 + segment_length
        payload = data[position + 4 : segment_end]
        if marker == JPEG_APP1 and payload.startswith(JPEG_XMP_HEADER):
            xmp_segment = SegmentSpan(start=position, end=segment_end)
        if marker in {JPEG_APP0, JPEG_APP1} and insert_position == position:
            insert_position = segment_end
        position = segment_end
    if xmp_segment is None:
        new_xml = build_xmp_packet(xmp_rating=xmp_rating)
        return data[:insert_position] + build_xmp_segment(new_xml) + data[insert_position:]
    old_xml = data[xmp_segment.start + 4 + len(JPEG_XMP_HEADER) : xmp_segment.end].decode("utf-8", errors="replace")
    new_segment = build_xmp_segment(set_xmp_rating(old_xml, xmp_rating=xmp_rating))
    return data[: xmp_segment.start] + new_segment + data[xmp_segment.end :]


def build_xmp_segment(xml: str) -> bytes:
    payload = JPEG_XMP_HEADER + xml.encode("utf-8")
    segment_length = len(payload) + 2
    if segment_length > JPEG_MAX_SEGMENT_BYTES:
        raise ValueError("XMP block too large for a JPEG segment")
    return bytes([0xFF, JPEG_APP1]) + struct.pack(">H", segment_length) + payload


def replace_file_atomically(path: Path, data: bytes, *, keep_times_of: Path | None) -> None:
    """Write to a temporary file next to ``path`` and rename it over ``path``: a crash never leaves half a file."""
    original_stat = keep_times_of.stat() if keep_times_of is not None else None
    file_descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(data)
        if path.exists():
            shutil.copymode(path, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    if original_stat is not None:
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))


def write_rating_file(path: Path, *, rating: int | None, mode: RatingFileMode) -> int | None:
    """Store the rating with the photo as ``mode`` says. Returns the photo's new size if it was rewritten."""
    if mode is RatingFileMode.OFF:
        return None
    if mode is RatingFileMode.EMBEDDED and path.suffix.lower() in JPEG_EXTENSIONS:
        return write_embedded_jpeg_rating(path, rating=rating)
    write_sidecar_rating(path, rating=rating)
    return None


class RatingFileWriter:
    """Writes ratings to files on one background thread, in the order they were given (rewriting a large JPEG on a
    NAS must never delay the next key press).

    ``on_resized(relative_path, size_bytes)`` reports photos whose size changed (embedded ratings), so caches keyed
    by size can follow; ``on_error(message)`` reports failures. Both are called on the writer thread.
    """

    def __init__(
        self,
        *,
        root: Path,
        on_resized: Callable[[str, int], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._root = root
        self._on_resized = on_resized
        self._on_error = on_error
        self._mode = RatingFileMode.OFF
        self._pending: queue.Queue[RatingFileJob] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="lazyrate-rating-files", daemon=True)
        self._thread.start()

    def __repr__(self) -> str:
        return f"RatingFileWriter(mode={self._mode.value}, pending={self._pending.qsize()})"

    @property
    def mode(self) -> RatingFileMode:
        return self._mode

    @property
    def pending_count(self) -> int:
        return self._pending.qsize()

    def set_mode(self, mode: RatingFileMode) -> None:
        self._mode = mode

    def write(self, *, relative_path: str, rating: int | None) -> None:
        if self._mode is not RatingFileMode.OFF:
            self._pending.put(RatingFileJob(relative_path=relative_path, rating=rating, mode=self._mode))

    def wait_until_written(self) -> None:
        """Block until every queued rating is written (tests, ``lazyrate`` commands before exiting)."""
        self._pending.join()

    def _run(self) -> None:
        while True:
            job = self._pending.get()
            try:
                self._write_one(job)
            finally:
                self._pending.task_done()

    def _write_one(self, job: RatingFileJob) -> None:
        try:
            new_size_bytes = write_rating_file(self._root / job.relative_path, rating=job.rating, mode=job.mode)
        except (OSError, ValueError) as error:
            self._on_error(f"Could not write the rating of {job.relative_path}: {error}")
            return
        if new_size_bytes is not None:
            self._on_resized(job.relative_path, new_size_bytes)
