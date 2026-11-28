"""Fast, cached, background-decoded image previews."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from lazyrate.media import IMAGE_EXTENSIONS

if TYPE_CHECKING:
    from pathlib import Path

register_heif_opener()


@dataclass(frozen=True, slots=True)
class Preview:
    """A downscaled, orientation-corrected image (or why there is none)."""

    image: Image.Image | None
    error: str | None


@dataclass(frozen=True, slots=True)
class CameraDetails:
    camera: str | None
    """``NIKON Z 6_2``, ``Apple iPhone 15 Pro``."""
    lens: str | None
    """``NIKKOR Z 24-70mm f/4 S``; for phones without the repeated model name (``back triple camera 6.86mm f/1.78``)."""


@dataclass(frozen=True, slots=True)
class PhotoDetails:
    """Metadata of the original file, read from its header without decoding the pixels."""

    width_px: int
    height_px: int
    captured_at: str | None
    camera: str | None
    lens: str | None


CAMERA_MAKER_SUFFIXES = ("corporation", "corp.", "corp", "co.,ltd.", "co., ltd.", "co.,ltd", "imaging", "company")
"""Dropped from the EXIF maker: ``NIKON CORPORATION`` -> ``NIKON``, ``OLYMPUS IMAGING CORP.`` -> ``OLYMPUS``."""


def load_preview(path: Path, *, max_edge_px: int) -> Preview:
    """Decode ``path`` downscaled to at most ``max_edge_px`` on its longest edge."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return Preview(image=None, error="no preview for this format — press o to open externally")
    try:
        with Image.open(path) as source_image:
            # JPEG: let libjpeg decode at 1/2, 1/4 or 1/8 scale — by far the biggest speedup.
            source_image.draft("RGB", (max_edge_px, max_edge_px))
            oriented_image = ImageOps.exif_transpose(source_image)
            oriented_image.thumbnail((max_edge_px, max_edge_px), Image.Resampling.BILINEAR)
            preview_image = oriented_image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError) as error:
        return Preview(image=None, error=f"cannot decode: {error}")
    return Preview(image=preview_image, error=None)


def read_photo_details(path: Path) -> PhotoDetails | None:
    """Size, capture date, camera and lens; ``None`` for videos and unreadable files. About a millisecond."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            camera_details = read_camera_details(exif)
            return PhotoDetails(
                width_px=image.width,
                height_px=image.height,
                captured_at=read_capture_date(exif),
                camera=camera_details.camera,
                lens=camera_details.lens,
            )
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        return None


def read_camera_details(exif: Image.Exif) -> CameraDetails:
    """Camera and lens names from EXIF, without the maker repeated: ``NIKON CORPORATION`` + ``NIKON Z 6_2`` gives
    ``NIKON Z 6_2``; ``Apple`` + ``iPhone 15 Pro`` gives ``Apple iPhone 15 Pro``."""
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    maker = shorten_camera_maker(read_exif_text(exif.get(ExifTags.Base.Make)))
    model = read_exif_text(exif.get(ExifTags.Base.Model))
    lens = read_exif_text(exif_ifd.get(ExifTags.Base.LensModel))
    if model and maker and not model.casefold().startswith(maker.casefold()):
        camera = f"{maker} {model}"
    else:
        camera = model or maker
    if lens and model and lens.casefold().startswith(model.casefold()) and len(lens) > len(model):
        lens = lens[len(model) :].strip()
    return CameraDetails(camera=camera, lens=lens)


def read_exif_text(value: object) -> str | None:
    """EXIF strings may be bytes and padded with NULs or spaces; empty means missing."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    text = value.strip("\x00 ")
    return text or None


def shorten_camera_maker(maker: str | None) -> str | None:
    if maker is None:
        return None
    for suffix in CAMERA_MAKER_SUFFIXES:
        if maker.casefold().endswith(" " + suffix):
            maker = maker[: -len(suffix) - 1].strip()
    return maker or None


def read_capture_date(exif: Image.Exif) -> str | None:
    captured_at = read_exif_text(exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.DateTimeOriginal)) or read_exif_text(
        exif.get(ExifTags.Base.DateTime)
    )
    return captured_at


class PreviewCache:
    """LRU cache of preview futures, decoded on a thread pool so prefetching never blocks the UI."""

    DEFAULT_CAPACITY = 48
    DEFAULT_WORKER_COUNT = 3

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CAPACITY,
        worker_count: int = DEFAULT_WORKER_COUNT,
    ) -> None:
        self._capacity = capacity
        self._executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="preview")
        self._futures: OrderedDict[tuple[Path, int], Future[Preview]] = OrderedDict()

    def __repr__(self) -> str:
        return f"PreviewCache(entries={len(self._futures)}, capacity={self._capacity})"

    def request(self, path: Path, *, max_edge_px: int) -> Future[Preview]:
        """Return a future for the preview, scheduling the decode if not yet cached."""
        key = (path, max_edge_px)
        future = self._futures.get(key)
        if future is not None and not future.cancelled():
            self._futures.move_to_end(key)
            return future
        future = self._executor.submit(load_preview, path, max_edge_px=max_edge_px)
        self._futures[key] = future
        self._evict_oldest()
        return future

    async def get(self, path: Path, *, max_edge_px: int) -> Preview:
        # Shield: cancelling the awaiting UI worker must not cancel the cached decode.
        return await asyncio.shield(asyncio.wrap_future(self.request(path, max_edge_px=max_edge_px)))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _evict_oldest(self) -> None:
        while len(self._futures) > self._capacity:
            _, evicted_future = self._futures.popitem(last=False)
            evicted_future.cancel()
