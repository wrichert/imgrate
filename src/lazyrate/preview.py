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
    """A downscaled, orientation-corrected image plus metadata of the original."""

    image: Image.Image | None
    original_width_px: int | None
    original_height_px: int | None
    captured_at: str | None
    error: str | None


def load_preview(path: Path, *, max_edge_px: int) -> Preview:
    """Decode ``path`` downscaled to at most ``max_edge_px`` on its longest edge."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return Preview(None, None, None, None, error="no preview for this format — press o to open externally")
    try:
        with Image.open(path) as source_image:
            original_width_px, original_height_px = source_image.size
            captured_at = _read_capture_date(source_image)
            # JPEG: let libjpeg decode at 1/2, 1/4 or 1/8 scale — by far the biggest speedup.
            source_image.draft("RGB", (max_edge_px, max_edge_px))
            oriented_image = ImageOps.exif_transpose(source_image)
            oriented_image.thumbnail((max_edge_px, max_edge_px), Image.Resampling.BILINEAR)
            preview_image = oriented_image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError) as error:
        return Preview(None, None, None, None, error=f"cannot decode: {error}")
    return Preview(
        image=preview_image,
        original_width_px=original_width_px,
        original_height_px=original_height_px,
        captured_at=captured_at,
        error=None,
    )


def _read_capture_date(image: Image.Image) -> str | None:
    exif = image.getexif()
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    captured_at = exif_ifd.get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
    return str(captured_at) if captured_at else None


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
