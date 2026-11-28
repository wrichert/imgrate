"""Fast, cached, background-decoded image previews."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

from PIL import ExifTags, Image

from lazyrate.media import PHOTO_EXTENSIONS
from lazyrate.photo_files import PHOTO_READ_ERRORS, open_photo, orient_photo, read_photo_exif

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Preview:
    """A downscaled, orientation-corrected image (or why there is none)."""

    image: Image.Image | None
    error: str | None


@dataclass(frozen=True, slots=True)
class CellBox:
    """A size in terminal cells."""

    columns: int
    rows: int


def fit_image_to_cells(
    *,
    image_width_px: int,
    image_height_px: int,
    max_columns: int,
    max_rows: int,
    cell_width_px: int,
    cell_height_px: int,
) -> CellBox:
    """Largest cell box with the image's aspect ratio that fits into ``max_columns`` x ``max_rows``."""
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError(f"Invalid image size {image_width_px}x{image_height_px}")
    scale_factor = min(
        max_columns * cell_width_px / image_width_px,
        max_rows * cell_height_px / image_height_px,
    )
    return CellBox(
        columns=max(1, min(max_columns, math.floor(image_width_px * scale_factor / cell_width_px))),
        rows=max(1, min(max_rows, math.floor(image_height_px * scale_factor / cell_height_px))),
    )


@dataclass(frozen=True, slots=True)
class DisplayTarget:
    """Where a preview will be shown: the largest box available (cells) and the terminal's cell size."""

    max_columns: int
    max_rows: int
    cell_width_px: int
    cell_height_px: int

    def fit(self, image: Image.Image) -> CellBox:
        return fit_image_to_cells(
            image_width_px=image.width,
            image_height_px=image.height,
            max_columns=self.max_columns,
            max_rows=self.max_rows,
            cell_width_px=self.cell_width_px,
            cell_height_px=self.cell_height_px,
        )


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
    if path.suffix.lower() not in PHOTO_EXTENSIONS:
        return Preview(image=None, error="no preview for this format — press o to open externally")
    try:
        with open_photo(path) as source_image:
            # JPEG: let libjpeg decode at 1/2, 1/4 or 1/8 scale — by far the biggest speedup.
            source_image.draft("RGB", (max_edge_px, max_edge_px))
            oriented_image = orient_photo(source_image)
            oriented_image.thumbnail((max_edge_px, max_edge_px), Image.Resampling.BILINEAR)
            preview_image = oriented_image.convert("RGB")
    except PHOTO_READ_ERRORS as error:
        return Preview(image=None, error=f"cannot decode: {error}")
    return Preview(image=preview_image, error=None)


def load_full_resolution(path: Path) -> Image.Image | None:
    """The whole photo at full size, upright (for checking focus at 100%); ``None`` if it cannot be decoded."""
    if path.suffix.lower() not in PHOTO_EXTENSIONS:
        return None
    try:
        with open_photo(path) as source_image:
            return orient_photo(source_image).convert("RGB")
    except PHOTO_READ_ERRORS:
        return None


@dataclass(frozen=True, slots=True)
class CropBox:
    left_px: int
    top_px: int
    right_px: int
    bottom_px: int

    @property
    def width_px(self) -> int:
        return self.right_px - self.left_px

    @property
    def height_px(self) -> int:
        return self.bottom_px - self.top_px

    def as_pillow_box(self) -> tuple[int, int, int, int]:
        return (self.left_px, self.top_px, self.right_px, self.bottom_px)


@dataclass(frozen=True, slots=True)
class ZoomView:
    """100% view: which part of the photo is shown, as the center in shares of its width and height.

    Kept when moving to the next photo, so the same spot of a burst can be compared shot by shot.
    """

    PAN_STEP_RATIO: ClassVar[float] = 0.25
    """One pan step moves by this share of the visible area."""

    center_x_ratio: float = 0.5
    center_y_ratio: float = 0.5

    def pan(self, *, x_steps: int, y_steps: int, visible_width_ratio: float, visible_height_ratio: float) -> ZoomView:
        """Move by whole steps; ``visible_*_ratio`` is the share of the photo on screen (keeps the crop inside)."""
        return replace(
            self,
            center_x_ratio=clamp_center(
                self.center_x_ratio + x_steps * self.PAN_STEP_RATIO * visible_width_ratio,
                visible_ratio=visible_width_ratio,
            ),
            center_y_ratio=clamp_center(
                self.center_y_ratio + y_steps * self.PAN_STEP_RATIO * visible_height_ratio,
                visible_ratio=visible_height_ratio,
            ),
        )


def clamp_center(center_ratio: float, *, visible_ratio: float) -> float:
    half_visible_ratio = min(visible_ratio, 1.0) / 2
    return min(max(center_ratio, half_visible_ratio), 1.0 - half_visible_ratio)


def compute_zoom_crop(
    *, image_width_px: int, image_height_px: int, view_width_px: int, view_height_px: int, zoom_view: ZoomView
) -> CropBox:
    """The part of the photo that fills the view at one photo pixel per screen pixel, kept inside the photo."""
    crop_width_px = min(image_width_px, view_width_px)
    crop_height_px = min(image_height_px, view_height_px)
    left_px = round(zoom_view.center_x_ratio * image_width_px - crop_width_px / 2)
    top_px = round(zoom_view.center_y_ratio * image_height_px - crop_height_px / 2)
    left_px = min(max(left_px, 0), image_width_px - crop_width_px)
    top_px = min(max(top_px, 0), image_height_px - crop_height_px)
    return CropBox(left_px=left_px, top_px=top_px, right_px=left_px + crop_width_px, bottom_px=top_px + crop_height_px)


def read_photo_details(path: Path) -> PhotoDetails | None:
    """Size, capture date, camera and lens; ``None`` for videos and unreadable files. About a millisecond."""
    if path.suffix.lower() not in PHOTO_EXTENSIONS:
        return None
    try:
        exif = read_photo_exif(path)
        with open_photo(path) as image:
            width_px, height_px = image.size
    except PHOTO_READ_ERRORS:
        return None
    camera_details = read_camera_details(exif)
    return PhotoDetails(
        width_px=width_px,
        height_px=height_px,
        captured_at=read_capture_date(exif),
        camera=camera_details.camera,
        lens=camera_details.lens,
    )


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
    """LRU cache of preview futures, decoded on a thread pool so prefetching never blocks the UI.

    With a ``display_target``, the worker also runs ``prepare_display`` (encoding the image for the terminal at the
    size it will be shown), so showing a prefetched photo costs the UI thread almost nothing.
    """

    DEFAULT_CAPACITY = 48
    DEFAULT_WORKER_COUNT = 3

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CAPACITY,
        worker_count: int = DEFAULT_WORKER_COUNT,
        prepare_display: Callable[[Image.Image, DisplayTarget], None] | None = None,
    ) -> None:
        self._capacity = capacity
        self._prepare_display = prepare_display
        self._executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="preview")
        self._futures: OrderedDict[tuple[Path, int, DisplayTarget | None], Future[Preview]] = OrderedDict()

    def __repr__(self) -> str:
        return f"PreviewCache(entries={len(self._futures)}, capacity={self._capacity})"

    def request(self, path: Path, *, max_edge_px: int, display_target: DisplayTarget | None = None) -> Future[Preview]:
        """Return a future for the preview, scheduling the decode if not yet cached."""
        key = (path, max_edge_px, display_target)
        future = self._futures.get(key)
        if future is not None and not future.cancelled():
            self._futures.move_to_end(key)
            return future
        future = self._executor.submit(self._load, path, max_edge_px=max_edge_px, display_target=display_target)
        self._futures[key] = future
        self._evict_oldest()
        return future

    async def get(self, path: Path, *, max_edge_px: int, display_target: DisplayTarget | None = None) -> Preview:
        # Shield: cancelling the awaiting UI worker must not cancel the cached decode.
        future = self.request(path, max_edge_px=max_edge_px, display_target=display_target)
        return await asyncio.shield(asyncio.wrap_future(future))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _evict_oldest(self) -> None:
        while len(self._futures) > self._capacity:
            _, evicted_future = self._futures.popitem(last=False)
            evicted_future.cancel()

    def _load(self, path: Path, *, max_edge_px: int, display_target: DisplayTarget | None) -> Preview:
        """Worker thread."""
        preview = load_preview(path, max_edge_px=max_edge_px)
        if preview.image is not None and display_target is not None and self._prepare_display is not None:
            self._prepare_display(preview.image, display_target)
        return preview
