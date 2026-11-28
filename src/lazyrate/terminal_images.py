"""Showing photos in the terminal without encoding them on the UI thread.

With the kitty graphics protocol (Ghostty, kitty, WezTerm), textual-image sends every image as a base64 PNG scaled
to the exact pixel size of its cells. For a full-screen photo that is about 140 ms of resizing and PNG encoding,
done while rendering, on the UI thread, each time a photo is shown. Here the preview workers prepare that data
ahead of time (prefetched neighbours included), and the renderer only writes it to the terminal.
"""

from __future__ import annotations

import base64
import io
import threading
import weakref
from typing import TYPE_CHECKING, ClassVar

from textual_image.renderable import Image as AutoRenderable
from textual_image.renderable.tgp import Image as TGPRenderable
from textual_image.renderable.tgp import _send_tgp_message
from textual_image.widget import Image as AutoImageWidget
from textual_image.widget._base import Image as BaseImageWidget

if TYPE_CHECKING:
    from PIL import Image

    from lazyrate.preview import DisplayTarget


class TerminalPayloadStore:
    """Base64 PNGs per image and pixel size, prepared by worker threads; forgotten with the image."""

    PNG_COMPRESS_LEVEL = 2
    """As textual-image: level 2 is about as small as the default and three times as fast."""

    def __init__(self) -> None:
        self._payloads_by_image_id: dict[int, dict[tuple[int, int], str]] = {}
        """Keyed by ``id(image)`` (PIL images are not hashable); an entry is dropped when its image is freed."""
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"TerminalPayloadStore(images={len(self._payloads_by_image_id)})"

    def prepare(self, image: Image.Image, display_target: DisplayTarget) -> None:
        """Encode ``image`` at the pixel size it gets when fitted into ``display_target``."""
        cell_box = display_target.fit(image)
        pixel_size = (cell_box.columns * display_target.cell_width_px, cell_box.rows * display_target.cell_height_px)
        self.find_or_encode(image, pixel_size=pixel_size)

    def find_or_encode(self, image: Image.Image, *, pixel_size: tuple[int, int]) -> str:
        image_id = id(image)
        with self._lock:
            payload = self._payloads_by_image_id.get(image_id, {}).get(pixel_size)
        if payload is not None:
            return payload
        payload = encode_terminal_payload(image, pixel_size=pixel_size, compress_level=self.PNG_COMPRESS_LEVEL)
        with self._lock:
            if image_id not in self._payloads_by_image_id:
                self._payloads_by_image_id[image_id] = {}
                weakref.finalize(image, self._forget, image_id)
            self._payloads_by_image_id[image_id][pixel_size] = payload
        return payload

    def _forget(self, image_id: int) -> None:
        with self._lock:
            self._payloads_by_image_id.pop(image_id, None)


def encode_terminal_payload(image: Image.Image, *, pixel_size: tuple[int, int], compress_level: int) -> str:
    """``image`` scaled to ``pixel_size`` as base64 PNG, the way textual-image sends it."""
    scaled_image = image if image.size == pixel_size else image.resize(pixel_size)
    png_buffer = io.BytesIO()
    scaled_image.save(png_buffer, format="png", compress_level=compress_level)
    return base64.b64encode(png_buffer.getvalue()).decode("ascii")


TERMINAL_PAYLOADS = TerminalPayloadStore()
"""Shared by the preview workers and the renderer, which textual-image constructs without our arguments."""


class PreparedTGPRenderable(TGPRenderable):
    """textual-image's kitty renderer, sending the payload the workers prepared (encoding only if they did not)."""

    TGP_CHUNK_SIZE: ClassVar[int] = 4096

    def __init__(self, image: Image.Image, width: int | str | None = None, height: int | str | None = None) -> None:
        super().__init__(image, width, height)
        self._source_image = image

    def _send_image_to_terminal(self, width: int, height: int) -> None:
        self.terminal_image_id = next(TGPRenderable._image_id_counter)
        image_data = TERMINAL_PAYLOADS.find_or_encode(self._source_image, pixel_size=(width, height))
        for chunk_start in range(0, len(image_data), self.TGP_CHUNK_SIZE):
            chunk_end = chunk_start + self.TGP_CHUNK_SIZE
            _send_tgp_message(
                i=self.terminal_image_id,
                m=1 if chunk_end < len(image_data) else 0,
                f=100,
                payload=image_data[chunk_start:chunk_end],
                q=2,
            )


class PreparedTGPImageWidget(BaseImageWidget, Renderable=PreparedTGPRenderable):
    pass


ImageWidget = BaseImageWidget
"""Common base of every image widget, for queries."""
IS_KITTY_GRAPHICS = AutoRenderable is TGPRenderable
PhotoWidget: type[BaseImageWidget] = PreparedTGPImageWidget if IS_KITTY_GRAPHICS else AutoImageWidget
"""The image widget to use: prepared kitty graphics where the terminal supports them, else textual-image's choice
(Sixel, half cells), which encodes while rendering as before."""
