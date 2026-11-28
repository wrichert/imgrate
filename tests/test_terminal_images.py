from __future__ import annotations

import base64
import io
import threading
from typing import TYPE_CHECKING

from PIL import Image

from lazyrate import terminal_images
from lazyrate.preview import DisplayTarget, PreviewCache
from lazyrate.terminal_images import PreparedTGPRenderable, TerminalPayloadStore, encode_terminal_payload

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

CELL_WIDTH_PX = 10
CELL_HEIGHT_PX = 20


def build_display_target(*, max_columns: int = 40, max_rows: int = 10) -> DisplayTarget:
    return DisplayTarget(
        max_columns=max_columns, max_rows=max_rows, cell_width_px=CELL_WIDTH_PX, cell_height_px=CELL_HEIGHT_PX
    )


def test_payload_is_a_png_of_the_requested_size() -> None:
    image = Image.new("RGB", (400, 300), color=(10, 20, 30))

    payload = encode_terminal_payload(image, pixel_size=(200, 150), compress_level=2)

    with Image.open(io.BytesIO(base64.b64decode(payload))) as decoded_image:
        assert (decoded_image.format, decoded_image.size) == ("PNG", (200, 150))


def test_prepared_payload_is_reused_for_the_fitted_size() -> None:
    store = TerminalPayloadStore()
    image = Image.new("RGB", (400, 300))
    display_target = build_display_target()
    store.prepare(image, display_target)
    cell_box = display_target.fit(image)
    pixel_size = (cell_box.columns * CELL_WIDTH_PX, cell_box.rows * CELL_HEIGHT_PX)
    prepared_payload = store.find_or_encode(image, pixel_size=pixel_size)

    payload = store.find_or_encode(image, pixel_size=pixel_size)

    assert payload is prepared_payload


def test_payloads_are_forgotten_with_their_image() -> None:
    store = TerminalPayloadStore()
    image = Image.new("RGB", (40, 30))
    store.find_or_encode(image, pixel_size=(20, 15))

    del image

    assert repr(store) == "TerminalPayloadStore(images=0)"


def test_renderer_sends_the_prepared_payload_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.effect_noise((300, 200), 64).convert("RGB")  # noise: a payload of several chunks
    display_target = build_display_target()
    terminal_images.TERMINAL_PAYLOADS.prepare(image, display_target)
    cell_box = display_target.fit(image)
    pixel_size = (cell_box.columns * CELL_WIDTH_PX, cell_box.rows * CELL_HEIGHT_PX)
    sent_messages: list[dict[str, object]] = []
    monkeypatch.setattr(terminal_images, "_send_tgp_message", lambda **message: sent_messages.append(message))
    renderable = PreparedTGPRenderable(image, cell_box.columns, cell_box.rows)

    renderable._send_image_to_terminal(*pixel_size)

    sent_payload = "".join(str(message["payload"]) for message in sent_messages)
    assert sent_payload == terminal_images.TERMINAL_PAYLOADS.find_or_encode(image, pixel_size=pixel_size)
    assert len(sent_messages) > 1
    assert [message["m"] for message in sent_messages][-2:] == [1, 0]


def test_preview_cache_prepares_the_display_in_its_worker(tmp_path: Path) -> None:
    photo_path = tmp_path / "photo.jpg"
    Image.new("RGB", (400, 300), color=(200, 10, 10)).save(photo_path)
    prepared_thread_names: list[str] = []
    preview_cache = PreviewCache(
        prepare_display=lambda _image, _target: prepared_thread_names.append(threading.current_thread().name)
    )

    preview = preview_cache.request(photo_path, max_edge_px=200, display_target=build_display_target()).result()
    preview_cache.shutdown()

    assert preview.image is not None
    assert len(prepared_thread_names) == 1
    assert prepared_thread_names[0].startswith("preview")
