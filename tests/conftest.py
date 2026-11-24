from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PIL import Image

from imgrate.media import scan_media_files
from imgrate.session import RatingSession
from imgrate.store import RatingStore

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
