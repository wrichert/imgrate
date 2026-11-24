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
