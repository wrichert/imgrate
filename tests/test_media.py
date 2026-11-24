from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from imgrate.media import normalize_extension, scan_media_files

if TYPE_CHECKING:
    from pathlib import Path


def test_scan_collects_media_recursively_and_skips_hidden_and_non_media(photo_root: Path) -> None:
    scan_result = scan_media_files(photo_root)

    assert [media_file.relative_path for media_file in scan_result.files] == [
        "2023/a.jpg",
        "2023/b.JPG",
        "2023/sub/c.png",
        "2024/clip.mp4",
        "2024/d.jpg",
        "2024/e.heic",
    ]
    assert scan_result.extension_counts == {".jpg": 3, ".png": 1, ".mp4": 1, ".heic": 1}


def test_scan_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a directory"):
        scan_media_files(tmp_path / "missing")


@pytest.mark.parametrize(("raw", "expected"), [("JPG", ".jpg"), (".Png", ".png"), (" mp4 ", ".mp4")])
def test_normalize_extension(raw: str, expected: str) -> None:
    assert normalize_extension(raw) == expected
