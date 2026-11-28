"""Discovery of media files below a root folder."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".webp",
        ".tif",
        ".tiff",
        ".gif",
        ".bmp",
        ".avif",
    }
)
"""Extensions that are enabled by default and can be previewed in the terminal."""

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".m2ts", ".3gp", ".webm", ".wmv", ".mpg", ".mpeg"}
)

RAW_EXTENSIONS: frozenset[str] = frozenset(
    {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".srw"}
)

PHOTO_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | RAW_EXTENSIONS
"""Everything with a picture lazyrate can show (RAW files through their embedded JPEG)."""

MEDIA_EXTENSIONS: frozenset[str] = PHOTO_EXTENSIONS | VIDEO_EXTENSIONS
"""Only files with these extensions are collected by the scanner."""

IGNORED_DIRECTORY_NAMES: frozenset[str] = frozenset({"@eaDir", "#recycle", "$RECYCLE.BIN", "__MACOSX"})


@dataclass(frozen=True, slots=True)
class MediaFile:
    """A media file, addressed relative to the scanned root."""

    relative_path: str
    extension: str
    size_bytes: int
    modified_timestamp_utc: float


@dataclass(frozen=True, slots=True)
class ScanResult:
    """All media files found below a root folder."""

    root: Path
    files: list[MediaFile]
    extension_counts: dict[str, int] = field(default_factory=dict)


def normalize_extension(extension: str) -> str:
    """Return the extension lower-cased and with a leading dot (``JPG`` -> ``.jpg``)."""
    stripped_extension = extension.strip().lower()
    if not stripped_extension:
        raise ValueError("Extension must not be empty")
    return stripped_extension if stripped_extension.startswith(".") else f".{stripped_extension}"


def build_path_sort_key(relative_path: str) -> tuple[str, str]:
    """Folder first, then name: keeps each folder contiguous (``a/x.jpg`` before ``a/sub/y.jpg`` … ``a/z.jpg``
    would otherwise interleave)."""
    folder, _, file_name = relative_path.lower().rpartition("/")
    return (folder, file_name)


def is_ignored_directory(name: str) -> bool:
    """Hidden folders and NAS/OS metadata folders are skipped."""
    return name.startswith(".") or name in IGNORED_DIRECTORY_NAMES


def scan_media_files(root: Path) -> ScanResult:
    """Recursively collect all media files below ``root``, sorted by relative path."""
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"Not a directory: {resolved_root}")

    files: list[MediaFile] = []
    for directory_path, directory_names, file_names in os.walk(resolved_root):
        directory_names[:] = sorted(name for name in directory_names if not is_ignored_directory(name))
        for file_name in file_names:
            media_file = _build_media_file(root=resolved_root, directory_path=Path(directory_path), file_name=file_name)
            if media_file is not None:
                files.append(media_file)

    files.sort(key=lambda media_file: build_path_sort_key(media_file.relative_path))
    extension_counts = dict(Counter(media_file.extension for media_file in files).most_common())
    return ScanResult(root=resolved_root, files=files, extension_counts=extension_counts)


def _build_media_file(*, root: Path, directory_path: Path, file_name: str) -> MediaFile | None:
    if file_name.startswith("."):
        return None
    extension = Path(file_name).suffix.lower()
    if extension not in MEDIA_EXTENSIONS:
        return None
    absolute_path = directory_path / file_name
    try:
        stat_result = absolute_path.stat()
    except OSError:
        return None
    return MediaFile(
        relative_path=absolute_path.relative_to(root).as_posix(),
        extension=extension,
        size_bytes=stat_result.st_size,
        modified_timestamp_utc=stat_result.st_mtime,
    )


def build_companion_key(relative_path: str) -> str:
    """Folder plus file name without extension, case-insensitive: ``a/DSC_1234.NEF`` and ``a/dsc_1234.jpg`` match."""
    folder, _, file_name = relative_path.rpartition("/")
    stem = file_name.rsplit(".", 1)[0]
    return f"{folder}/{stem}".casefold()


def find_raw_companions(files: list[MediaFile], *, enabled_extensions: set[str]) -> dict[str, list[MediaFile]]:
    """RAW files shot together with a JPEG / HEIC of the same name (RAW+JPEG): ``{image path: [raw files]}``.

    The image is shown and rated, the RAW follows it (same rating, exported with it). Only when both formats are
    enabled; a RAW without a twin is a photo of its own.
    """
    images_by_key: dict[str, MediaFile] = {}
    for media_file in files:
        if media_file.extension in IMAGE_EXTENSIONS and media_file.extension in enabled_extensions:
            images_by_key.setdefault(build_companion_key(media_file.relative_path), media_file)
    companions_by_image: defaultdict[str, list[MediaFile]] = defaultdict(list)
    for media_file in files:
        if media_file.extension not in RAW_EXTENSIONS or media_file.extension not in enabled_extensions:
            continue
        image_file = images_by_key.get(build_companion_key(media_file.relative_path))
        if image_file is not None:
            companions_by_image[image_file.relative_path].append(media_file)
    return dict(companions_by_image)
