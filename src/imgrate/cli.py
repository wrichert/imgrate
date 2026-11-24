"""Command line entry point: ``imgrate [ROOT]``, ``imgrate stats ROOT``, ``imgrate export ROOT DEST``."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from imgrate.media import scan_media_files
from imgrate.ratings import ALL_BUCKETS, MAX_STARS, REJECTED, describe_bucket
from imgrate.store import RatingStore

SUBCOMMANDS = frozenset({"rate", "stats", "export"})


@dataclass(frozen=True, slots=True)
class ExportSummary:
    exported_count: int
    missing_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imgrate", description="Rate all images in a folder tree, fast.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rate_parser = subparsers.add_parser("rate", help="open the rating TUI (default command)")
    rate_parser.add_argument("root", nargs="?", default=".", type=Path, help="photo folder (default: .)")
    _add_database_argument(rate_parser)

    stats_parser = subparsers.add_parser("stats", help="print rating counts")
    stats_parser.add_argument("root", type=Path)
    _add_database_argument(stats_parser)

    export_parser = subparsers.add_parser("export", help="copy or link rated files to a folder")
    export_parser.add_argument("root", type=Path)
    export_parser.add_argument("destination", type=Path)
    rating_selection = export_parser.add_mutually_exclusive_group()
    rating_selection.add_argument("--min-rating", type=int, default=None, help="minimum stars (default: 5)")
    rating_selection.add_argument(
        "--rating",
        type=int,
        action="append",
        dest="ratings",
        help="exactly this rating, repeatable (0 = rejected), e.g. --rating 3 --rating 4",
    )
    export_parser.add_argument("--symlink", action="store_true", help="create symlinks instead of copies")
    export_parser.add_argument("--flat", action="store_true", help="do not recreate the sub folder structure")
    export_parser.add_argument("--dry-run", action="store_true", help="only print what would be exported")
    _add_database_argument(export_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    # `imgrate` and `imgrate ~/Photos` are shorthands for `imgrate rate ...`.
    if not arguments or (arguments[0] not in SUBCOMMANDS and arguments[0] not in {"-h", "--help"}):
        arguments = ["rate", *arguments]
    parsed = build_parser().parse_args(arguments)
    root = parsed.root.expanduser().resolve()
    if not root.is_dir():
        print(f"imgrate: not a directory: {root}", file=sys.stderr)
        return 2
    database_path = parsed.db or root / RatingStore.DEFAULT_FILE_NAME

    if parsed.command == "rate":
        return run_tui(root=root, database_path=database_path)
    store = RatingStore(database_path)
    try:
        if parsed.command == "stats":
            print_stats(store)
            return 0
        selected_ratings = select_export_ratings(min_rating=parsed.min_rating, exact_ratings=parsed.ratings)
        if not selected_ratings:
            print(
                f"imgrate: --rating must be {REJECTED}..{MAX_STARS}, --min-rating must be 1..{MAX_STARS}",
                file=sys.stderr,
            )
            return 2
        summary = export_rated_files(
            store=store,
            root=root,
            destination=parsed.destination.expanduser().resolve(),
            ratings=selected_ratings,
            is_symlink=parsed.symlink,
            is_flat=parsed.flat,
            is_dry_run=parsed.dry_run,
        )
        print(f"exported {summary.exported_count} files, {summary.missing_count} missing on disk")
        return 0
    finally:
        store.close()


def run_tui(*, root: Path, database_path: Path) -> int:
    # Imported lazily: textual_image probes the terminal on import, which is not wanted for stats/export.
    from imgrate.app import ImageRaterApp
    from imgrate.session import RatingSession

    print(f"Scanning {root} …", file=sys.stderr)
    scan_result = scan_media_files(root)
    store = RatingStore(database_path)
    try:
        session = RatingSession(scan_result=scan_result, store=store)
        ImageRaterApp(session=session, database_path=database_path).run()
    finally:
        store.close()
    return 0


def print_stats(store: RatingStore) -> None:
    ratings = store.load_ratings()
    for bucket in reversed(ALL_BUCKETS[1:]):
        count = sum(1 for rating in ratings.values() if rating == bucket)
        print(f"{describe_bucket(bucket):<9} {count:>7,}")
    print(f"{'rated':<9} {sum(1 for rating in ratings.values() if rating != REJECTED):>7,}")


def export_rated_files(
    *,
    store: RatingStore,
    root: Path,
    destination: Path,
    ratings: frozenset[int],
    is_symlink: bool,
    is_flat: bool,
    is_dry_run: bool,
) -> ExportSummary:
    selected_paths = sorted(path for path, rating in store.load_ratings().items() if rating in ratings)
    exported_count = 0
    missing_count = 0
    used_flat_names: set[str] = set()
    for relative_path in selected_paths:
        source_path = root / relative_path
        if not source_path.is_file():
            missing_count += 1
            continue
        target_relative_path = (
            build_unique_flat_name(relative_path=relative_path, used_names=used_flat_names)
            if is_flat
            else relative_path
        )
        target_path = destination / target_relative_path
        print(f"{relative_path} -> {target_path}")
        exported_count += 1
        if is_dry_run:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() or target_path.is_symlink():
            continue
        if is_symlink:
            target_path.symlink_to(source_path)
        else:
            shutil.copy2(source_path, target_path)
    return ExportSummary(exported_count=exported_count, missing_count=missing_count)


def select_export_ratings(*, min_rating: int | None, exact_ratings: list[int] | None) -> frozenset[int]:
    """Ratings to export; empty if the arguments are out of range. Default: 5 stars only."""
    if exact_ratings:
        if any(not REJECTED <= rating <= MAX_STARS for rating in exact_ratings):
            return frozenset()
        return frozenset(exact_ratings)
    effective_min_rating = MAX_STARS if min_rating is None else min_rating
    if not 1 <= effective_min_rating <= MAX_STARS:
        return frozenset()
    return frozenset(range(effective_min_rating, MAX_STARS + 1))


def build_unique_flat_name(*, relative_path: str, used_names: set[str]) -> str:
    """``2023/summer/IMG_1.jpg`` -> ``IMG_1.jpg``; on collision ``2023_summer_IMG_1.jpg``. Records the name."""
    file_name = Path(relative_path).name
    candidate_name = file_name if file_name not in used_names else relative_path.replace("/", "_")
    used_names.add(candidate_name)
    return candidate_name


def _add_database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"SQLite file (default: ROOT/{RatingStore.DEFAULT_FILE_NAME})",
    )
