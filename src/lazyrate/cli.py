"""Command line entry point: ``lazyrate ROOT``, ``lazyrate stats ROOT``, ``lazyrate export ROOT DEST``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lazyrate.export import ExportOptions, export_files, select_export_ratings
from lazyrate.media import scan_media_files
from lazyrate.ratings import ALL_BUCKETS, MAX_STARS, REJECTED, describe_bucket
from lazyrate.store import LEGACY_DATABASE_FILE_NAME, RatingStore, find_legacy_database, migrate_legacy_database

SUBCOMMANDS = frozenset({"rate", "stats", "export"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lazyrate", description="Rate all images in a folder tree, fast.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rate_parser = subparsers.add_parser("rate", help="open the rating TUI (default command)")
    rate_parser.add_argument("root", type=Path, help="photo folder")
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
    parser = build_parser()
    if not arguments:
        parser.print_usage(sys.stderr)
        print("lazyrate: error: photo folder missing, e.g. `lazyrate ~/Pictures/2025`", file=sys.stderr)
        return 2
    # `lazyrate ~/Photos` is shorthand for `lazyrate rate ~/Photos`.
    if arguments[0] not in SUBCOMMANDS and arguments[0] not in {"-h", "--help"}:
        arguments = ["rate", *arguments]
    parsed = parser.parse_args(arguments)
    root = parsed.root.expanduser().resolve()
    if not root.is_dir():
        print(f"lazyrate: not a directory: {root}", file=sys.stderr)
        return 2
    if parsed.db is None and find_legacy_database(root) is not None:
        migrate_legacy_database(root)
        print(f"lazyrate: renamed {LEGACY_DATABASE_FILE_NAME} to {RatingStore.DEFAULT_FILE_NAME}", file=sys.stderr)
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
                f"lazyrate: --rating must be {REJECTED}..{MAX_STARS}, --min-rating must be 1..{MAX_STARS}",
                file=sys.stderr,
            )
            return 2
        relative_paths = [path for path, rating in store.load_ratings().items() if rating in selected_ratings]
        try:
            summary = export_files(
                root=root,
                relative_paths=relative_paths,
                options=ExportOptions(
                    destination=parsed.destination,
                    is_flat=parsed.flat,
                    is_symlink=parsed.symlink,
                    is_dry_run=parsed.dry_run,
                ),
            )
        except ValueError as error:
            print(f"lazyrate: {error}", file=sys.stderr)
            return 2
        for target_path in summary.target_paths:
            print(target_path)
        verb = "would export" if parsed.dry_run else "exported"
        print(
            f"{verb} {summary.exported_count} files, {summary.skipped_existing_count} already there, "
            f"{summary.missing_count} missing on disk",
            file=sys.stderr,
        )
        return 0
    finally:
        store.close()


def run_tui(*, root: Path, database_path: Path) -> int:
    # Imported lazily: textual_image probes the terminal on import, which is not wanted for stats/export.
    from lazyrate.app import ImageRaterApp
    from lazyrate.session import RatingSession

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


def _add_database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"SQLite file (default: ROOT/{RatingStore.DEFAULT_FILE_NAME})",
    )
