"""Command line entry point: ``lazyrate ROOT`` (TUI), ``stats``, ``export``, ``classify``, ``search``."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from lazyrate.embeddings import ClipEmbedder, is_embedding_available
from lazyrate.export import ExportOptions, export_files, select_export_ratings
from lazyrate.learning import (
    build_current_embedding_matrix,
    compute_embeddings,
    find_files_missing_embeddings,
    train_and_predict,
)
from lazyrate.media import scan_media_files
from lazyrate.ratings import ALL_BUCKETS, MAX_STARS, REJECTED, describe_bucket
from lazyrate.session import describe_suggestion
from lazyrate.store import LEGACY_DATABASE_FILE_NAME, RatingStore, find_legacy_database, migrate_legacy_database

if TYPE_CHECKING:
    from lazyrate.embeddings import EmbeddingMatrix, ImageEmbedder
    from lazyrate.learning import LearningResult
    from lazyrate.media import ScanResult

SUBCOMMANDS = frozenset({"rate", "stats", "export", "classify", "search"})
ML_INSTALL_HINT = "install the AI extra: uv tool install 'lazyrate[ml] @ git+https://github.com/wrichert/lazyrate'"
PROGRESS_BAR_WIDTH = 30
DEFAULT_SEARCH_LIMIT = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lazyrate", description="Rate all images in a folder tree, fast.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rate_parser = subparsers.add_parser("rate", help="open the rating TUI (default command)")
    rate_parser.add_argument("root", type=Path, help="photo folder")
    rate_parser.add_argument("--no-ai", action="store_true", help="no AI suggestions / search even if installed")
    _add_database_argument(rate_parser)

    stats_parser = subparsers.add_parser("stats", help="print rating counts (and AI accuracy, if indexed)")
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

    classify_parser = subparsers.add_parser(
        "classify", help="index image content, learn from your ratings, report accuracy and suggestions (AI)"
    )
    classify_parser.add_argument("root", type=Path)
    _add_database_argument(classify_parser)

    search_parser = subparsers.add_parser("search", help="find photos by a text description (AI)")
    search_parser.add_argument("root", type=Path)
    search_parser.add_argument("query", help='e.g. "beach sunset" or "receipt"')
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="number of results")
    _add_database_argument(search_parser)
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
        return run_tui(root=root, database_path=database_path, is_ai_enabled=not parsed.no_ai)
    store = RatingStore(database_path)
    try:
        if parsed.command == "stats":
            return run_stats(root=root, store=store)
        if parsed.command == "classify":
            return run_classify(root=root, store=store)
        if parsed.command == "search":
            return run_search(root=root, store=store, query=parsed.query, limit=parsed.limit)
        return run_export(root=root, store=store, parsed=parsed)
    finally:
        store.close()


def run_tui(*, root: Path, database_path: Path, is_ai_enabled: bool) -> int:
    # Imported lazily: textual_image probes the terminal on import, which is not wanted for the other commands.
    from lazyrate.app import ImageRaterApp
    from lazyrate.session import RatingSession

    print(f"Scanning {root} …", file=sys.stderr)
    scan_result = scan_media_files(root)
    store = RatingStore(database_path)
    try:
        session = RatingSession(scan_result=scan_result, store=store)
        ImageRaterApp(
            session=session,
            database_path=database_path,
            embedder_factory=ClipEmbedder if is_ai_enabled and is_embedding_available() else None,
        ).run()
    finally:
        store.close()
    return 0


def run_stats(*, root: Path, store: RatingStore) -> int:
    print_rating_counts(store)
    embeddings = store.load_embeddings(ClipEmbedder.MODEL_NAME)
    if not embeddings:
        return 0
    scan_result = scan_media_files(root)
    embedding_matrix = build_current_embedding_matrix(files=scan_result.files, embeddings=embeddings)
    print()
    print_learning_report(train_and_predict(embedding_matrix=embedding_matrix, ratings=store.load_ratings()), store)
    return 0


def run_classify(*, root: Path, store: RatingStore) -> int:
    if not is_embedding_available():
        print(f"lazyrate: classify needs the AI extra; {ML_INSTALL_HINT}", file=sys.stderr)
        return 2
    scan_result = scan_media_files(root)
    embedding_matrix = index_embeddings(root=root, store=store, scan_result=scan_result, embedder=ClipEmbedder())
    print_learning_report(train_and_predict(embedding_matrix=embedding_matrix, ratings=store.load_ratings()), store)
    return 0


def run_search(*, root: Path, store: RatingStore, query: str, limit: int) -> int:
    if not is_embedding_available():
        print(f"lazyrate: search needs the AI extra; {ML_INSTALL_HINT}", file=sys.stderr)
        return 2
    scan_result = scan_media_files(root)
    embedder = ClipEmbedder()
    embedding_matrix = index_embeddings(root=root, store=store, scan_result=scan_result, embedder=embedder)
    scores = embedding_matrix.rank_by_similarity(embedder.embed_text(query))
    for relative_path, score in sorted(scores.items(), key=lambda item: -item[1])[:limit]:
        print(f"{score:.3f}  {root / relative_path}")
    return 0


def run_export(*, root: Path, store: RatingStore, parsed: argparse.Namespace) -> int:
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


def index_embeddings(
    *, root: Path, store: RatingStore, scan_result: ScanResult, embedder: ImageEmbedder
) -> EmbeddingMatrix:
    """Embed new / changed photos (progress on stderr) and return all current embeddings."""
    embeddings = store.load_embeddings(embedder.model_name)
    pending_files = find_files_missing_embeddings(files=scan_result.files, embeddings=embeddings)
    if pending_files:
        compute_embeddings(
            root=root,
            files=pending_files,
            embedder=embedder,
            store=store,
            on_progress=print_progress,
            should_stop=lambda: False,
        )
        print(file=sys.stderr)
        embeddings = store.load_embeddings(embedder.model_name)
    return build_current_embedding_matrix(files=scan_result.files, embeddings=embeddings)


def print_progress(done_count: int, total_count: int) -> None:
    filled_width = PROGRESS_BAR_WIDTH * done_count // max(1, total_count)
    progress_bar = "█" * filled_width + "░" * (PROGRESS_BAR_WIDTH - filled_width)
    print(f"\rReading image content {progress_bar} {done_count:,}/{total_count:,}", end="", file=sys.stderr, flush=True)


def print_rating_counts(store: RatingStore) -> None:
    ratings = store.load_ratings()
    for bucket in reversed(ALL_BUCKETS[1:]):
        count = sum(1 for rating in ratings.values() if rating == bucket)
        print(f"{describe_bucket(bucket):<9} {count:>7,}")
    print(f"{'rated':<9} {sum(1 for rating in ratings.values() if rating != REJECTED):>7,}")


def print_learning_report(result: LearningResult, store: RatingStore) -> None:
    readiness = result.readiness
    print(f"AI index: {result.embedded_count:,} photos")
    print(
        f"Training data: {readiness.rejected_count:,} rejected, {readiness.starred_count:,} starred "
        f"({readiness.distinct_star_count} star levels)"
    )
    if not readiness.is_junk_model_ready:
        print(
            "Not enough ratings yet: rate at least "
            f"{readiness.missing_rejected_count} more rejected and enough starred photos"
        )
        return
    evaluation = result.evaluation
    if evaluation is None or evaluation.junk_evaluated_count == 0:
        print("Accuracy:    not measurable yet (each check holds out a fifth of the ratings; rate a few more)")
    if evaluation is not None and evaluation.junk_precision is not None and evaluation.junk_recall is not None:
        print(
            f"Junk model:  precision {evaluation.junk_precision:.0%} · recall {evaluation.junk_recall:.0%} "
            f"(cross-validated on {evaluation.junk_evaluated_count:,} rated photos)"
        )
    if evaluation is not None and evaluation.star_within_one_accuracy is not None:
        print(
            f"Star model:  exact {evaluation.star_exact_accuracy:.0%} · within ±1 star "
            f"{evaluation.star_within_one_accuracy:.0%} · mean error {evaluation.star_mean_absolute_error:.2f} stars "
            f"({evaluation.star_evaluated_count:,} starred photos)"
        )
    elif not readiness.is_star_model_ready:
        print(f"Star model:  needs {readiness.missing_starred_count} more starred photos")
    ratings = store.load_ratings()
    suggestion_counts = Counter(
        describe_suggestion(suggestion)
        for relative_path, suggestion in result.suggestions.items()
        if relative_path not in ratings
    )
    print(f"Suggestions for {sum(suggestion_counts.values()):,} unrated photos:")
    for label, count in sorted(suggestion_counts.items(), key=lambda item: -item[1]):
        print(f"  {label:<20} {count:>7,}")


def _add_database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"SQLite file (default: ROOT/{RatingStore.DEFAULT_FILE_NAME})",
    )
