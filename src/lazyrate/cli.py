"""Command line entry point: ``lazyrate ROOT`` (TUI), ``stats``, ``export``, ``classify``, ``search``, ``map``."""

from __future__ import annotations

import argparse
import functools
import sys
import webbrowser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lazyrate.embeddings import ClipEmbedder, is_embedding_available
from lazyrate.export import ExportOptions, export_files, select_export_ratings
from lazyrate.geo_map import BROWSER_MAP_FILE_NAME, BrowserMapWriter, MapPoint
from lazyrate.geo_search import (
    DEFAULT_RADIUS_KM,
    SearchResult,
    find_place_match,
    parse_radius_query,
    rank_within_place,
    search_by_place,
    search_by_radius,
)
from lazyrate.learning import (
    build_current_embedding_matrix,
    compute_embeddings,
    find_files_missing_embeddings,
    train_and_predict,
)
from lazyrate.locations import LocationReader, PlaceIndex, find_files_missing_locations
from lazyrate.media import scan_media_files
from lazyrate.ratings import ALL_BUCKETS, MAX_STARS, REJECTED, describe_bucket
from lazyrate.session import describe_suggestion
from lazyrate.store import LEGACY_DATABASE_FILE_NAME, RatingStore, find_legacy_database, migrate_legacy_database

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazyrate.embeddings import EmbeddingMatrix, ImageEmbedder
    from lazyrate.learning import LearningResult
    from lazyrate.locations import PhotoLocation
    from lazyrate.media import ScanResult

CURRENT_FOLDER = Path()
SUBCOMMANDS_WITH_QUERY = frozenset({"search", "map"})
SUBCOMMANDS = frozenset({"rate", "stats", "export", "classify", "search", "map"})
ML_INSTALL_HINT = "install the AI extra: uv tool install 'lazyrate[ml] @ git+https://github.com/wrichert/lazyrate'"
PROGRESS_BAR_WIDTH = 30
DEFAULT_TEXT_SEARCH_LIMIT = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lazyrate", description="Rate all images in a folder tree, fast.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rate_parser = subparsers.add_parser("rate", help="open the rating TUI (default command)")
    rate_parser.add_argument("root", type=Path, nargs="?", default=CURRENT_FOLDER, help="photo folder (default: .)")
    rate_parser.add_argument("--no-ai", action="store_true", help="no AI suggestions / search even if installed")
    _add_database_argument(rate_parser)

    stats_parser = subparsers.add_parser("stats", help="print rating counts (and AI accuracy, if indexed)")
    stats_parser.add_argument("root", type=Path, nargs="?", default=CURRENT_FOLDER, help="default: .")
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
    classify_parser.add_argument("root", type=Path, nargs="?", default=CURRENT_FOLDER, help="default: .")
    _add_database_argument(classify_parser)

    search_parser = subparsers.add_parser(
        "search", help="find photos by place (GPS), coordinates, or text description (AI)"
    )
    search_parser.add_argument("root", nargs="?", default=None, help="photo folder (default: .)")
    search_parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help='a place ("madeira", "funchal", "portugal"), "@32.75,-16.95 50km", or a description '
        '("beach sunset", AI); combine: "beach madeira"',
    )
    _add_near_arguments(search_parser)
    search_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"number of results (default: all for place searches, {DEFAULT_TEXT_SEARCH_LIMIT} for text)",
    )
    _add_database_argument(search_parser)

    map_parser = subparsers.add_parser(
        "map", help="open a browser map of all photos with GPS data, or of a search's matches"
    )
    map_parser.add_argument("root", nargs="?", default=None, help="photo folder (default: .)")
    map_parser.add_argument("query", nargs="?", default=None, help="optional search, same forms as `search`")
    _add_near_arguments(map_parser)
    map_parser.add_argument(
        "--output", type=Path, default=None, help=f"HTML file (default: {BROWSER_MAP_FILE_NAME} next to the database)"
    )
    map_parser.add_argument("--no-open", action="store_true", help="only write the file, do not open the browser")
    _add_database_argument(map_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    # `lazyrate ~/Photos` is shorthand for `lazyrate rate ~/Photos`, plain `lazyrate` rates the current folder.
    if not arguments or (arguments[0] not in SUBCOMMANDS and arguments[0] not in {"-h", "--help"}):
        arguments = ["rate", *arguments]
    parsed = parser.parse_args(arguments)
    if parsed.command in SUBCOMMANDS_WITH_QUERY:
        search_target = resolve_search_target(first_argument=parsed.root, query=parsed.query)
        parsed = argparse.Namespace(**{**vars(parsed), "root": search_target.root, "query": search_target.query})
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
            return run_search(root=root, store=store, parsed=parsed)
        if parsed.command == "map":
            return run_map(root=root, store=store, parsed=parsed)
        return run_export(root=root, store=store, parsed=parsed)
    finally:
        store.close()


@dataclass(frozen=True, slots=True)
class SearchTarget:
    root: Path
    query: str | None


def resolve_search_target(*, first_argument: str | None, query: str | None) -> SearchTarget:
    """``lazyrate search madeira`` searches the current folder: a first word that is no folder (and does not look
    like a path) belongs to the query."""
    if first_argument is None:
        return SearchTarget(root=CURRENT_FOLDER, query=query)
    looks_like_path = first_argument.startswith(("/", "~", ".")) or "/" in first_argument
    if looks_like_path or Path(first_argument).expanduser().is_dir():
        return SearchTarget(root=Path(first_argument), query=query)
    return SearchTarget(root=CURRENT_FOLDER, query=first_argument if query is None else f"{first_argument} {query}")


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
    print_location_counts(store)
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


class SearchError(Exception):
    """A search that cannot run (bad query, missing AI extra); the message is shown as is."""


@dataclass(frozen=True, slots=True)
class ResolvedSearch:
    result: SearchResult
    format_rank: Callable[[float], str] | None
    """Formats the score (distance, similarity); ``None``: show the section label (town) instead."""
    default_limit: int | None


def run_search(*, root: Path, store: RatingStore, parsed: argparse.Namespace) -> int:
    try:
        query = build_query_text(parsed)
        if query is None:
            raise SearchError("give a search query or --near LAT,LON")
        scan_result = scan_media_files(root)
        locations = index_locations(root=root, store=store, scan_result=scan_result)
        resolved_search = resolve_search(
            query=query, root=root, store=store, scan_result=scan_result, locations=locations
        )
    except SearchError as error:
        print(f"lazyrate: {error}", file=sys.stderr)
        return 2
    limit = resolved_search.default_limit if parsed.limit is None else parsed.limit
    return print_search_result(
        root=root, result=resolved_search.result, limit=limit, format_rank=resolved_search.format_rank
    )


def run_map(*, root: Path, store: RatingStore, parsed: argparse.Namespace) -> int:
    """All photos with GPS data, or the matches of a search, on a browser map."""
    try:
        query = build_query_text(parsed)
        scan_result = scan_media_files(root)
        locations = index_locations(root=root, store=store, scan_result=scan_result)
        if query is None:
            matching_paths = [path for path, location in locations.items() if location.coordinates is not None]
            title = f"{root.name} · all photos"
        else:
            result = resolve_search(
                query=query, root=root, store=store, scan_result=scan_result, locations=locations
            ).result
            matching_paths = sorted(result.scores, key=lambda relative_path: -result.scores[relative_path])
            title = f"{root.name} · {query}"
    except SearchError as error:
        print(f"lazyrate: {error}", file=sys.stderr)
        return 2
    ratings = store.load_ratings()
    map_points = [
        MapPoint(
            relative_path=relative_path,
            coordinates=location.coordinates,
            rating=ratings.get(relative_path),
            place=location.place,
        )
        for relative_path in matching_paths
        if (location := locations.get(relative_path)) is not None and location.coordinates is not None
    ]
    if not map_points:
        print("lazyrate: no matching photos with GPS data", file=sys.stderr)
        return 1
    output_path = parsed.output or store.database_path.parent / BROWSER_MAP_FILE_NAME
    BrowserMapWriter(root=root).write(
        map_points,
        output_path=output_path,
        title=f"{title} · {len(map_points):,} photos",
        on_progress=functools.partial(print_progress, "Building map"),
        should_stop=lambda: False,
    )
    print(file=sys.stderr)
    print(f"Map of {len(map_points):,} photos: {output_path}", file=sys.stderr)
    if not parsed.no_open:
        webbrowser.open(output_path.resolve().as_uri())
    return 0


def build_query_text(parsed: argparse.Namespace) -> str | None:
    """The query, or ``--near`` / ``--radius`` written as a radius query; ``None`` if neither was given."""
    if parsed.near is None:
        return parsed.query or None
    query = f"{parsed.near} {parsed.radius:g}km"
    try:
        is_radius_query = parse_radius_query(query) is not None
    except ValueError as error:
        raise SearchError(str(error)) from error
    if not is_radius_query:
        raise SearchError(f"--near needs LAT,LON, e.g. --near=32.75,-16.95, got {parsed.near!r}")
    return query


def resolve_search(
    *,
    query: str,
    root: Path,
    store: RatingStore,
    scan_result: ScanResult,
    locations: dict[str, PhotoLocation],
) -> ResolvedSearch:
    """Coordinates -> radius search; a place name -> photos taken there (ranked by the rest of the query with AI);
    anything else -> text search (AI)."""
    try:
        radius_query = parse_radius_query(query)
    except ValueError as error:
        raise SearchError(str(error)) from error
    if radius_query is not None:
        result = search_by_radius(query=radius_query, locations=locations.values())
        return ResolvedSearch(result=result, format_rank=format_distance, default_limit=None)
    place_match = find_place_match(query_text=query, locations=locations)
    if place_match is not None and (not place_match.remaining_text or not is_embedding_available()):
        if place_match.remaining_text:
            print(f"lazyrate: text search needs the AI extra; ignoring '{place_match.remaining_text}'", file=sys.stderr)
        result = search_by_place(match=place_match, locations=locations)
        return ResolvedSearch(result=result, format_rank=None, default_limit=None)
    if not is_embedding_available():
        located_count = sum(1 for location in locations.values() if location.coordinates is not None)
        raise SearchError(
            f"no photo taken in a place called '{query}' ({located_count:,} photos have GPS data); "
            f"searching by content needs the AI extra; {ML_INSTALL_HINT}"
        )
    embedder = ClipEmbedder()
    embedding_matrix = index_embeddings(root=root, store=store, scan_result=scan_result, embedder=embedder)
    text_query = query if place_match is None else place_match.remaining_text
    text_scores = embedding_matrix.rank_by_similarity(embedder.embed_text(text_query))
    if place_match is None:
        result = SearchResult(scores=text_scores, section_labels={}, summary="best matches first")
    else:
        result = rank_within_place(match=place_match, text_scores=text_scores)
    return ResolvedSearch(result=result, format_rank=format_similarity, default_limit=DEFAULT_TEXT_SEARCH_LIMIT)


def format_distance(score: float) -> str:
    """Radius search scores are negative distances."""
    return f"{-score:8.1f} km"


def format_similarity(score: float) -> str:
    return f"{score:.3f}"


def print_search_result(
    *, root: Path, result: SearchResult, limit: int | None, format_rank: Callable[[float], str] | None
) -> int:
    """One line per match, best first: rank (distance or similarity) or section label (town), then the path."""
    print(result.summary if result.scores else f"No matches: {result.summary}", file=sys.stderr)
    ranked_paths = sorted(result.scores, key=lambda relative_path: -result.scores[relative_path])
    for relative_path in ranked_paths if limit is None else ranked_paths[:limit]:
        if format_rank is None:
            rank_text = result.section_labels.get(relative_path, "")
        else:
            rank_text = format_rank(result.scores[relative_path])
        print(f"{rank_text}  {root / relative_path}")
    return 0 if result.scores else 1


def index_locations(*, root: Path, store: RatingStore, scan_result: ScanResult) -> dict[str, PhotoLocation]:
    """Read GPS data of new / changed photos (progress on stderr) and return all current locations."""
    locations = store.load_locations()
    pending_files = find_files_missing_locations(files=scan_result.files, locations=locations)
    if pending_files:
        LocationReader(root=root, place_index=PlaceIndex()).read_locations(
            pending_files,
            on_batch=functools.partial(save_location_batch, store, len(pending_files)),
            should_stop=lambda: False,
        )
        print(file=sys.stderr)
        locations = store.load_locations()
    current_paths = {media_file.relative_path for media_file in scan_result.files}
    return {relative_path: location for relative_path, location in locations.items() if relative_path in current_paths}


def save_location_batch(store: RatingStore, total_count: int, locations: list[PhotoLocation], done_count: int) -> None:
    store.save_locations(locations)
    print_progress("Reading GPS locations", done_count, total_count)


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
            on_progress=functools.partial(print_progress, "Reading image content"),
            should_stop=lambda: False,
        )
        print(file=sys.stderr)
        embeddings = store.load_embeddings(embedder.model_name)
    return build_current_embedding_matrix(files=scan_result.files, embeddings=embeddings)


def print_progress(title: str, done_count: int, total_count: int) -> None:
    filled_width = PROGRESS_BAR_WIDTH * done_count // max(1, total_count)
    progress_bar = "█" * filled_width + "░" * (PROGRESS_BAR_WIDTH - filled_width)
    print(f"\r{title} {progress_bar} {done_count:,}/{total_count:,}", end="", file=sys.stderr, flush=True)


def print_rating_counts(store: RatingStore) -> None:
    ratings = store.load_ratings()
    for bucket in reversed(ALL_BUCKETS[1:]):
        count = sum(1 for rating in ratings.values() if rating == bucket)
        print(f"{describe_bucket(bucket):<9} {count:>7,}")
    print(f"{'rated':<9} {sum(1 for rating in ratings.values() if rating != REJECTED):>7,}")


def print_location_counts(store: RatingStore) -> None:
    """GPS summary, if the TUI or ``search`` has read locations already."""
    locations = [location for location in store.load_locations().values() if location.place is not None]
    if not locations:
        return
    country_counts = Counter(location.place.country for location in locations if location.place is not None)
    town_count = len({location.place.path_label for location in locations if location.place is not None})
    top_countries = ", ".join(f"{country} {count:,}" for country, count in country_counts.most_common(5))
    print(f"GPS: {len(locations):,} photos in {town_count:,} towns ({top_countries})")


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


def _add_near_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--near", metavar="LAT,LON", help="photos near these coordinates, e.g. --near=32.75,-16.95 (= for negatives)"
    )
    parser.add_argument(
        "--radius", type=float, default=DEFAULT_RADIUS_KM, metavar="KM", help="radius for --near (default: 10)"
    )


def _add_database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"SQLite file (default: ROOT/{RatingStore.DEFAULT_FILE_NAME})",
    )
