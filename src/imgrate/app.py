"""Textual TUI for rating images."""

from __future__ import annotations

import functools
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Label, Static
from textual.worker import get_current_worker
from textual_image.widget import Image as ImageWidget
from textual_image.widget import get_cell_size

from imgrate.export import ExportOptions, ExportSummary, export_files
from imgrate.fingerprints import Fingerprint, compute_image_signature
from imgrate.grid_layout import (
    GridPage,
    GridRow,
    build_grid_rows,
    find_index_in_adjacent_row,
    find_page_index,
    paginate_grid_rows,
)
from imgrate.media import scan_media_files
from imgrate.preview import PreviewCache
from imgrate.ratings import ALL_BUCKETS, FILTER_PRESETS, MAX_STARS, REJECTED, describe_bucket, format_stars
from imgrate.screens import ExportRequest, ExportScreen, ExtensionsScreen, HelpScreen, RatingFilterScreen
from imgrate.session import SortOrder
from imgrate.store import RatingStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from textual.binding import BindingType
    from textual.timer import Timer
    from textual.widget import Widget

    from imgrate.media import MediaFile, ScanResult
    from imgrate.preview import Preview
    from imgrate.ratings import RatingFilter
    from imgrate.session import RatingSession


BYTES_PER_KILOBYTE = 1024


@dataclass(frozen=True, slots=True)
class CellBox:
    """A size in terminal cells."""

    columns: int
    rows: int


def fit_image_to_cells(
    *,
    image_width_px: int,
    image_height_px: int,
    max_columns: int,
    max_rows: int,
    cell_width_px: int,
    cell_height_px: int,
) -> CellBox:
    """Largest cell box with the image's aspect ratio that fits into ``max_columns`` x ``max_rows``."""
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError(f"Invalid image size {image_width_px}x{image_height_px}")
    scale_factor = min(
        max_columns * cell_width_px / image_width_px,
        max_rows * cell_height_px / image_height_px,
    )
    return CellBox(
        columns=max(1, min(max_columns, math.floor(image_width_px * scale_factor / cell_width_px))),
        rows=max(1, min(max_rows, math.floor(image_height_px * scale_factor / cell_height_px))),
    )


def format_size(size_bytes: int) -> str:
    if size_bytes < BYTES_PER_KILOBYTE:
        return f"{size_bytes} B"
    size_value = size_bytes / BYTES_PER_KILOBYTE
    for unit in ("KB", "MB"):
        if size_value < BYTES_PER_KILOBYTE:
            return f"{size_value:.1f} {unit}"
        size_value /= BYTES_PER_KILOBYTE
    return f"{size_value:.1f} GB"


def truncate_from_left(text: str, *, max_length: int) -> str:
    """Keep the end of long labels (folder names are most specific at the end)."""
    return text if len(text) <= max_length else "…" + text[-(max_length - 1) :]


def shorten_home_path(path: Path) -> str:
    home = os.path.expanduser("~")
    path_text = str(path)
    return "~" + path_text[len(home) :] if path_text.startswith(home) else path_text


@dataclass(frozen=True, slots=True)
class IndexingProgress:
    done_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class GridLayoutKey:
    """Everything the grid's rows and pages depend on; a change triggers a re-layout."""

    view_version: int
    column_count: int
    available_height: int


def build_rating_text(rating: int | None) -> Text:
    stars = format_stars(rating)
    if rating is None:
        return Text(stars, style="dim")
    if rating == REJECTED:
        return Text(stars, style="bold red")
    return Text(stars, style="bold yellow")


class ThumbnailCell(Vertical):
    """One tile of the grid: thumbnail plus rating line. ``file_index`` is ``None`` for padding tiles."""

    def __init__(self, *, file_index: int | None) -> None:
        super().__init__(classes="thumbnail" if file_index is not None else "thumbnail empty")
        self.file_index = file_index
        self.relative_path: str | None = None

    def compose(self) -> ComposeResult:
        with Container(classes="thumbnail-image-box"):
            yield ImageWidget(classes="thumbnail-image")
        yield Label("", classes="thumbnail-label")


class ImageRaterApp(App[None]):
    """Rate thousands of photos fast; review any rating bucket; persist everything in SQLite."""

    TITLE = "imgrate"
    EXPORT_OPTIONS_SETTING_KEY = "export_options"
    SECTION_LIST_MAX_LINES = 9
    SECTION_LABEL_WIDTH = 20
    PREVIEW_EDGE_STEP_PX = 256
    PREVIEW_MIN_EDGE_PX = 512
    PREVIEW_MAX_EDGE_PX = 1920
    THUMBNAIL_EDGE_PX = 384
    THUMBNAIL_WIDTH_CELLS = 28
    THUMBNAIL_HEIGHT_CELLS = 13
    """Must match ``.thumbnail-row { height }`` in CSS."""
    SECTION_HEADER_HEIGHT_CELLS = 1
    PAGE_JUMP_SIZE = 10
    PREFETCH_AHEAD_COUNT = 3
    RESIZE_DEBOUNCE_SECONDS = 0.15
    FINGERPRINT_WORKER_COUNT = 2
    FINGERPRINT_BATCH_SIZE = 64
    SORT_ORDERS: tuple[SortOrder, ...] = tuple(SortOrder)

    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    #viewer { width: 1fr; height: 1fr; align: center middle; }
    #preview { width: auto; height: auto; }
    #placeholder { width: 100%; height: 100%; content-align: center middle; color: $text-muted; }
    #grid { width: 1fr; height: 1fr; display: none; }
    .section-header { height: 1; padding: 0 1; background: $accent 35%; color: $text; text-style: bold; }
    .thumbnail-row { height: 13; }
    .thumbnail { border: round $panel; height: 100%; width: 1fr; }
    .thumbnail.empty { border: none; }
    .thumbnail.current { border: heavy $accent; }
    .thumbnail-image-box { height: 1fr; align: center middle; }
    .thumbnail-image { width: auto; height: auto; }
    .thumbnail-label { width: 100%; height: 1; content-align: center middle; }
    #sidebar { width: 34; border-left: vkey $panel; padding: 0 1; }
    #sidebar Static { margin-bottom: 1; }
    #statusbar { height: 1; padding: 0 1; background: $panel; }
    .dialog { width: 64; height: auto; max-height: 90%; padding: 1 2; background: $surface; border: thick $accent; }
    .dialog SelectionList { height: auto; max-height: 24; margin: 1 0; }
    .dialog .buttons { height: auto; align-horizontal: right; }
    .dialog Button { margin-left: 1; }
    .help { width: 84; }
    ExtensionsScreen, RatingFilterScreen, ExportScreen, HelpScreen { align: center middle; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("right,l,space", "move(1)", "Next", show=False),
        Binding("left,h", "move(-1)", "Prev", show=False),
        Binding("down,j", "move_row(1)", "Down", show=False),
        Binding("up,k", "move_row(-1)", "Up", show=False),
        Binding("pagedown", "move_page(1)", "Page down", show=False),
        Binding("pageup", "move_page(-1)", "Page up", show=False),
        Binding("J,right_square_bracket,full_stop", "move_section(1)", "Section ↓", key_display="J"),
        Binding("K,left_square_bracket,comma", "move_section(-1)", "Section ↑", key_display="K"),
        Binding("home", "move_to_start", "First", show=False),
        Binding("end", "move_to_end", "Last", show=False),
        *(
            Binding(
                str(stars),
                f"rate({stars})",
                "Rate",
                show=stars == MAX_STARS,
                key_display="1-5" if stars == MAX_STARS else None,
            )
            for stars in range(1, MAX_STARS + 1)
        ),
        Binding("0,x", f"rate({REJECTED})", "Reject"),
        Binding("u", "clear_rating", "Unrate"),
        Binding("z", "undo", "Undo"),
        Binding("n", "next_unrated", "Next unrated"),
        Binding("v", "cycle_filter(1)", "View"),
        Binding("V", "cycle_filter(-1)", "View back", show=False),
        Binding("f", "choose_filter", "Filter"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("g", "toggle_grid", "Grid"),
        Binding("d", "toggle_duplicates", "Dupes"),
        Binding("b", "keep_best_of_group", "Best of group"),
        Binding("enter", "open_from_grid", "Open", show=False),
        Binding("E", "export", "Export"),
        Binding("e", "choose_extensions", "Formats"),
        Binding("o", "open_external", "Open ext"),
        Binding("a", "toggle_auto_advance", "Auto-adv", show=False),
        Binding("r", "rescan", "Rescan", show=False),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, *, session: RatingSession, database_path: Path) -> None:
        super().__init__()
        self._session = session
        self._database_path = database_path
        self._indexing_progress: IndexingProgress | None = None
        self._preview_cache = PreviewCache()
        self._is_auto_advance = True
        self._is_grid_mode = False
        self._grid_layout_key: GridLayoutKey | None = None
        self._grid_rows: list[GridRow] = []
        self._grid_pages: list[GridPage] = []
        self._shown_grid_page: GridPage | None = None
        self._resize_timer: Timer | None = None
        self._current_preview: Preview | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with Container(id="viewer"):
                yield ImageWidget(id="preview")
                yield Static("", id="placeholder")
            yield Container(id="grid")
            with Vertical(id="sidebar"):
                yield Static(id="info")
                yield Static(id="sections")
                yield Static(id="counts")
                yield Static(id="settings")
        yield Static(id="statusbar")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_all()
        self._start_fingerprint_indexing()

    def on_unmount(self) -> None:
        self._preview_cache.shutdown()

    def on_resize(self) -> None:
        if self._resize_timer is not None:
            self._resize_timer.stop()
        self._resize_timer = self.set_timer(self.RESIZE_DEBOUNCE_SECONDS, self._refresh_all)

    # ----------------------------------------------------------------- actions

    def action_move(self, delta: int) -> None:
        self._navigate(lambda: self._session.move_by(delta))

    def action_move_row(self, direction: int) -> None:
        if not (self._is_grid_mode and self._grid_rows):
            self._navigate(lambda: self._session.move_by(direction))
            return
        target_index = find_index_in_adjacent_row(
            self._grid_rows, file_index=self._session.current_index, direction=direction
        )
        self._navigate(lambda: self._session.move_to(target_index))

    def action_move_page(self, direction: int) -> None:
        if not (self._is_grid_mode and self._grid_pages):
            self._navigate(lambda: self._session.move_by(direction * self.PAGE_JUMP_SIZE))
            return
        page_index = find_page_index(self._grid_pages, self._session.current_index)
        target_page_index = max(0, min(page_index + direction, len(self._grid_pages) - 1))
        target_index = self._grid_pages[target_page_index].start_index
        self._navigate(lambda: self._session.move_to(target_index))

    def action_move_section(self, direction: int) -> None:
        self._navigate(lambda: self._session.move_by_section(direction))

    def action_move_to_start(self) -> None:
        self._navigate(lambda: self._session.move_to(0))

    def action_move_to_end(self) -> None:
        self._navigate(lambda: self._session.move_to(self._session.visible_count - 1))

    def action_next_unrated(self) -> None:
        self._navigate(self._move_to_next_unrated)

    def action_rate(self, rating: int) -> None:
        self._apply_rating(rating)

    def action_clear_rating(self) -> None:
        self._apply_rating(None)

    def action_undo(self) -> None:
        changes = self._session.undo()
        if not changes:
            self.notify("Nothing to undo")
            return
        if len(changes) == 1:
            self.notify(f"Undone: {changes[0].relative_path} → {format_stars(changes[0].previous_rating)}")
        else:
            self.notify(f"Undone: {len(changes)} rating changes")
        self._refresh_all()

    def action_cycle_sort(self) -> None:
        if self._session.is_duplicates_mode:
            self.notify("Duplicate groups are always ordered by time (press d to leave)")
            return
        sort_index = self.SORT_ORDERS.index(self._session.sort_order)
        sort_order = self.SORT_ORDERS[(sort_index + 1) % len(self.SORT_ORDERS)]
        self._session.set_sort_order(sort_order)
        message = f"Sort: {sort_order.value}"
        if sort_order is SortOrder.DATE and self._indexing_progress is not None:
            message += " (still reading dates; unknown ones use file time)"
        self.notify(message, timeout=2)
        self._refresh_all()

    def action_toggle_duplicates(self) -> None:
        is_duplicates_mode = not self._session.is_duplicates_mode
        if is_duplicates_mode:
            self._session.rebuild_duplicate_groups()
        self._session.set_duplicates_mode(is_duplicates_mode=is_duplicates_mode)
        if is_duplicates_mode:
            self.notify(self._describe_duplicates_state(), timeout=3)
        self._refresh_all()

    def action_keep_best_of_group(self) -> None:
        if not self._session.is_duplicates_mode:
            self.notify("Only in duplicates mode (press d)")
            return
        changes = self._session.reject_others_in_group()
        if not changes:
            return
        self.notify(f"Kept 1, rejected {len(changes)} — z to undo", timeout=2)
        self._session.move_by_section(1)
        self._refresh_all()

    def action_cycle_filter(self, direction: int) -> None:
        current_filter = self._session.rating_filter
        preset_index = next(
            (index for index, preset in enumerate(FILTER_PRESETS) if preset.buckets == current_filter.buckets),
            -1 if direction > 0 else 0,
        )
        self._apply_filter(FILTER_PRESETS[(preset_index + direction) % len(FILTER_PRESETS)])

    def action_choose_filter(self) -> None:
        self.push_screen(
            RatingFilterScreen(rating_counts=self._session.count_ratings(), current_filter=self._session.rating_filter),
            self._handle_filter_chosen,
        )

    def action_choose_extensions(self) -> None:
        self.push_screen(
            ExtensionsScreen(
                extension_counts=self._session.extension_counts,
                enabled_extensions=self._session.enabled_extensions,
            ),
            self._handle_extensions_chosen,
        )

    def action_toggle_grid(self) -> None:
        self._is_grid_mode = not self._is_grid_mode
        self.query_one("#viewer").display = not self._is_grid_mode
        self.query_one("#grid").display = self._is_grid_mode
        self.call_after_refresh(self._refresh_all)

    def action_open_from_grid(self) -> None:
        if self._is_grid_mode:
            self.action_toggle_grid()

    def action_open_external(self) -> None:
        current_file = self._session.current_file
        if current_file is None:
            return
        opener_command = "open" if sys.platform == "darwin" else "xdg-open"
        try:
            subprocess.Popen(
                [opener_command, str(self._session.resolve_path(current_file))],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            self.notify(f"Cannot open: {error}", severity="error")

    def action_toggle_auto_advance(self) -> None:
        self._is_auto_advance = not self._is_auto_advance
        self.notify(f"Auto-advance {'on' if self._is_auto_advance else 'off'}")
        self._refresh_sidebar()

    def action_rescan(self) -> None:
        self.notify("Rescanning…")
        self.run_worker(self._rescan, thread=True, exclusive=True, group="scan")

    def action_export(self) -> None:
        current_filter_buckets = self._session.rating_filter.buckets
        initial_buckets = (
            frozenset({MAX_STARS}) if current_filter_buckets == frozenset(ALL_BUCKETS) else current_filter_buckets
        )
        saved_options = json.loads(self._session.load_setting(self.EXPORT_OPTIONS_SETTING_KEY) or "{}")
        default_destination = self._session.root.parent / f"{self._session.root.name}-selection"
        self.push_screen(
            ExportScreen(
                root=self._session.root,
                rating_counts=self._session.count_ratings(),
                initial_buckets=initial_buckets,
                initial_destination=saved_options.get("destination", shorten_home_path(default_destination)),
                is_flat=saved_options.get("is_flat", True),
                is_symlink=saved_options.get("is_symlink", False),
            ),
            self._handle_export_requested,
        )

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    # ---------------------------------------------------------------- handlers

    def _handle_filter_chosen(self, rating_filter: RatingFilter | None) -> None:
        if rating_filter is not None:
            self._apply_filter(rating_filter)

    def _handle_export_requested(self, request: ExportRequest | None) -> None:
        if request is None:
            return
        self._session.save_setting(
            key=self.EXPORT_OPTIONS_SETTING_KEY,
            value=json.dumps(
                {
                    "destination": shorten_home_path(request.options.destination),
                    "is_flat": request.options.is_flat,
                    "is_symlink": request.options.is_symlink,
                }
            ),
        )
        relative_paths = self._session.find_relative_paths_with_ratings(request.buckets)
        if not relative_paths:
            self.notify("No files with these ratings", severity="warning")
            return
        self.notify(f"Exporting {len(relative_paths):,} files…")
        self.run_worker(
            functools.partial(self._run_export, relative_paths, request.options),
            thread=True,
            group="export",
        )

    def _run_export(self, relative_paths: list[str], options: ExportOptions) -> None:
        try:
            summary = export_files(root=self._session.root, relative_paths=relative_paths, options=options)
        except (OSError, ValueError) as error:
            self.call_from_thread(self.notify, f"Export failed: {error}", severity="error", timeout=10)
            return
        self.call_from_thread(self._handle_export_done, summary, options)

    def _handle_export_done(self, summary: ExportSummary, options: ExportOptions) -> None:
        message = f"Exported {summary.exported_count:,} files to {shorten_home_path(options.destination)}"
        if summary.skipped_existing_count:
            message += f" ({summary.skipped_existing_count:,} already there)"
        if summary.missing_count:
            message += f" ({summary.missing_count:,} missing on disk)"
        self.notify(message, timeout=8)

    def _handle_extensions_chosen(self, extensions: set[str] | None) -> None:
        if extensions is None:
            return
        self._session.set_enabled_extensions(extensions)
        self._refresh_all()

    def _rescan(self) -> None:
        scan_result = scan_media_files(self._session.root)
        self.call_from_thread(self._handle_rescan_done, scan_result)

    def _handle_rescan_done(self, scan_result: ScanResult) -> None:
        self._session.replace_scan(scan_result)
        self.notify(f"Found {len(scan_result.files):,} media files")
        self._refresh_all()
        self._start_fingerprint_indexing()

    # ------------------------------------------------------ fingerprint index

    def _start_fingerprint_indexing(self) -> None:
        pending_files = self._session.find_files_missing_fingerprints()
        if not pending_files:
            self._indexing_progress = None
            return
        self._indexing_progress = IndexingProgress(done_count=0, total_count=len(pending_files))
        self.run_worker(
            functools.partial(self._index_fingerprints, pending_files),
            thread=True,
            exclusive=True,
            group="fingerprints",
        )

    def _index_fingerprints(self, pending_files: list[MediaFile]) -> None:
        """Runs in a worker thread with its own SQLite connection."""
        worker = get_current_worker()
        store = RatingStore(self._database_path)
        try:
            with ThreadPoolExecutor(self.FINGERPRINT_WORKER_COUNT, thread_name_prefix="fingerprint") as executor:
                signatures = executor.map(
                    compute_image_signature, [self._session.resolve_path(media_file) for media_file in pending_files]
                )
                batch: list[Fingerprint] = []
                for done_count, (media_file, signature) in enumerate(
                    zip(pending_files, signatures, strict=True), start=1
                ):
                    if worker.is_cancelled:
                        executor.shutdown(cancel_futures=True)
                        return
                    batch.append(
                        Fingerprint(
                            relative_path=media_file.relative_path,
                            size_bytes=media_file.size_bytes,
                            modified_timestamp_utc=media_file.modified_timestamp_utc,
                            difference_hash=signature.difference_hash,
                            capture_timestamp=signature.capture_timestamp,
                        )
                    )
                    if len(batch) >= self.FINGERPRINT_BATCH_SIZE or done_count == len(pending_files):
                        store.save_fingerprints(batch)
                        self.call_from_thread(
                            self._handle_fingerprints_computed,
                            batch,
                            IndexingProgress(done_count=done_count, total_count=len(pending_files)),
                        )
                        batch = []
        finally:
            store.close()
        self.call_from_thread(self._handle_indexing_done)

    def _handle_fingerprints_computed(self, fingerprints: list[Fingerprint], progress: IndexingProgress) -> None:
        self._session.add_fingerprints(fingerprints)
        self._indexing_progress = progress
        self._refresh_sidebar()

    def _handle_indexing_done(self) -> None:
        self._indexing_progress = None
        self._session.rebuild_duplicate_groups()
        if self._session.is_duplicates_mode:
            self.notify(self._describe_duplicates_state(), timeout=3)
            self._refresh_all()
        else:
            self._refresh_sidebar()

    def _describe_order(self) -> str:
        if self._session.is_duplicates_mode:
            return " · duplicates"
        return f" · sort: {self._session.sort_order.value}"

    def _describe_duplicates_state(self) -> str:
        description = f"{self._session.duplicate_group_count:,} duplicate groups"
        if self._indexing_progress is not None:
            description += (
                f" so far (indexed {self._indexing_progress.done_count:,} / {self._indexing_progress.total_count:,})"
            )
        return description

    # ----------------------------------------------------------------- helpers

    def _move_to_next_unrated(self) -> None:
        if not self._session.move_to_next_unrated():
            self.notify("No unrated files in this view")

    def _navigate(self, move: Callable[[], object]) -> None:
        previous_index = self._session.current_index
        move()
        if self._session.current_index != previous_index:
            self._refresh_all()

    def _apply_rating(self, rating: int | None) -> None:
        if self._session.rate_current(rating) is None:
            return
        if self._is_auto_advance and rating is not None:
            self._session.move_by(1)
        self._refresh_all()

    def _apply_filter(self, rating_filter: RatingFilter) -> None:
        self._session.set_rating_filter(rating_filter)
        self.notify(f"View: {rating_filter.name} ({self._session.visible_count:,})", timeout=2)
        self._refresh_all()

    # --------------------------------------------------------------- rendering

    def _refresh_all(self) -> None:
        if self._is_grid_mode:
            self._refresh_grid()
        else:
            self.run_worker(self._show_current_preview(), exclusive=True, group="preview")
        self._refresh_sidebar()
        self._refresh_statusbar()

    def _refresh_statusbar(self) -> None:
        current_file = self._session.current_file
        if current_file is None:
            self.query_one("#statusbar", Static).update(Text("no files in this view", style="dim"))
            return
        section_position = self._session.current_section_position
        section_text = (
            ""
            if section_position is None
            else f"§ {section_position.section.label}  {section_position.member_number}/"
            f"{section_position.section.size}  ({section_position.section_number}/{section_position.section_count})"
        )
        status_text = Text.assemble(
            build_rating_text(self._session.current_rating),
            "   ",
            (f"{self._session.current_index + 1:,} / {self._session.visible_count:,}", "bold"),
            "   ",
            (section_text, "magenta"),
            "   ",
            (current_file.relative_path, ""),
            "   ",
            (f"view: {self._session.rating_filter.name}", "cyan"),
            (self._describe_order(), "cyan"),
        )
        self.query_one("#statusbar", Static).update(status_text)

    def _refresh_sidebar(self) -> None:
        self.query_one("#info", Static).update(self._build_info_text())
        self.query_one("#sections", Static).update(self._build_sections_text())
        self.query_one("#counts", Static).update(self._build_counts_text())
        self.query_one("#settings", Static).update(self._build_settings_text())

    def _build_info_text(self) -> Text:
        current_file = self._session.current_file
        if current_file is None:
            if self._session.is_duplicates_mode:
                return Text(f"No duplicate groups in this view.\n{self._describe_duplicates_state()}", style="dim")
            return Text("No files match.\nv / f: change view\ne: change formats", style="dim")
        folder, _, file_name = current_file.relative_path.rpartition("/")
        info_text = Text.assemble(
            (file_name, "bold"),
            "\n",
            (folder or ".", "dim"),
            "\n\n",
            build_rating_text(self._session.current_rating),
            "\n",
        )
        preview = self._current_preview
        if preview is not None and preview.original_width_px is not None:
            info_text.append(f"{preview.original_width_px}×{preview.original_height_px} · ")  # noqa: RUF001
        info_text.append(format_size(current_file.size_bytes))
        if preview is not None and preview.captured_at:
            info_text.append(f"\ntaken {preview.captured_at}")
        else:
            modified_at = datetime.fromtimestamp(current_file.modified_timestamp_utc).astimezone()
            info_text.append(f"\nmodified {modified_at:%Y-%m-%d %H:%M}", style="dim")
        return info_text

    def _build_sections_text(self) -> Text:
        """Window of the section list around the current section."""
        sections = self._session.sections
        section_position = self._session.current_section_position
        if section_position is None:
            return Text("")
        title = (
            "Duplicate groups" if self._session.is_duplicates_mode else f"Sections ({self._session.sort_order.value})"
        )
        sections_text = Text.assemble((title, "bold"), (f"  {len(sections):,}  J/K\n", "dim"))
        current_offset = section_position.section_number - 1
        window_start = max(
            0,
            min(current_offset - self.SECTION_LIST_MAX_LINES // 2, len(sections) - self.SECTION_LIST_MAX_LINES),
        )
        window_end = min(len(sections), window_start + self.SECTION_LIST_MAX_LINES)
        if window_start > 0:
            sections_text.append(f"  … {window_start:,} more\n", style="dim")
        for offset in range(window_start, window_end):
            section = sections[offset]
            label = truncate_from_left(section.label, max_length=self.SECTION_LABEL_WIDTH)
            line = f"{label:<{self.SECTION_LABEL_WIDTH}} {section.size:>6,}"
            if offset == current_offset:
                sections_text.append(f"▶ {line}\n", style="bold reverse")
            else:
                sections_text.append(f"  {line}\n")
        if window_end < len(sections):
            sections_text.append(f"  … {len(sections) - window_end:,} more\n", style="dim")
        sections_text.rstrip()
        return sections_text

    def _build_counts_text(self) -> Text:
        rating_counts = self._session.count_ratings()
        counts_text = Text("Ratings (enabled formats)\n", style="bold")
        for bucket in reversed(ALL_BUCKETS[1:]):
            counts_text.append(f"  {describe_bucket(bucket):<9} {rating_counts.get(bucket, 0):>7,}\n")
        counts_text.append(f"  {describe_bucket(None):<9} {rating_counts.get(None, 0):>7,}", style="dim")
        return counts_text

    def _build_settings_text(self) -> Text:
        if self._indexing_progress is not None:
            duplicates_text = f"indexing {self._indexing_progress.done_count:,}/{self._indexing_progress.total_count:,}"
        else:
            duplicates_text = f"{self._session.duplicate_group_count:,} groups"
        return Text.assemble(
            ("View  ", "bold"),
            (self._session.rating_filter.name, "cyan"),
            ("\nSort  ", "bold"),
            "duplicate groups" if self._session.is_duplicates_mode else self._session.sort_order.value,
            ("\nDuplicates  ", "bold"),
            duplicates_text,
            ("\nFormats  ", "bold"),
            " ".join(sorted(self._session.enabled_extensions)) or "(none)",
            ("\nAuto-advance  ", "bold"),
            "on" if self._is_auto_advance else "off",
            ("\n\ndb: " + shorten_home_path(self._database_path), "dim"),
            ("\n? for help", "dim"),
        )

    def _compute_preview_edge_px(self) -> int:
        viewer_size = self.query_one("#viewer").content_size
        cell_size = get_cell_size()
        longest_edge_px = max(viewer_size.width * cell_size.width, viewer_size.height * cell_size.height)
        rounded_edge_px = math.ceil(longest_edge_px / self.PREVIEW_EDGE_STEP_PX) * self.PREVIEW_EDGE_STEP_PX
        return max(self.PREVIEW_MIN_EDGE_PX, min(self.PREVIEW_MAX_EDGE_PX, rounded_edge_px))

    async def _show_current_preview(self) -> None:
        current_file = self._session.current_file
        preview_widget = self.query_one("#preview", ImageWidget)
        placeholder = self.query_one("#placeholder", Static)
        if current_file is None:
            self._current_preview = None
            preview_widget.image = None
            preview_widget.display = False
            placeholder.display = True
            placeholder.update("No files in this view — press v / f to change the view, e for formats")
            return

        max_edge_px = self._compute_preview_edge_px()
        preview_future = self._preview_cache.request(self._session.resolve_path(current_file), max_edge_px=max_edge_px)
        self._prefetch_neighbours(max_edge_px=max_edge_px)
        if not preview_future.done():
            placeholder.update("loading…")
        preview = await self._preview_cache.get(self._session.resolve_path(current_file), max_edge_px=max_edge_px)
        if self._session.current_file != current_file:
            return
        self._current_preview = preview
        self._display_preview(preview)
        self._refresh_sidebar()

    def _display_preview(self, preview: Preview) -> None:
        preview_widget = self.query_one("#preview", ImageWidget)
        placeholder = self.query_one("#placeholder", Static)
        if preview.image is None:
            preview_widget.image = None
            preview_widget.display = False
            placeholder.display = True
            placeholder.update(preview.error or "no preview")
            return
        viewer_size = self.query_one("#viewer").content_size
        cell_size = get_cell_size()
        cell_box = fit_image_to_cells(
            image_width_px=preview.image.width,
            image_height_px=preview.image.height,
            max_columns=max(1, viewer_size.width),
            max_rows=max(1, viewer_size.height),
            cell_width_px=cell_size.width,
            cell_height_px=cell_size.height,
        )
        placeholder.display = False
        preview_widget.display = True
        preview_widget.styles.width = cell_box.columns
        preview_widget.styles.height = cell_box.rows
        preview_widget.image = preview.image

    def _prefetch_neighbours(self, *, max_edge_px: int) -> None:
        visible_files = self._session.visible_files
        current_index = self._session.current_index
        neighbour_indices = [current_index + offset for offset in range(1, self.PREFETCH_AHEAD_COUNT + 1)]
        neighbour_indices.append(current_index - 1)
        for neighbour_index in neighbour_indices:
            if 0 <= neighbour_index < len(visible_files):
                self._preview_cache.request(
                    self._session.resolve_path(visible_files[neighbour_index]), max_edge_px=max_edge_px
                )

    # -------------------------------------------------------------------- grid

    def _refresh_grid(self) -> None:
        grid = self.query_one("#grid", Container)
        grid_size = grid.content_size
        if grid_size.width == 0 or grid_size.height == 0:
            self.call_after_refresh(self._refresh_grid)
            return
        if not self._session.visible_files:
            self._shown_grid_page = None
            grid.remove_children()
            grid.mount(Static(Text("No files in this view", style="dim"), classes="section-header"))
            return
        layout_key = GridLayoutKey(
            view_version=self._session.view_version,
            column_count=max(1, grid_size.width // self.THUMBNAIL_WIDTH_CELLS),
            available_height=grid_size.height,
        )
        if layout_key != self._grid_layout_key:
            self._grid_layout_key = layout_key
            self._grid_rows = build_grid_rows(self._session.sections, column_count=layout_key.column_count)
            self._grid_pages = paginate_grid_rows(
                self._grid_rows,
                available_height=layout_key.available_height,
                header_height=self.SECTION_HEADER_HEIGHT_CELLS,
                thumbnail_row_height=self.THUMBNAIL_HEIGHT_CELLS,
            )
            self._shown_grid_page = None
        page = self._grid_pages[find_page_index(self._grid_pages, self._session.current_index)]
        if page is not self._shown_grid_page:
            self._shown_grid_page = page
            grid.remove_children()
            grid.mount_all(self._build_grid_page_widgets(page, column_count=layout_key.column_count))
            self.call_after_refresh(self._fill_grid)
            return
        self._fill_grid()

    def _build_grid_page_widgets(self, page: GridPage, *, column_count: int) -> list[Widget]:
        widgets: list[Widget] = []
        for row in page.rows:
            if row.is_header:
                widgets.append(Static(self._build_section_header_text(row), classes="section-header"))
                continue
            file_indices = list(range(row.start_index, row.end_index))
            padding = [None] * (column_count - len(file_indices))
            widgets.append(
                Horizontal(
                    *(ThumbnailCell(file_index=file_index) for file_index in [*file_indices, *padding]),
                    classes="thumbnail-row",
                )
            )
        return widgets

    def _build_section_header_text(self, row: GridRow) -> Text:
        section_number = self._session.sections.index(row.section) + 1
        return Text.assemble(
            (row.section.label, "bold"),
            (f"   {row.section.size:,} {'file' if row.section.size == 1 else 'files'}", ""),
            (f"   section {section_number}/{len(self._session.sections)}", "dim"),
        )

    def _fill_grid(self) -> None:
        grid = self.query_one("#grid", Container)
        visible_files = self._session.visible_files
        cell_columns = max(1, grid.content_size.width // max(1, self._grid_layout_key.column_count) - 2)
        cell_rows = self.THUMBNAIL_HEIGHT_CELLS - 3  # border + rating line
        for cell in grid.query(ThumbnailCell):
            if cell.file_index is None or cell.file_index >= len(visible_files):
                continue
            media_file = visible_files[cell.file_index]
            cell.set_class(cell.file_index == self._session.current_index, "current")
            cell.query_one(".thumbnail-label", Label).update(build_rating_text(self._session.get_rating(media_file)))
            if cell.relative_path != media_file.relative_path:
                cell.relative_path = media_file.relative_path
                self.run_worker(
                    self._load_thumbnail(
                        cell=cell, media_file=media_file, max_columns=cell_columns, max_rows=cell_rows
                    ),
                    group="thumbnails",
                )

    async def _load_thumbnail(
        self, *, cell: ThumbnailCell, media_file: MediaFile, max_columns: int, max_rows: int
    ) -> None:
        preview = await self._preview_cache.get(
            self._session.resolve_path(media_file), max_edge_px=self.THUMBNAIL_EDGE_PX
        )
        if cell.relative_path != media_file.relative_path:
            return
        thumbnail_widget = cell.query_one(ImageWidget)
        if preview.image is None:
            cell.query_one(".thumbnail-label", Label).update(Text(f"{media_file.extension} (no preview)", style="dim"))
            return
        cell_size = get_cell_size()
        cell_box = fit_image_to_cells(
            image_width_px=preview.image.width,
            image_height_px=preview.image.height,
            max_columns=max_columns,
            max_rows=max_rows,
            cell_width_px=cell_size.width,
            cell_height_px=cell_size.height,
        )
        thumbnail_widget.styles.width = cell_box.columns
        thumbnail_widget.styles.height = cell_box.rows
        thumbnail_widget.image = preview.image
