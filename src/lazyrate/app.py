"""Textual TUI for rating images."""

from __future__ import annotations

import functools
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, Label, ProgressBar, Static
from textual.worker import get_current_worker
from textual_image.widget import Image as ImageWidget
from textual_image.widget import get_cell_size

from lazyrate.export import ExportOptions, ExportSummary, export_files
from lazyrate.fingerprints import Fingerprint, compute_image_signature
from lazyrate.geo_map import (
    BROWSER_MAP_FILE_NAME,
    Basemap,
    BrowserMapWriter,
    MapViewport,
    build_town_labels,
    find_points_near,
)
from lazyrate.geo_search import (
    SearchResult,
    find_place_match,
    parse_radius_query,
    rank_within_place,
    search_by_place,
    search_by_radius,
)
from lazyrate.grid_layout import (
    GridPage,
    GridRow,
    build_grid_rows,
    find_index_in_adjacent_row,
    find_page_index,
    paginate_grid_rows,
)
from lazyrate.learning import (
    build_current_embedding_matrix,
    compute_embeddings,
    find_files_missing_embeddings,
    train_and_predict,
)
from lazyrate.locations import LocationReader, PlaceIndex
from lazyrate.media import scan_media_files
from lazyrate.preview import PreviewCache, read_photo_details
from lazyrate.ratings import ALL_BUCKETS, FILTER_PRESETS, MAX_STARS, REJECTED, describe_bucket, format_stars
from lazyrate.screens import (
    ExportRequest,
    ExportScreen,
    ExtensionsScreen,
    HelpScreen,
    RatingFilterScreen,
    SearchScreen,
)
from lazyrate.session import SortOrder
from lazyrate.store import RatingStore
from lazyrate.suggestions import RatingModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from PIL import Image
    from textual import events
    from textual.binding import BindingType
    from textual.geometry import Size
    from textual.timer import Timer
    from textual.widget import Widget

    from lazyrate.embeddings import EmbeddingMatrix, ImageEmbedder
    from lazyrate.geo_map import MapPoint, RenderedMap
    from lazyrate.geo_search import PlaceMatch
    from lazyrate.learning import LearningResult
    from lazyrate.locations import GeoCoordinates, PhotoLocation
    from lazyrate.media import MediaFile, ScanResult
    from lazyrate.preview import PhotoDetails, Preview
    from lazyrate.ratings import RatingFilter
    from lazyrate.session import RatingSession
    from lazyrate.suggestions import Suggestion


BYTES_PER_KILOBYTE = 1024
CLIPBOARD_COMMANDS = (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"])


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


def find_clipboard_command() -> list[str] | None:
    """First installed clipboard tool (macOS, Wayland, X11), or None."""
    for command in CLIPBOARD_COMMANDS:
        if shutil.which(command[0]) is not None:
            return list(command)
    return None


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


def build_suggestion_text(suggestion: Suggestion) -> Text:
    """Ghost rating for an unrated photo, e.g. ``★★★★? 80%`` or ``junk? 92%``."""
    confidence_text = f" {suggestion.confidence:.0%}"
    suggested_rating = suggestion.suggested_rating
    if suggested_rating is None:
        return Text(f"keeper?{confidence_text}", style="italic dim")
    if suggested_rating == REJECTED:
        return Text(f"junk?{confidence_text}", style="italic red")
    return Text(f"{'★' * suggested_rating}?{confidence_text}", style="italic dim yellow")


class SidePaneMode(StrEnum):
    """What the right pane shows besides the metadata (persisted per database)."""

    INFO = "info"
    PREVIEW = "preview"
    """Large preview of the highlighted image (grid view only)."""
    MAP = "map"
    """Offline map of the photos in the view (or the search matches), the current one marked."""


@dataclass(frozen=True, slots=True)
class SideMapKey:
    """Everything the base map depends on; while it is unchanged only the current-photo marker is redrawn."""

    view_version: int
    search_query: str | None
    located_count: int
    width_px: int
    height_px: int
    viewport: MapViewport


@dataclass(frozen=True, slots=True)
class PanePosition:
    """A point in a pane as 0..1 of its width and height, from the top left."""

    x_ratio: float
    y_ratio: float


class MapPane(Container):
    """The right pane's image box. While it shows the map: the mouse wheel zooms, dragging pans, a click selects the
    photo under the mouse (or centers the map there)."""

    class ZoomRequested(Message):
        def __init__(self, *, steps: int, anchor: PanePosition) -> None:
            super().__init__()
            self.steps = steps
            self.anchor = anchor

    class DragRequested(Message):
        """The mouse moved by ``x_ratio_delta`` / ``y_ratio_delta`` (shares of the pane) with the button held."""

        def __init__(self, *, x_ratio_delta: float, y_ratio_delta: float) -> None:
            super().__init__()
            self.x_ratio_delta = x_ratio_delta
            self.y_ratio_delta = y_ratio_delta

    class Clicked(Message):
        def __init__(self, *, position: PanePosition) -> None:
            super().__init__()
            self.position = position

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._drag_start: PanePosition | None = None
        self._drag_last: PanePosition | None = None
        self._is_dragging = False

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._request_zoom(event, steps=1)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._request_zoom(event, steps=-1)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        event.stop()
        self._drag_start = self._drag_last = self._locate(event)
        self._is_dragging = False
        self.capture_mouse()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._drag_start is None or self._drag_last is None:
            return
        position = self._locate(event)
        if not self._is_dragging and position == self._drag_start:
            return
        self._is_dragging = True
        self.post_message(
            self.DragRequested(
                x_ratio_delta=position.x_ratio - self._drag_last.x_ratio,
                y_ratio_delta=position.y_ratio - self._drag_last.y_ratio,
            )
        )
        self._drag_last = position

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._drag_start is None:
            return
        event.stop()
        self.release_mouse()
        if not self._is_dragging:
            self.post_message(self.Clicked(position=self._drag_start))
        self._drag_start = self._drag_last = None
        self._is_dragging = False

    def _request_zoom(self, event: events.MouseEvent, *, steps: int) -> None:
        event.stop()
        self.post_message(self.ZoomRequested(steps=steps, anchor=self._locate(event)))

    def _locate(self, event: events.MouseEvent) -> PanePosition:
        """Mouse position in the pane, at the middle of the cell under the mouse (outside 0..1 while dragging out)."""
        region = self.content_region
        return PanePosition(
            x_ratio=(event.screen_x - region.x + 0.5) / max(region.width, 1),
            y_ratio=(event.screen_y - region.y + 0.5) / max(region.height, 1),
        )


class TaskProgressRow(Horizontal):
    """One running background task: title plus progress bar (indeterminate while ``total`` is None)."""

    def __init__(self, *, title: str, total: int | None) -> None:
        super().__init__(classes="progress-row")
        self._title = title
        self._total = total

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="progress-title", markup=False)
        yield ProgressBar(total=self._total, show_eta=True, classes="progress-bar")

    def set_progress(self, *, done_count: int, total_count: int) -> None:
        # A fast task can report before the row's children are composed: skip that update, the next one lands.
        title_label = next(iter(self.query(".progress-title").results(Label)), None)
        progress_bar = next(iter(self.query(ProgressBar)), None)
        if title_label is None or progress_bar is None:
            return
        title_label.update(f"{self._title}  {done_count:,}/{total_count:,}")
        progress_bar.update(total=total_count, progress=done_count)


class SectionHeader(Static):
    """Separator line above a section in the grid; re-rendered after every rating to keep its counts live."""

    def __init__(self, header_text: Text, *, row: GridRow) -> None:
        super().__init__(header_text, classes="section-header")
        self.row = row


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

    def find_image_widget(self) -> ImageWidget | None:
        """None while the cell is not (or no longer) mounted, e.g. after a page switch during fast scrolling."""
        if not self.is_attached:
            return None
        return next(iter(self.query(ImageWidget)), None)

    def find_label(self) -> Label | None:
        if not self.is_attached:
            return None
        return next(iter(self.query(".thumbnail-label").results(Label)), None)


class ImageRaterApp(App[None]):
    """Rate thousands of photos fast; review any rating bucket; persist everything in SQLite."""

    TITLE = "lazyrate"
    DEFAULT_SORT_ORDER = SortOrder.RATING
    EXPORT_OPTIONS_SETTING_KEY = "export_options"
    SIDE_PANE_SETTING_KEY = "side_pane"
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
    RETRAIN_AFTER_RATING_COUNT = 10
    """Retrain the suggestion model in the background after this many rating changes."""
    PROGRESS_REPORT_STEPS = 100
    MAP_CONTROLS_HINT = "Map: wheel or + / - zoom · drag or shift+arrows pan · click a dot selects · * fits all"
    MAP_CLICK_RADIUS_CELLS = 1.0
    """A click this close to a dot (in cell widths) selects its photo."""
    PHOTO_DETAILS_CACHE_SIZE = 4096
    """Threads report progress at most this many times per task (each report crosses to the UI thread)."""

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
    #side-preview { height: 1fr; align: center middle; display: none; }
    #side-preview-image { width: auto; height: auto; }
    #sidebar.-preview { width: 50%; }
    #sidebar.-preview #side-preview { display: block; }
    #sidebar.-preview #info { height: auto; margin: 1 0 0 0; }
    #sidebar.-preview #sections, #sidebar.-preview #counts, #sidebar.-preview #settings { display: none; }
    #statusbar { height: 1; padding: 0 1; background: $panel; }
    #progress { height: auto; background: $panel; }
    .progress-row { height: 1; padding: 0 1; }
    .progress-title { width: 48; color: $text-muted; }
    .progress-bar { width: 1fr; }
    .dialog { width: 64; height: auto; max-height: 90%; padding: 1 2; background: $surface; border: thick $accent; }
    .dialog SelectionList { height: auto; max-height: 24; margin: 1 0; }
    .dialog .buttons { height: auto; align-horizontal: right; }
    .dialog Button { margin-left: 1; }
    .help { width: 110; max-width: 96%; }
    ExtensionsScreen, RatingFilterScreen, ExportScreen, SearchScreen, HelpScreen { align: center middle; }
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
        Binding("home,ctrl+up,ctrl+k", "move_to_start", "First", show=False),
        Binding("end,ctrl+down,ctrl+j", "move_to_end", "Last", show=False),
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
        Binding("y", "accept_suggestion", "Accept AI"),
        Binding("slash", "search", "Search", key_display="/"),
        Binding("u", "clear_rating", "Unrate", show=False),
        Binding("z", "undo", "Undo"),
        Binding("n", "next_unrated", "Next unrated"),
        Binding("v", "cycle_filter(1)", "View"),
        Binding("V", "cycle_filter(-1)", "View back", show=False),
        Binding("f", "choose_filter", "Filter"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("S", "refresh_view", "Re-sort"),
        Binding("g", "toggle_grid", "Grid"),
        Binding("d", "toggle_duplicates", "Dupes"),
        Binding("b", "keep_best_of_group", "Best of group"),
        Binding("enter", "toggle_full_view", "Full view", show=False),
        Binding("escape", "escape", "Back", show=False),
        Binding("Y", "accept_section_suggestions", "Accept section", show=False),
        Binding("B", "keep_suggested_best", "Keep AI best", show=False),
        Binding("L", "retrain", "Retrain", show=False),
        Binding("E", "export", "Export"),
        Binding("e", "choose_extensions", "Formats"),
        Binding("o", "open_external", "Open ext"),
        Binding("c", "copy_path", "Copy path"),
        Binding("p", "toggle_side_pane('preview')", "Big preview"),
        Binding("m", "toggle_side_pane('map')", "Map"),
        Binding("M", "open_browser_map", "Browser map", show=False),
        Binding("plus,equals_sign", "zoom_map(1)", "Zoom in", show=False),
        Binding("minus", "zoom_map(-1)", "Zoom out", show=False),
        Binding("asterisk", "fit_map", "Fit map", show=False),
        Binding("shift+left", "pan_map(-1, 0)", "Pan west", show=False),
        Binding("shift+right", "pan_map(1, 0)", "Pan east", show=False),
        Binding("shift+up", "pan_map(0, -1)", "Pan north", show=False),
        Binding("shift+down", "pan_map(0, 1)", "Pan south", show=False),
        Binding("a", "toggle_auto_advance", "Auto-adv", show=False),
        Binding("r", "rescan", "Rescan", show=False),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        session: RatingSession,
        database_path: Path,
        embedder_factory: Callable[[], ImageEmbedder] | None = None,
    ) -> None:
        """``embedder_factory`` enables suggestions and search; None (e.g. without ``lazyrate[ml]``) disables them."""
        super().__init__()
        self._session = session
        self._database_path = database_path
        self._embedder_factory = embedder_factory
        self._embedder: ImageEmbedder | None = None
        self._embedding_matrix: EmbeddingMatrix | None = None
        self._learning_result: LearningResult | None = None
        self._learning_error: str | None = None
        self._is_training = False
        self._rating_changes_since_training = 0
        self._progress_rows: dict[str, TaskProgressRow] = {}
        self._indexing_progress: IndexingProgress | None = None
        self._preview_cache = PreviewCache()
        self._place_index = PlaceIndex()
        self._is_auto_advance = True
        self._is_grid_mode = True
        self._is_full_view = False
        self._side_pane_mode = SidePaneMode.INFO
        self._basemap = Basemap()
        self._side_map_key: SideMapKey | None = None
        self._side_map_base: RenderedMap | None = None
        self._side_map_points: list[MapPoint] = []
        """The dots of the shown map, for clicking on them."""
        self._map_viewport = MapViewport()
        self._map_subject: tuple[str | None, str] | None = None
        """Search and rating view the map shows; zoom and pan reset when it changes."""
        self._grid_layout_key: GridLayoutKey | None = None
        self._grid_rows: list[GridRow] = []
        self._grid_pages: list[GridPage] = []
        self._shown_grid_page: GridPage | None = None
        self._view_refresh_timer: Timer | None = None
        self._photo_details: dict[str, PhotoDetails | None] = {}
        """Header metadata per relative path (``None``: none readable), filled on demand for the info panel."""
        self._pending_details_path: str | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with Container(id="viewer"):
                yield ImageWidget(id="preview")
                yield Static("", id="placeholder")
            yield Container(id="grid")
            with Vertical(id="sidebar"):
                with MapPane(id="side-preview"):
                    yield ImageWidget(id="side-preview-image")
                yield Static(id="info")
                yield Static(id="sections")
                yield Static(id="counts")
                yield Static(id="settings")
        yield Vertical(id="progress")
        yield Static(id="statusbar")
        yield Footer()

    def on_mount(self) -> None:
        self._session.set_sort_order(self.DEFAULT_SORT_ORDER)
        self._session.move_to_first_unrated()
        saved_side_pane_mode = self._session.load_setting(self.SIDE_PANE_SETTING_KEY)
        if saved_side_pane_mode in set(SidePaneMode):
            self._side_pane_mode = SidePaneMode(saved_side_pane_mode)
        self._apply_view_mode()
        self._refresh_all()
        self._start_fingerprint_indexing()
        self._start_location_reading()
        self._start_learning_pipeline()

    def on_unmount(self) -> None:
        self._preview_cache.shutdown()

    def on_resize(self) -> None:
        if self._view_refresh_timer is not None:
            self._view_refresh_timer.stop()
        self._view_refresh_timer = self.set_timer(self.RESIZE_DEBOUNCE_SECONDS, self._refresh_all)

    # ----------------------------------------------------------------- actions

    def action_move(self, delta: int) -> None:
        self._navigate(lambda: self._session.move_by(delta))

    def action_move_row(self, direction: int) -> None:
        if not (self._is_grid_shown and self._grid_rows):
            self._navigate(lambda: self._session.move_by(direction))
            return
        target_index = find_index_in_adjacent_row(
            self._grid_rows, file_index=self._session.current_index, direction=direction
        )
        self._navigate(lambda: self._session.move_to(target_index))

    def action_move_page(self, direction: int) -> None:
        if not (self._is_grid_shown and self._grid_pages):
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
        self._note_rating_changes(len(changes))
        if len(changes) == 1:
            self.notify(
                f"Undone: {changes[0].relative_path} → {format_stars(changes[0].previous_rating)}", markup=False
            )
        else:
            self.notify(f"Undone: {len(changes)} rating changes")
        self._refresh_all()

    def action_cycle_sort(self) -> None:
        if self._session.is_duplicates_mode:
            self.notify("Duplicate groups are always ordered by time (press d to leave)")
            return
        sort_orders = self._session.available_sort_orders
        sort_index = sort_orders.index(self._session.sort_order) if self._session.sort_order in sort_orders else -1
        sort_order = sort_orders[(sort_index + 1) % len(sort_orders)]
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
        self._note_rating_changes(len(changes))
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

    def action_toggle_side_pane(self, mode_name: str) -> None:
        """``p``: large preview <-> metadata; ``m``: map <-> metadata."""
        requested_mode = SidePaneMode(mode_name)
        self._side_pane_mode = SidePaneMode.INFO if self._side_pane_mode is requested_mode else requested_mode
        self._session.save_setting(key=self.SIDE_PANE_SETTING_KEY, value=self._side_pane_mode.value)
        if self._side_pane_mode is SidePaneMode.PREVIEW and not self._is_grid_mode:
            self.notify("The large preview in the right pane is shown in grid view (g)", timeout=3)
        if self._side_pane_mode is SidePaneMode.MAP:
            self.notify(self.MAP_CONTROLS_HINT, timeout=4)
        self._apply_view_mode()
        self.call_after_refresh(self._refresh_all)

    def action_zoom_map(self, steps: int) -> None:
        self._change_map_viewport(self._map_viewport.zoom(steps=steps))

    def action_pan_map(self, x_steps: int, y_steps: int) -> None:
        self._change_map_viewport(self._map_viewport.pan(x_steps=x_steps, y_steps=y_steps))

    def action_fit_map(self) -> None:
        self._change_map_viewport(MapViewport())

    def on_map_pane_zoom_requested(self, message: MapPane.ZoomRequested) -> None:
        self._change_map_viewport(
            self._map_viewport.zoom(
                steps=message.steps, anchor_x_ratio=message.anchor.x_ratio, anchor_y_ratio=message.anchor.y_ratio
            )
        )

    def on_map_pane_drag_requested(self, message: MapPane.DragRequested) -> None:
        self._change_map_viewport(
            self._map_viewport.drag(x_ratio_delta=message.x_ratio_delta, y_ratio_delta=message.y_ratio_delta)
        )

    def on_map_pane_clicked(self, message: MapPane.Clicked) -> None:
        """A click on a dot selects its photo (repeated clicks cycle through photos taken at the same spot); a click
        elsewhere centers the map there."""
        if not self._is_side_map_shown or self._side_map_base is None:
            return
        cell_size = get_cell_size()
        clicked_paths = find_points_near(
            self._side_map_points,
            extent=self._side_map_base.extent,
            image_size=self._side_map_base.image.size,
            x_ratio=message.position.x_ratio,
            y_ratio=message.position.y_ratio,
            max_distance_px=self.MAP_CLICK_RADIUS_CELLS * cell_size.width,
        )
        if not clicked_paths:
            self._change_map_viewport(
                self._map_viewport.center_on(x_ratio=message.position.x_ratio, y_ratio=message.position.y_ratio)
            )
            return
        current_file = self._session.current_file
        current_path = None if current_file is None else current_file.relative_path
        next_offset = clicked_paths.index(current_path) + 1 if current_path in clicked_paths else 0
        selected_path = clicked_paths[next_offset % len(clicked_paths)]
        if not self._session.move_to_file(selected_path):
            return
        if len(clicked_paths) > 1:
            position_text = f"{next_offset % len(clicked_paths) + 1}/{len(clicked_paths)}"
            self.notify(f"{len(clicked_paths)} photos here ({position_text}), click again for the next", timeout=2)
        self._refresh_all()

    def action_open_browser_map(self) -> None:
        """Write the photos of the view (or the search matches) to an HTML map and open it in the browser."""
        map_points = self._session.collect_map_points()
        if not map_points:
            self.notify("No photos with GPS data in this view", timeout=3)
            return
        output_path = self._database_path.parent / BROWSER_MAP_FILE_NAME
        title = f"{self._session.search_query or self._session.rating_filter.name} · {len(map_points):,} photos"
        self._start_progress("browser-map", title="Building map", total=len(map_points))
        self.run_worker(
            functools.partial(self._write_browser_map, map_points, output_path, title),
            thread=True,
            exclusive=True,
            group="browser-map",
        )

    def action_toggle_grid(self) -> None:
        self._is_grid_mode = not self._is_grid_mode
        self._is_full_view = False
        self._apply_view_mode()
        self.call_after_refresh(self._refresh_all)

    def action_refresh_view(self) -> None:
        self._session.refresh_view()
        self.notify(f"Re-sorted ({self._session.visible_count:,} files)", timeout=2)
        self._refresh_all()

    def action_toggle_full_view(self) -> None:
        """Enter: the current image alone on the screen; enter / escape returns to the grid or single view."""
        self._is_full_view = not self._is_full_view
        self._apply_view_mode()
        self.call_after_refresh(self._refresh_all)

    def action_escape(self) -> None:
        """Leave full view first, then an active search."""
        if self._is_full_view:
            self.action_toggle_full_view()
            return
        if self._session.search_query is not None:
            self._session.clear_search()
            self.notify("Search ended", timeout=2)
            self._refresh_all()

    def action_accept_suggestion(self) -> None:
        current_file = self._session.current_file
        if current_file is not None and self._session.get_rating(current_file) is not None:
            self.notify("Already rated", timeout=2)
            return
        if self._session.accept_suggestion_for_current() is None:
            self.notify(self._describe_missing_suggestion(), timeout=3)
            return
        self._note_rating_changes(1)
        if self._is_auto_advance:
            self._session.move_by(1)
        self._refresh_all()

    def action_accept_section_suggestions(self) -> None:
        changes = self._session.accept_suggestions_in_current_section()
        if not changes:
            self.notify("No unrated photos with suggestions in this section", timeout=3)
            return
        self._note_rating_changes(len(changes))
        self.notify(f"Accepted {len(changes):,} suggestions — z to undo", timeout=3)
        self._refresh_all()

    def action_keep_suggested_best(self) -> None:
        section_position = self._session.current_section_position
        if not self._session.is_duplicates_mode or section_position is None:
            self.notify("Only in duplicates mode (press d)")
            return
        best_index = self._session.find_suggested_best_index(section_position.section)
        if best_index is None:
            self.notify("No suggestions for this group yet", timeout=3)
            return
        self._session.move_to(best_index)
        self.action_keep_best_of_group()

    def action_search(self) -> None:
        self.push_screen(
            SearchScreen(current_query=self._session.search_query, is_text_search_available=self._is_ai_search_ready),
            self._handle_search_query,
        )

    def action_retrain(self) -> None:
        if self._embedding_matrix is None:
            self.notify(self._describe_ai_unavailable(), timeout=5)
            return
        if self._is_training:
            self.notify("Already learning", timeout=2)
            return
        self._start_training()

    def action_open_external(self) -> None:
        current_file = self._session.current_file
        if current_file is None:
            return
        self._open_with_default_app(self._session.resolve_path(current_file))

    def action_copy_path(self) -> None:
        """Copy the absolute path via the system clipboard tool; OSC 52 (terminal clipboard) as a fallback."""
        current_file = self._session.current_file
        if current_file is None:
            return
        full_path = str(self._session.resolve_path(current_file).resolve())
        clipboard_command = find_clipboard_command()
        is_copied = False
        if clipboard_command is not None:
            try:
                subprocess.run(clipboard_command, input=full_path, text=True, check=True, timeout=5)
                is_copied = True
            except (OSError, subprocess.SubprocessError):
                is_copied = False
        if not is_copied:
            self.copy_to_clipboard(full_path)
        self.notify(f"Copied {full_path}", timeout=3, markup=False)

    def action_toggle_auto_advance(self) -> None:
        self._is_auto_advance = not self._is_auto_advance
        self.notify(f"Auto-advance {'on' if self._is_auto_advance else 'off'}")
        self._refresh_sidebar()

    def action_rescan(self) -> None:
        self._start_progress("scan", title="Rescanning folder", total=None)
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

    def _handle_search_query(self, query: str | None) -> None:
        """Coordinates -> radius search; a known place name -> place search (plus text ranking); else text search."""
        if query is None:
            return
        if not query:
            self._session.clear_search()
            self._refresh_all()
            return
        try:
            radius_query = parse_radius_query(query)
        except ValueError as error:
            self.notify(str(error), severity="warning", markup=False)
            return
        locations = self._session.locations
        if radius_query is not None:
            self._apply_search_result(query, search_by_radius(query=radius_query, locations=locations.values()))
            return
        place_match = find_place_match(query_text=query, locations=locations)
        if place_match is not None and (not place_match.remaining_text or not self._is_ai_search_ready):
            if place_match.remaining_text:
                self.notify(f"Text search needs AI; ignoring '{place_match.remaining_text}'", timeout=4, markup=False)
            self._apply_search_result(query, search_by_place(match=place_match, locations=locations))
            return
        if not self._is_ai_search_ready:
            self.notify(self._describe_unmatched_search(query), timeout=6, markup=False)
            return
        text_query = query if place_match is None else place_match.remaining_text
        self._start_progress("search", title=f"Searching: {query}", total=None)
        self.run_worker(
            functools.partial(self._run_text_search, query, text_query, place_match),
            thread=True,
            exclusive=True,
            group="search",
        )

    def _run_text_search(self, query: str, text_query: str, place_match: PlaceMatch | None) -> None:
        if self._embedder is None or self._embedding_matrix is None:
            return
        text_scores = self._embedding_matrix.rank_by_similarity(self._embedder.embed_text(text_query))
        if place_match is None:
            result = SearchResult(scores=text_scores, section_labels={}, summary="Best matches first")
        else:
            result = rank_within_place(match=place_match, text_scores=text_scores)
        self.call_from_thread(self._handle_text_search_done, query, result)

    def _handle_text_search_done(self, query: str, result: SearchResult) -> None:
        self._finish_progress("search")
        self._apply_search_result(query, result)

    def _apply_search_result(self, query: str, result: SearchResult) -> None:
        if not result.scores:
            self.notify(f"No matches: {result.summary}", severity="warning", timeout=4, markup=False)
            return
        self._session.set_search(query=query, scores=result.scores, section_labels=result.section_labels)
        self.notify(f"{result.summary} · esc ends the search", timeout=4, markup=False)
        self._refresh_all()

    def _describe_unmatched_search(self, query: str) -> str:
        if self._session.located_count == 0:
            location_hint = "no photo here has GPS data"
        else:
            location_hint = f"no photo was taken in a town, region or country called '{query}'"
        return f"{location_hint}; searching by content needs AI ({self._describe_ai_unavailable()})"

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
        self._start_progress("export", title=f"Exporting {len(relative_paths):,} files", total=len(relative_paths))
        self.run_worker(
            functools.partial(self._run_export, relative_paths, request.options),
            thread=True,
            group="export",
        )

    def _run_export(self, relative_paths: list[str], options: ExportOptions) -> None:
        try:
            summary = export_files(
                root=self._session.root,
                relative_paths=relative_paths,
                options=options,
                on_progress=functools.partial(self._report_progress_from_thread, "export"),
            )
        except (OSError, ValueError) as error:
            self.call_from_thread(self._finish_progress, "export")
            self.call_from_thread(self.notify, f"Export failed: {error}", severity="error", timeout=10, markup=False)
            return
        self.call_from_thread(self._handle_export_done, summary, options)

    def _handle_export_done(self, summary: ExportSummary, options: ExportOptions) -> None:
        self._finish_progress("export")
        message = f"Exported {summary.exported_count:,} files to {shorten_home_path(options.destination)}"
        if summary.skipped_existing_count:
            message += f" ({summary.skipped_existing_count:,} already there)"
        if summary.missing_count:
            message += f" ({summary.missing_count:,} missing on disk)"
        self.notify(message, timeout=8, markup=False)

    def _handle_extensions_chosen(self, extensions: set[str] | None) -> None:
        if extensions is None:
            return
        self._session.set_enabled_extensions(extensions)
        self._refresh_all()

    def _rescan(self) -> None:
        scan_result = scan_media_files(self._session.root)
        self.call_from_thread(self._handle_rescan_done, scan_result)

    def _handle_rescan_done(self, scan_result: ScanResult) -> None:
        self._finish_progress("scan")
        self._session.replace_scan(scan_result)
        self.notify(f"Found {len(scan_result.files):,} media files")
        self._refresh_all()
        self._start_fingerprint_indexing()
        self._start_location_reading()
        self._start_learning_pipeline()

    # ------------------------------------------------------ fingerprint index

    def _start_fingerprint_indexing(self) -> None:
        pending_files = self._session.find_files_missing_fingerprints()
        if not pending_files:
            self._indexing_progress = None
            return
        self._indexing_progress = IndexingProgress(done_count=0, total_count=len(pending_files))
        self._start_progress("fingerprints", title="Finding near duplicates", total=len(pending_files))
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
        self._update_progress("fingerprints", done_count=progress.done_count, total_count=progress.total_count)
        self._refresh_sidebar()

    def _handle_indexing_done(self) -> None:
        self._indexing_progress = None
        self._finish_progress("fingerprints")
        self._session.rebuild_duplicate_groups()
        if self._session.is_duplicates_mode:
            self.notify(self._describe_duplicates_state(), timeout=3)
            self._refresh_all()
        else:
            self._refresh_sidebar()

    # ------------------------------------------------------- GPS locations

    def _start_location_reading(self) -> None:
        pending_files = self._session.find_files_missing_locations()
        if not pending_files:
            return
        self._start_progress("locations", title="Reading GPS locations", total=len(pending_files))
        self.run_worker(
            functools.partial(self._read_locations, pending_files),
            thread=True,
            exclusive=True,
            group="locations",
        )

    def _read_locations(self, pending_files: list[MediaFile]) -> None:
        """Runs in a worker thread with its own SQLite connection."""
        worker = get_current_worker()
        store = RatingStore(self._database_path)
        reader = LocationReader(root=self._session.root, place_index=self._place_index)
        try:
            reader.read_locations(
                pending_files,
                on_batch=functools.partial(self._save_location_batch, store, len(pending_files)),
                should_stop=lambda: worker.is_cancelled,
            )
        finally:
            store.close()
        if not worker.is_cancelled:
            self.call_from_thread(self._handle_locations_done)

    def _save_location_batch(
        self, store: RatingStore, total_count: int, locations: list[PhotoLocation], done_count: int
    ) -> None:
        store.save_locations(locations)
        self.call_from_thread(self._handle_locations_read, locations, done_count, total_count)

    def _handle_locations_read(self, locations: list[PhotoLocation], done_count: int, total_count: int) -> None:
        self._session.add_locations(locations)
        self._update_progress("locations", done_count=done_count, total_count=total_count)

    def _handle_locations_done(self) -> None:
        self._finish_progress("locations")
        self._refresh_sidebar()
        if self._is_side_map_shown:
            self._show_side_map()

    # ---------------------------------------------------- suggestions (AI)

    def _start_learning_pipeline(self) -> None:
        if self._embedder_factory is None:
            return
        self.run_worker(self._run_learning_pipeline, thread=True, exclusive=True, group="learning")

    def _run_learning_pipeline(self) -> None:
        """Worker thread: load the model once, embed new / changed photos, then hand the matrix to the UI."""
        worker = get_current_worker()
        embedder = self._load_embedder()
        if embedder is None:
            return
        store = RatingStore(self._database_path)
        try:
            embeddings = store.load_embeddings(embedder.model_name)
            pending_files = find_files_missing_embeddings(files=self._session.all_files, embeddings=embeddings)
            if pending_files:
                self.call_from_thread(
                    self._start_progress, "embeddings", title="Reading image content (AI)", total=len(pending_files)
                )
                compute_embeddings(
                    root=self._session.root,
                    files=pending_files,
                    embedder=embedder,
                    store=store,
                    on_progress=functools.partial(self._report_progress_from_thread, "embeddings"),
                    should_stop=lambda: worker.is_cancelled,
                )
                self.call_from_thread(self._finish_progress, "embeddings")
                embeddings = store.load_embeddings(embedder.model_name)
        finally:
            store.close()
        if worker.is_cancelled:
            return
        embedding_matrix = build_current_embedding_matrix(files=self._session.all_files, embeddings=embeddings)
        self.call_from_thread(self._handle_embeddings_ready, embedding_matrix)

    def _load_embedder(self) -> ImageEmbedder | None:
        """Worker thread. Loading CLIP can fail in many ways (download, torch backend), so report them all."""
        if self._embedder is not None or self._embedder_factory is None:
            return self._embedder
        self.call_from_thread(
            self._start_progress, "model", title="Loading AI model (first run downloads ~600 MB)", total=None
        )
        try:
            self._embedder = self._embedder_factory()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.call_from_thread(self._handle_learning_failed, f"AI model unavailable: {error}")
        finally:
            self.call_from_thread(self._finish_progress, "model")
        return self._embedder

    def _handle_learning_failed(self, message: str) -> None:
        self._learning_error = message
        self.notify(message, severity="warning", timeout=8, markup=False)
        self._refresh_sidebar()

    def _handle_embeddings_ready(self, embedding_matrix: EmbeddingMatrix) -> None:
        self._embedding_matrix = embedding_matrix
        self._start_training()

    def _note_rating_changes(self, change_count: int) -> None:
        self._rating_changes_since_training += change_count
        if self._rating_changes_since_training >= self.RETRAIN_AFTER_RATING_COUNT:
            self._start_training()

    def _start_training(self) -> None:
        if self._embedding_matrix is None or self._is_training:
            return
        self._is_training = True
        self._rating_changes_since_training = 0
        self._start_progress("training", title="Learning from your ratings", total=None)
        self.run_worker(
            functools.partial(self._train, self._embedding_matrix, self._session.ratings),
            thread=True,
            group="training",
        )

    def _train(self, embedding_matrix: EmbeddingMatrix, ratings: dict[str, int]) -> None:
        self.call_from_thread(
            self._handle_learning_result, train_and_predict(embedding_matrix=embedding_matrix, ratings=ratings)
        )

    def _handle_learning_result(self, result: LearningResult) -> None:
        self._is_training = False
        self._finish_progress("training")
        had_suggestions = self._session.suggestion_count > 0
        self._learning_result = result
        self._session.set_suggestions(result.suggestions)
        if result.suggestions and not had_suggestions:
            self.notify("AI suggestions ready: s → sort 'suggested rating', y accepts, Y accepts a section", timeout=8)
        self._refresh_all()
        if self._rating_changes_since_training >= self.RETRAIN_AFTER_RATING_COUNT:
            self._start_training()

    def _describe_ai_unavailable(self) -> str:
        if self._embedder_factory is None:
            return "AI features are off (install with: uv tool install 'lazyrate[ml]' …)"
        if self._learning_error is not None:
            return self._learning_error
        return "AI is still reading the photos — see the progress bar"

    def _describe_missing_suggestion(self) -> str:
        if self._learning_result is None:
            return self._describe_ai_unavailable()
        if not self._learning_result.readiness.is_junk_model_ready:
            return "Not enough ratings yet to suggest (see AI in the sidebar)"
        return "No star suggestion yet for this photo (needs more starred photos)"

    def _describe_ai_state(self) -> str:
        if self._embedder_factory is None:
            return "off"
        if self._learning_error is not None:
            return "unavailable"
        result = self._learning_result
        if result is None:
            return "reading photos…"
        readiness = result.readiness
        if not readiness.is_junk_model_ready:
            missing_rejected = readiness.missing_rejected_count
            missing_starred = max(0, RatingModel.MIN_EXAMPLES_PER_SIDE - readiness.starred_count)
            return f"learning: rate {missing_rejected} more rejects, {missing_starred} more starred"
        stars_state = (
            "stars ✓" if readiness.is_star_model_ready else f"stars: {readiness.missing_starred_count} more starred"
        )
        state = f"junk ✓ · {stars_state}"
        evaluation = result.evaluation
        if evaluation is not None and evaluation.star_within_one_accuracy is not None:
            state += f" · ±1★ {evaluation.star_within_one_accuracy:.0%}"
        return state

    # ----------------------------------------------------------- progress panel

    def _start_progress(self, task_key: str, *, title: str, total: int | None) -> None:
        self._finish_progress(task_key)
        progress_panel = next(iter(self.query("#progress")), None)
        if progress_panel is None:
            return
        progress_row = TaskProgressRow(title=title, total=total)
        self._progress_rows[task_key] = progress_row
        progress_panel.mount(progress_row)

    def _update_progress(self, task_key: str, *, done_count: int, total_count: int) -> None:
        progress_row = self._progress_rows.get(task_key)
        if progress_row is not None:
            progress_row.set_progress(done_count=done_count, total_count=total_count)

    def _finish_progress(self, task_key: str) -> None:
        progress_row = self._progress_rows.pop(task_key, None)
        if progress_row is not None and progress_row.is_attached:
            progress_row.remove()

    def _report_progress_from_thread(self, task_key: str, done_count: int, total_count: int) -> None:
        """Worker threads: forward progress to the UI, throttled to about PROGRESS_REPORT_STEPS updates per task."""
        report_step = max(1, total_count // self.PROGRESS_REPORT_STEPS)
        if done_count % report_step == 0 or done_count == total_count:
            self.call_from_thread(self._update_progress, task_key, done_count=done_count, total_count=total_count)

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

    @property
    def _is_side_preview_shown(self) -> bool:
        """In single view the main area already shows the image large, so the pane keeps the metadata."""
        return self._side_pane_mode is SidePaneMode.PREVIEW and self._is_grid_shown

    @property
    def _is_side_map_shown(self) -> bool:
        return self._side_pane_mode is SidePaneMode.MAP and not self._is_full_view

    @property
    def _is_grid_shown(self) -> bool:
        """Full view temporarily replaces the grid; ``_is_grid_mode`` remembers where to return to."""
        return self._is_grid_mode and not self._is_full_view

    @property
    def _is_ai_search_ready(self) -> bool:
        """Searching by content needs the CLIP model and every photo described once."""
        return self._embedder is not None and self._embedding_matrix is not None

    def _apply_view_mode(self) -> None:
        self.query_one("#viewer").display = not self._is_grid_shown
        self.query_one("#grid").display = self._is_grid_shown
        self.query_one("#sidebar").display = not self._is_full_view
        self.query_one(Footer).display = not self._is_full_view
        self.query_one("#sidebar").set_class(self._is_side_preview_shown or self._is_side_map_shown, "-preview")

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
        self._note_rating_changes(1)
        if self._is_auto_advance and rating is not None:
            self._session.move_by(1)
        self._refresh_all()

    def _apply_filter(self, rating_filter: RatingFilter) -> None:
        self._session.set_rating_filter(rating_filter)
        self.notify(f"View: {rating_filter.name} ({self._session.visible_count:,})", timeout=2)
        self._refresh_all()

    # --------------------------------------------------------------- rendering

    def _refresh_all(self) -> None:
        if self._find_grid() is None:
            return
        if self._is_grid_shown:
            self._refresh_grid()
        else:
            self.run_worker(self._show_current_preview(), exclusive=True, group="preview")
        if self._is_side_preview_shown:
            self.run_worker(self._show_side_preview(), exclusive=True, group="side-preview")
        if self._is_side_map_shown:
            self._show_side_map()
        self._refresh_sidebar()
        self._refresh_statusbar()

    def _refresh_statusbar(self) -> None:
        if self._find_grid() is None:
            return
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
            self._build_suggestion_hint(current_file),
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
        if self._find_grid() is None:
            return
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
            self._build_suggestion_hint(current_file),
            "\n",
        )
        if self._is_suggested_best_of_group(self._session.current_index):
            info_text.append("suggested best of this group (B keeps it)\n", style="green")
        details = self._find_photo_details(current_file)
        if details is not None:
            info_text.append(f"{details.width_px}×{details.height_px} · ")  # noqa: RUF001
        info_text.append(format_size(current_file.size_bytes))
        if details is not None and details.captured_at:
            info_text.append(f"\ntaken {details.captured_at}")
        else:
            modified_at = datetime.fromtimestamp(current_file.modified_timestamp_utc).astimezone()
            info_text.append(f"\nmodified {modified_at:%Y-%m-%d %H:%M}", style="dim")
        if details is not None and details.camera:
            info_text.append(f"\n{details.camera}")
        if details is not None and details.lens:
            info_text.append(f"\n{details.lens}", style="dim")
        location = self._session.get_location(current_file)
        if location is not None and location.place is not None and location.coordinates is not None:
            place = location.place
            info_text.append(f"\n{', '.join(name for name in (place.town, place.region, place.country) if name)}")
            info_text.append(f"\n{location.coordinates}", style="dim")
        return info_text

    def _find_photo_details(self, media_file: MediaFile) -> PhotoDetails | None:
        """Cached details, or ``None`` while they are read in the background (the info panel refreshes then)."""
        relative_path = media_file.relative_path
        if relative_path in self._photo_details:
            return self._photo_details[relative_path]
        if self._pending_details_path != relative_path:
            self._pending_details_path = relative_path
            self.run_worker(
                functools.partial(self._read_photo_details, relative_path, self._session.resolve_path(media_file)),
                thread=True,
                exclusive=True,
                group="photo-details",
            )
        return None

    def _read_photo_details(self, relative_path: str, path: Path) -> None:
        details = read_photo_details(path)
        if not get_current_worker().is_cancelled:
            self.call_from_thread(self._store_photo_details, relative_path, details)

    def _store_photo_details(self, relative_path: str, details: PhotoDetails | None) -> None:
        if len(self._photo_details) >= self.PHOTO_DETAILS_CACHE_SIZE:
            del self._photo_details[next(iter(self._photo_details))]
        self._photo_details[relative_path] = details
        if self._pending_details_path == relative_path:
            self._pending_details_path = None
        current_file = self._session.current_file
        if current_file is not None and current_file.relative_path == relative_path:
            self._refresh_sidebar()

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
            section_counts = self._session.count_section_members(section)
            pending_marker = " ↻" if section_counts.is_pending_resort else ""
            line = f"{label:<{self.SECTION_LABEL_WIDTH}} {section_counts.member_count:>6,}{pending_marker}"
            if offset == current_offset:
                sections_text.append(f"▶ {line}\n", style="bold reverse")
            else:
                sections_text.append(f"  {line}\n")
        if window_end < len(sections):
            sections_text.append(f"  … {len(sections) - window_end:,} more\n", style="dim")
        if any(
            self._session.count_section_members(section).is_pending_resort
            for section in sections[window_start:window_end]
        ):
            sections_text.append("  ↻ re-rated, S re-sorts\n", style="yellow")
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
            ("\nAI  ", "bold"),
            self._describe_ai_state(),
            ("" if self._session.search_query is None else "\nSearch  "),
            ("" if self._session.search_query is None else self._session.search_query, "cyan"),
            ("\n\ndb: " + shorten_home_path(self._database_path), "dim"),
            ("\n? for help", "dim"),
        )

    def _build_suggestion_hint(self, media_file: MediaFile) -> Text:
        """``  suggested ★★★★? 80%`` for unrated photos with a suggestion, else empty."""
        suggestion = self._session.get_suggestion(media_file)
        if suggestion is None or self._session.get_rating(media_file) is not None:
            return Text("")
        return Text.assemble(("  suggested ", "dim"), build_suggestion_text(suggestion))

    def _is_suggested_best_of_group(self, file_index: int) -> bool:
        section_position = self._session.current_section_position
        if not self._session.is_duplicates_mode or section_position is None:
            return False
        return self._session.find_suggested_best_index(section_position.section) == file_index

    def _build_thumbnail_label(self, media_file: MediaFile, *, is_suggested_best: bool) -> Text:
        rating = self._session.get_rating(media_file)
        suggestion = self._session.get_suggestion(media_file)
        label_text = (
            build_suggestion_text(suggestion)
            if rating is None and suggestion is not None
            else build_rating_text(rating)
        )
        if is_suggested_best:
            label_text.append(" · best?", style="green")
        return label_text

    def _compute_preview_edge_px(self, container_size: Size) -> int:
        """Decode size for a preview filling ``container_size`` (cells), rounded so the cache can reuse it."""
        cell_size = get_cell_size()
        longest_edge_px = max(container_size.width * cell_size.width, container_size.height * cell_size.height)
        rounded_edge_px = math.ceil(longest_edge_px / self.PREVIEW_EDGE_STEP_PX) * self.PREVIEW_EDGE_STEP_PX
        return max(self.PREVIEW_MIN_EDGE_PX, min(self.PREVIEW_MAX_EDGE_PX, rounded_edge_px))

    async def _show_current_preview(self) -> None:
        current_file = self._session.current_file
        preview_widget = self.query_one("#preview", ImageWidget)
        placeholder = self.query_one("#placeholder", Static)
        if current_file is None:
            preview_widget.image = None
            preview_widget.display = False
            placeholder.display = True
            placeholder.update("No files in this view — press v / f to change the view, e for formats")
            return

        max_edge_px = self._compute_preview_edge_px(self.query_one("#viewer").content_size)
        preview_future = self._preview_cache.request(self._session.resolve_path(current_file), max_edge_px=max_edge_px)
        self._prefetch_neighbours(max_edge_px=max_edge_px)
        if not preview_future.done():
            placeholder.update("loading…")
        preview = await self._preview_cache.get(self._session.resolve_path(current_file), max_edge_px=max_edge_px)
        if self._session.current_file != current_file:
            return
        self._display_preview(preview)

    async def _show_side_preview(self) -> None:
        current_file = self._session.current_file
        image_widget = self.query_one("#side-preview-image", ImageWidget)
        box_size = self.query_one("#side-preview").content_size
        if current_file is None or box_size.width == 0 or box_size.height == 0:
            image_widget.image = None
            return
        max_edge_px = self._compute_preview_edge_px(box_size)
        self._prefetch_neighbours(max_edge_px=max_edge_px)
        preview = await self._preview_cache.get(self._session.resolve_path(current_file), max_edge_px=max_edge_px)
        if self._session.current_file != current_file or not image_widget.is_attached:
            return
        if not self._is_side_preview_shown:  # switched to the map while decoding
            return
        if preview.image is None:
            image_widget.image = None
            return
        cell_size = get_cell_size()
        cell_box = fit_image_to_cells(
            image_width_px=preview.image.width,
            image_height_px=preview.image.height,
            max_columns=box_size.width,
            max_rows=box_size.height,
            cell_width_px=cell_size.width,
            cell_height_px=cell_size.height,
        )
        image_widget.styles.width = cell_box.columns
        image_widget.styles.height = cell_box.rows
        image_widget.image = preview.image

    def _show_side_map(self) -> None:
        """Render the map pane in a thread; the base map is cached until the view, search or pane size changes."""
        box_size = self.query_one("#side-preview").content_size
        if box_size.width == 0 or box_size.height == 0:
            return
        map_subject = (self._session.search_query, self._session.rating_filter.name)
        if map_subject != self._map_subject:
            self._map_subject = map_subject
            self._map_viewport = MapViewport()
        cell_size = get_cell_size()
        map_key = SideMapKey(
            view_version=self._session.view_version,
            search_query=self._session.search_query,
            located_count=self._session.located_count,
            width_px=box_size.width * cell_size.width,
            height_px=box_size.height * cell_size.height,
            viewport=self._map_viewport,
        )
        current_file = self._session.current_file
        current_location = None if current_file is None else self._session.get_location(current_file)
        highlighted = None if current_location is None else current_location.coordinates
        # Collected here, on the UI thread: the worker must not read session state that rating may change.
        map_points = None if map_key == self._side_map_key else self._session.collect_map_points()
        if map_points is not None:
            self._side_map_points = map_points
        self.run_worker(
            functools.partial(self._render_side_map, map_key, map_points, highlighted, box_size),
            thread=True,
            exclusive=True,
            group="side-map",
        )

    def _render_side_map(
        self,
        map_key: SideMapKey,
        map_points: list[MapPoint] | None,
        highlighted: GeoCoordinates | None,
        box_size: Size,
    ) -> None:
        """``map_points`` is ``None`` when the cached base map still fits ``map_key``."""
        base_map = self._side_map_base
        if map_points is not None or base_map is None:
            map_points = map_points or []
            if not map_points and highlighted is not None:
                framed_points = [highlighted]
            else:
                framed_points = [point.coordinates for point in map_points]
            base_map = self._basemap.render_base(
                points=framed_points,
                labels=build_town_labels(map_points),
                width_px=map_key.width_px,
                height_px=map_key.height_px,
                viewport=map_key.viewport,
                notice=None if map_points else self._describe_empty_map(map_key),
            )
            self._side_map_key = map_key
            self._side_map_base = base_map
        if get_current_worker().is_cancelled:
            return
        map_image = self._basemap.draw_highlight(base_map, highlighted=highlighted)
        self.call_from_thread(self._display_side_map, map_image, box_size)

    def _change_map_viewport(self, viewport: MapViewport) -> None:
        if not self._is_side_map_shown or viewport == self._map_viewport:
            return
        self._map_viewport = viewport
        self._show_side_map()

    def _describe_empty_map(self, map_key: SideMapKey) -> str:
        if map_key.located_count == 0:
            return "No photo here has GPS data"
        if map_key.search_query is not None:
            return "No search match has GPS data"
        return "No photo in this view has GPS data"

    def _open_with_default_app(self, path: Path) -> None:
        opener_command = "open" if sys.platform == "darwin" else "xdg-open"
        try:
            subprocess.Popen([opener_command, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            self.notify(f"Cannot open: {error}", severity="error", markup=False)

    def _display_side_map(self, map_image: Image.Image, box_size: Size) -> None:
        image_widget = next(iter(self.query("#side-preview-image").results(ImageWidget)), None)
        if image_widget is None or not self._is_side_map_shown:
            return
        image_widget.styles.width = box_size.width
        image_widget.styles.height = box_size.height
        image_widget.image = map_image

    def _write_browser_map(self, map_points: list[MapPoint], output_path: Path, title: str) -> None:
        worker = get_current_worker()
        try:
            is_written = BrowserMapWriter(root=self._session.root).write(
                map_points,
                output_path=output_path,
                title=title,
                on_progress=functools.partial(self._report_progress_from_thread, "browser-map"),
                should_stop=lambda: worker.is_cancelled,
            )
        except OSError as error:
            self.call_from_thread(self._handle_browser_map_failed, str(error))
            return
        if is_written:
            self.call_from_thread(self._handle_browser_map_written, output_path, len(map_points))

    def _handle_browser_map_written(self, output_path: Path, point_count: int) -> None:
        self._finish_progress("browser-map")
        self._open_with_default_app(output_path)
        self.notify(
            f"Map of {point_count:,} photos opened in the browser ({shorten_home_path(output_path)}). "
            "Map tiles load from OpenStreetMap / Esri; photos stay on this computer.",
            timeout=6,
            markup=False,
        )

    def _handle_browser_map_failed(self, error_text: str) -> None:
        self._finish_progress("browser-map")
        self.notify(f"Cannot write the map: {error_text}", severity="error", markup=False)

    def _display_preview(self, preview: Preview) -> None:
        preview_widget = self.query_one("#preview", ImageWidget)
        placeholder = self.query_one("#placeholder", Static)
        if preview.image is None:
            preview_widget.image = None
            preview_widget.display = False
            placeholder.display = True
            placeholder.update(Text(preview.error or "no preview"))
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

    def _find_grid(self) -> Container | None:
        """None during shutdown: deferred refreshes (``call_after_refresh``) can fire after the screen is gone."""
        return next(iter(self.query("#grid").results(Container)), None)

    def _refresh_grid(self) -> None:
        grid = self._find_grid()
        if grid is None:
            return
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
            self.workers.cancel_group(self, "thumbnails")
            grid.remove_children()
            grid.mount_all(self._build_grid_page_widgets(page, column_count=layout_key.column_count))
            self.call_after_refresh(self._fill_grid)
            return
        self._fill_grid()

    def _build_grid_page_widgets(self, page: GridPage, *, column_count: int) -> list[Widget]:
        widgets: list[Widget] = []
        for row in page.rows:
            if row.is_header:
                widgets.append(SectionHeader(self._build_section_header_text(row), row=row))
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
        section_counts = self._session.count_section_members(row.section)
        member_count = section_counts.member_count
        header_text = Text.assemble(
            (row.section.label, "bold"),
            (f"   {member_count:,} {'file' if member_count == 1 else 'files'}", ""),
            (f"   section {section_number}/{len(self._session.sections)}", "dim"),
        )
        if section_counts.is_pending_resort:
            header_text.append(
                f"   ↻ {section_counts.moved_in_count:,} in, {section_counts.moved_out_count:,} out · S re-sorts",
                style="yellow",
            )
        return header_text

    def _fill_grid(self) -> None:
        grid = self._find_grid()
        if grid is None or self._grid_layout_key is None:
            return
        visible_files = self._session.visible_files
        cell_columns = max(1, grid.content_size.width // max(1, self._grid_layout_key.column_count) - 2)
        cell_rows = self.THUMBNAIL_HEIGHT_CELLS - 3  # border + rating line
        current_sections = set(self._session.sections)
        for header in grid.query(SectionHeader):
            # A header can outlive a re-sort until the grid is laid out again; that re-layout redraws it.
            if header.row.section in current_sections:
                header.update(self._build_section_header_text(header.row))
        suggested_best_indices = self._find_suggested_best_indices_on_page()
        for cell in grid.query(ThumbnailCell):
            if cell.file_index is None or cell.file_index >= len(visible_files):
                continue
            label = cell.find_label()
            if label is None:
                continue
            media_file = visible_files[cell.file_index]
            cell.set_class(cell.file_index == self._session.current_index, "current")
            label.update(
                self._build_thumbnail_label(media_file, is_suggested_best=cell.file_index in suggested_best_indices)
            )
            if cell.relative_path != media_file.relative_path:
                cell.relative_path = media_file.relative_path
                self.run_worker(
                    self._load_thumbnail(
                        cell=cell, media_file=media_file, max_columns=cell_columns, max_rows=cell_rows
                    ),
                    group="thumbnails",
                )

    def _find_suggested_best_indices_on_page(self) -> set[int]:
        if not self._session.is_duplicates_mode or self._shown_grid_page is None:
            return set()
        page_sections = {row.section for row in self._shown_grid_page.rows}
        return {
            best_index
            for section in page_sections
            if (best_index := self._session.find_suggested_best_index(section)) is not None
        }

    async def _load_thumbnail(
        self, *, cell: ThumbnailCell, media_file: MediaFile, max_columns: int, max_rows: int
    ) -> None:
        preview = await self._preview_cache.get(
            self._session.resolve_path(media_file), max_edge_px=self.THUMBNAIL_EDGE_PX
        )
        thumbnail_widget = cell.find_image_widget()
        label = cell.find_label()
        if cell.relative_path != media_file.relative_path or thumbnail_widget is None or label is None:
            return
        if preview.image is None:
            label.update(Text(f"{media_file.extension} (no preview)", style="dim"))
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
