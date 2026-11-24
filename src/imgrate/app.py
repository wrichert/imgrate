"""Textual TUI for rating images."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Label, Static
from textual_image.widget import Image as ImageWidget
from textual_image.widget import get_cell_size

from imgrate.media import scan_media_files
from imgrate.preview import PreviewCache
from imgrate.ratings import ALL_BUCKETS, FILTER_PRESETS, MAX_STARS, REJECTED, describe_bucket, format_stars
from imgrate.screens import ExtensionsScreen, HelpScreen, RatingFilterScreen

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.binding import BindingType
    from textual.timer import Timer

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


def build_rating_text(rating: int | None) -> Text:
    stars = format_stars(rating)
    if rating is None:
        return Text(stars, style="dim")
    if rating == REJECTED:
        return Text(stars, style="bold red")
    return Text(stars, style="bold yellow")


class ThumbnailCell(Vertical):
    """One tile of the grid: thumbnail plus rating line."""

    def __init__(self, *, slot_index: int) -> None:
        super().__init__(classes="thumbnail")
        self.slot_index = slot_index
        self.relative_path: str | None = None

    def compose(self) -> ComposeResult:
        with Container(classes="thumbnail-image-box"):
            yield ImageWidget(classes="thumbnail-image")
        yield Label("", classes="thumbnail-label")


class ImageRaterApp(App[None]):
    """Rate thousands of photos fast; review any rating bucket; persist everything in SQLite."""

    TITLE = "imgrate"
    PREVIEW_EDGE_STEP_PX = 256
    PREVIEW_MIN_EDGE_PX = 512
    PREVIEW_MAX_EDGE_PX = 1920
    THUMBNAIL_EDGE_PX = 384
    THUMBNAIL_WIDTH_CELLS = 28
    THUMBNAIL_HEIGHT_CELLS = 13
    PAGE_JUMP_SIZE = 10
    PREFETCH_AHEAD_COUNT = 3
    RESIZE_DEBOUNCE_SECONDS = 0.15

    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    #viewer { width: 1fr; height: 1fr; align: center middle; }
    #preview { width: auto; height: auto; }
    #placeholder { width: 100%; height: 100%; content-align: center middle; color: $text-muted; }
    #grid { width: 1fr; height: 1fr; layout: grid; display: none; }
    .thumbnail { border: round $panel; height: 100%; }
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
    ExtensionsScreen, RatingFilterScreen, HelpScreen { align: center middle; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("right,l,space", "move(1)", "Next", show=False),
        Binding("left,h", "move(-1)", "Prev", show=False),
        Binding("down,j", "move_row(1)", "Down", show=False),
        Binding("up,k", "move_row(-1)", "Up", show=False),
        Binding("pagedown", "move_page(1)", "Page down", show=False),
        Binding("pageup", "move_page(-1)", "Page up", show=False),
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
        Binding("g", "toggle_grid", "Grid"),
        Binding("enter", "open_from_grid", "Open", show=False),
        Binding("e", "choose_extensions", "Formats"),
        Binding("o", "open_external", "Open ext"),
        Binding("a", "toggle_auto_advance", "Auto-adv", show=False),
        Binding("r", "rescan", "Rescan", show=False),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, *, session: RatingSession, database_label: str) -> None:
        super().__init__()
        self._session = session
        self._database_label = database_label
        self._preview_cache = PreviewCache()
        self._is_auto_advance = True
        self._is_grid_mode = False
        self._grid_columns = 0
        self._grid_rows = 0
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
                yield Static(id="counts")
                yield Static(id="settings")
        yield Static(id="statusbar")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_all()

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
        step = self._grid_columns if self._is_grid_mode and self._grid_columns else 1
        self._navigate(lambda: self._session.move_by(direction * step))

    def action_move_page(self, direction: int) -> None:
        step = self._grid_page_size if self._is_grid_mode else self.PAGE_JUMP_SIZE
        self._navigate(lambda: self._session.move_by(direction * step))

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
        change = self._session.undo()
        if change is None:
            self.notify("Nothing to undo")
            return
        self.notify(f"Undone: {change.relative_path} → {format_stars(change.previous_rating)}")
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

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    # ---------------------------------------------------------------- handlers

    def _handle_filter_chosen(self, rating_filter: RatingFilter | None) -> None:
        if rating_filter is not None:
            self._apply_filter(rating_filter)

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

    # ----------------------------------------------------------------- helpers

    @property
    def _grid_page_size(self) -> int:
        return max(1, self._grid_columns * self._grid_rows)

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
        status_text = Text.assemble(
            build_rating_text(self._session.current_rating),
            "   ",
            (f"{self._session.current_index + 1:,} / {self._session.visible_count:,}", "bold"),
            "   ",
            (current_file.relative_path, ""),
            "   ",
            (f"view: {self._session.rating_filter.name}", "cyan"),
        )
        self.query_one("#statusbar", Static).update(status_text)

    def _refresh_sidebar(self) -> None:
        self.query_one("#info", Static).update(self._build_info_text())
        self.query_one("#counts", Static).update(self._build_counts_text())
        self.query_one("#settings", Static).update(self._build_settings_text())

    def _build_info_text(self) -> Text:
        current_file = self._session.current_file
        if current_file is None:
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

    def _build_counts_text(self) -> Text:
        rating_counts = self._session.count_ratings()
        counts_text = Text("Ratings (enabled formats)\n", style="bold")
        for bucket in reversed(ALL_BUCKETS[1:]):
            counts_text.append(f"  {describe_bucket(bucket):<9} {rating_counts.get(bucket, 0):>7,}\n")
        counts_text.append(f"  {describe_bucket(None):<9} {rating_counts.get(None, 0):>7,}", style="dim")
        return counts_text

    def _build_settings_text(self) -> Text:
        return Text.assemble(
            ("View  ", "bold"),
            (self._session.rating_filter.name, "cyan"),
            ("\nFormats  ", "bold"),
            " ".join(sorted(self._session.enabled_extensions)) or "(none)",
            ("\nAuto-advance  ", "bold"),
            "on" if self._is_auto_advance else "off",
            ("\n\n" + self._database_label, "dim"),
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
        columns = max(1, grid_size.width // self.THUMBNAIL_WIDTH_CELLS)
        rows = max(1, grid_size.height // self.THUMBNAIL_HEIGHT_CELLS)
        if (columns, rows) != (self._grid_columns, self._grid_rows):
            self._grid_columns, self._grid_rows = columns, rows
            grid.remove_children()
            grid.styles.grid_size_columns = columns
            grid.styles.grid_size_rows = rows
            grid.mount_all(ThumbnailCell(slot_index=slot_index) for slot_index in range(columns * rows))
            self.call_after_refresh(self._fill_grid)
            return
        self._fill_grid()

    def _fill_grid(self) -> None:
        grid = self.query_one("#grid", Container)
        visible_files = self._session.visible_files
        page_size = self._grid_page_size
        page_start_index = (self._session.current_index // page_size) * page_size
        cell_columns = max(1, grid.content_size.width // self._grid_columns - 2)
        cell_rows = max(1, grid.content_size.height // self._grid_rows - 3)
        for cell in grid.query(ThumbnailCell):
            file_index = page_start_index + cell.slot_index
            media_file = visible_files[file_index] if file_index < len(visible_files) else None
            cell.set_class(file_index == self._session.current_index, "current")
            label = cell.query_one(".thumbnail-label", Label)
            if media_file is None:
                cell.relative_path = None
                cell.query_one(ImageWidget).image = None
                label.update("")
                continue
            label.update(build_rating_text(self._session.get_rating(media_file)))
            if cell.relative_path != media_file.relative_path:
                cell.relative_path = media_file.relative_path
                cell.query_one(ImageWidget).image = None
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
