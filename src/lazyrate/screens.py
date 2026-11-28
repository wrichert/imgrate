"""Modal screens: format selection, rating filter, export, search, help."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, OptionList, SelectionList, Static
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

from lazyrate.export import ExportOptions, validate_destination
from lazyrate.media import IMAGE_EXTENSIONS, PHOTO_EXTENSIONS, RAW_EXTENSIONS, VIDEO_EXTENSIONS
from lazyrate.ratings import ALL_BUCKETS, RatingBucket, RatingFilter, build_custom_filter, describe_bucket
from lazyrate.xmp import RatingFileMode

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.binding import BindingType


def describe_extension_kind(extension: str) -> str:
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in RAW_EXTENSIONS:
        return "raw"
    return "other"


class ExtensionsScreen(ModalScreen[set[str] | None]):
    """Pick which file formats take part in rating and review."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "apply", "Apply", priority=True),
        Binding("i", "select_images", "Images only"),
        Binding("a", "select_all", "All"),
        Binding("n", "select_none", "None"),
    ]

    def __init__(self, *, extension_counts: dict[str, int], enabled_extensions: frozenset[str]) -> None:
        super().__init__()
        self._extension_counts = extension_counts
        self._enabled_extensions = enabled_extensions

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("File formats  [dim](space toggles · i images only · a all · n none · enter applies)[/]")
            yield SelectionList[str](
                *(
                    Selection(
                        f"{extension:<7} {describe_extension_kind(extension):<6} {count:>7,}",
                        extension,
                        extension in self._enabled_extensions,
                    )
                    for extension, count in self._extension_counts.items()
                ),
                id="extension-list",
            )
            with Horizontal(classes="buttons"):
                yield Button("Apply", variant="primary", id="apply")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self.action_apply()
        else:
            self.action_cancel()

    def action_apply(self) -> None:
        self.dismiss(set(self.query_one(SelectionList).selected))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_select_images(self) -> None:
        selection_list = self.query_one(SelectionList)
        selection_list.deselect_all()
        for extension in self._extension_counts:
            if extension in PHOTO_EXTENSIONS:
                selection_list.select(extension)

    def action_select_all(self) -> None:
        self.query_one(SelectionList).select_all()

    def action_select_none(self) -> None:
        self.query_one(SelectionList).deselect_all()


class RatingFilesScreen(ModalScreen[RatingFileMode | None]):
    """Where ratings are stored besides lazyrate's database: XMP sidecars or inside the JPEGs."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]
    EXPLANATION = (
        "Lightroom, darktable, digiKam, Capture One and Bridge read XMP ratings (rejected = -1).\n"
        "Sidecars: DSC_1234.xmp next to the photo; an existing one only gets its rating changed.\n"
        "Inside JPEGs: travels with the file when it is renamed or moved; only the XMP block is rewritten,\n"
        "the picture stays byte for byte, the file date is kept. RAW and HEIC files always get sidecars.\n"
        "Switching on writes every existing rating once. Ratings found in files are imported for\n"
        "photos without a lazyrate rating."
    )

    def __init__(self, *, current_mode: RatingFileMode) -> None:
        super().__init__()
        self._current_mode = current_mode

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog rating-files"):
            yield Label("Store ratings  [dim](enter chooses · esc cancels)[/]")
            yield OptionList(
                *(
                    Option(
                        f"{'●' if mode is self._current_mode else '○'} {mode.value:<9} {mode.description}",
                        id=mode.value,
                    )
                    for mode in RatingFileMode
                )
            )
            yield Static(self.EXPLANATION, classes="hint")

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        option_list.highlighted = list(RatingFileMode).index(self._current_mode)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(RatingFileMode(event.option.id))

    def action_cancel(self) -> None:
        self.dismiss(None)


class RatingFilterScreen(ModalScreen[RatingFilter | None]):
    """Pick an arbitrary combination of rating buckets to review."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "apply", "Apply", priority=True),
    ]

    def __init__(self, *, rating_counts: dict[RatingBucket, int], current_filter: RatingFilter) -> None:
        super().__init__()
        self._rating_counts = rating_counts
        self._current_filter = current_filter

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("Show ratings  [dim](space toggles · enter applies)[/]")
            yield SelectionList[int](
                *(
                    Selection(
                        f"{describe_bucket(bucket):<9} {self._rating_counts.get(bucket, 0):>7,}",
                        index,
                        bucket in self._current_filter.buckets,
                    )
                    for index, bucket in enumerate(ALL_BUCKETS)
                ),
            )
            with Horizontal(classes="buttons"):
                yield Button("Apply", variant="primary", id="apply")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self.action_apply()
        else:
            self.action_cancel()

    def action_apply(self) -> None:
        # SelectionList values must be hashable and non-None, so buckets are addressed by index.
        selected_buckets = frozenset(ALL_BUCKETS[index] for index in self.query_one(SelectionList).selected)
        if not selected_buckets:
            self.notify("Select at least one rating", severity="warning")
            return
        self.dismiss(build_custom_filter(selected_buckets))

    def action_cancel(self) -> None:
        self.dismiss(None)


@dataclass(frozen=True, slots=True)
class ExportRequest:
    buckets: frozenset[RatingBucket]
    options: ExportOptions


class ExportScreen(ModalScreen[ExportRequest | None]):
    """Choose ratings, destination and copy mode; the export itself runs in the app."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "apply", "Export", priority=True),
    ]

    def __init__(
        self,
        *,
        root: Path,
        rating_counts: dict[RatingBucket, int],
        initial_buckets: frozenset[RatingBucket],
        initial_destination: str,
        is_flat: bool,
        is_symlink: bool,
    ) -> None:
        super().__init__()
        self._root = root
        self._rating_counts = rating_counts
        self._initial_buckets = initial_buckets
        self._initial_destination = initial_destination
        self._is_flat = is_flat
        self._is_symlink = is_symlink

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("Export  [dim](space toggles · tab moves · enter / ctrl+s exports · esc cancels)[/]")
            yield SelectionList[int](
                *(
                    Selection(
                        f"{describe_bucket(bucket):<9} {self._rating_counts.get(bucket, 0):>7,}",
                        index,
                        bucket in self._initial_buckets,
                    )
                    for index, bucket in reversed(list(enumerate(ALL_BUCKETS)))
                ),
                id="export-ratings",
            )
            yield Label("Destination folder")
            yield Input(self._initial_destination, id="export-destination")
            yield Checkbox("One flat folder (no sub folders)", self._is_flat, id="export-flat")
            yield Checkbox("Symlinks instead of copies", self._is_symlink, id="export-symlink")
            yield Static("", id="export-summary")
            with Horizontal(classes="buttons"):
                yield Button("Export", variant="primary", id="apply")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self._update_summary()

    def on_selection_list_selected_changed(self) -> None:
        self._update_summary()

    def on_input_submitted(self) -> None:
        self.action_apply()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self.action_apply()
        else:
            self.action_cancel()

    def action_apply(self) -> None:
        buckets = self._selected_buckets()
        if not buckets:
            self.notify("Select at least one rating", severity="warning")
            return
        destination_text = self.query_one("#export-destination", Input).value.strip()
        if not destination_text:
            self.notify("Enter a destination folder", severity="warning")
            return
        try:
            destination = validate_destination(root=self._root, destination=Path(destination_text))
        except ValueError as error:
            self.notify(str(error), severity="error", markup=False)
            return
        self.dismiss(
            ExportRequest(
                buckets=buckets,
                options=ExportOptions(
                    destination=destination,
                    is_flat=self.query_one("#export-flat", Checkbox).value,
                    is_symlink=self.query_one("#export-symlink", Checkbox).value,
                ),
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _selected_buckets(self) -> frozenset[RatingBucket]:
        return frozenset(ALL_BUCKETS[index] for index in self.query_one("#export-ratings", SelectionList).selected)

    def _update_summary(self) -> None:
        file_count = sum(self._rating_counts.get(bucket, 0) for bucket in self._selected_buckets())
        self.query_one("#export-summary", Static).update(f"[b]{file_count:,}[/b] files will be exported")


HELP_TEXT = """\
[b]Rating[/b]
  1 … 5          rate (auto-advances when auto-advance is on)
  0 / x          reject
  u              clear rating (unrated)
  z              undo last rating change

[b]Navigation[/b]
  → l space      next          ← h          previous
  ↓ / ↑          next / previous (grid: row down / up)
  i              100% view to check focus: arrows / hjkl pan, space next photo (same spot), i / esc back
  pgdn / pgup    jump 10 (grid: page)
  ctrl+↑ / ↓     first / last (also ctrl+k / j, home / end)
  n              next unrated
  J / K          next / previous section   (also ] . and \\[ ,)

[b]Review[/b]
  v / V          cycle rating view: all · unrated · not rejected · ≥★ … ★★★★★ · rejected
  f              choose rating view (any combination)
  s              cycle sort: rating (best first, default) · date taken · path · place (photos with GPS)
                 (+ suggested rating · most informative once AI suggestions exist) — also sets the sections
  S              re-sort / re-filter now (re-rated files move to their new section)
  g              toggle thumbnail grid (default) / single image
  p              grid: right pane shows the highlighted image large / metadata
  m              right pane shows a map of the photos in the view (or the search matches) / metadata
  + / -  *       map: zoom in / out (or mouse wheel) · fit all photos again
  shift+arrows   map: pan (or drag with the mouse); click a dot to select its photo, elsewhere to center there
  M              open that map in the browser (OpenStreetMap / satellite; thumbnails in the popups)
  enter          full view of the image; enter / esc returns to grid or single view
  RAW+JPEG       a RAW with a JPEG of the same name follows the JPEG: one photo, one rating, exported together

[b]Near duplicates[/b]  (fingerprints are computed in the background, once)
  d              toggle duplicates mode: groups of similar shots taken ≤ 15 min apart
  b              keep this one, reject the rest of its group, go to next group
  v / f          in duplicates mode: show groups with at least one matching member

[b]Places[/b]  (GPS data is read in the background, once; town names are looked up offline)
  /              search by place: madeira · funchal · portugal (town, region or country, any accents)
                 or by coordinates: @32.75,-16.95 50km · 32.75 -16.95 +-5km (default radius 10 km)
  esc            ends the search

[b]Filters[/b]  (add to any search, or use alone)
  camera:nikon · camera:phone · camera:"z 6" · lens:24-70 · 2024 · 2024-08 · 2024-08-15 · 2023..2024
  e.g. "madeira camera:nikon", "beach 2023", "camera:phone 2024-08" (alone: by month, oldest first)

[b]AI suggestions & search[/b]  (optional "ml" install; learns from your own ratings, nothing leaves the computer)
  y              accept the suggestion for this photo (ghost label, e.g. ★★★★? 80% or rejected? 92%)
  Y              accept the suggestions of every unrated photo in this section (one undo step)
  B              duplicates mode: keep the suggested best of the group, reject the rest
  /              search by content ("beach sunset", "receipt"); combine with a place: "beach madeira"
  L              retrain now (also happens automatically every 10 ratings)
  Suggestions appear after 10 rejected + 10 starred photos; stars after 30 starred photos.

[b]Other[/b]
  E              export: choose ratings, destination, flat / symlink
  X              store ratings also as XMP (Lightroom, darktable, …): off · sidecars · inside JPEGs
  e              choose file formats (e.g. exclude .mp4)
  o              open in system viewer
  c              copy full path to clipboard
  a              toggle auto-advance
  r              rescan folder
  q              quit

Ratings are saved immediately to the SQLite file shown in the sidebar; ratings stored in the files (camera,
Lightroom: XMP) are taken over for photos without a lazyrate rating. Photos moved or renamed inside the
folder (same size and file date) keep their rating. After 10 ratings the status bar shows your
pace and the time left for the unrated photos.
Export from the shell:  lazyrate export ROOT DEST --rating 5   (repeatable; or --min-rating 4)
"""


class SearchScreen(ModalScreen[str | None]):
    """Search by place, coordinates or (with AI) content. Returns the query; an empty query ends the search."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, *, current_query: str | None, is_text_search_available: bool) -> None:
        super().__init__()
        self._current_query = current_query
        self._is_text_search_available = is_text_search_available

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            if self._is_text_search_available:
                yield Label("Search photos  [dim](beach sunset · madeira · beach madeira · @32.75,-16.95 50km)[/]")
                placeholder = "describe the photo, name a place, or give lat,lon and a radius…"
            else:
                yield Label("Search photos by place  [dim](madeira · funchal · portugal · @32.75,-16.95 50km)[/]")
                placeholder = "town, region or country, or lat,lon and a radius…"
            yield Input(value=self._current_query or "", placeholder=placeholder, id="search-query")
            yield Label("[dim]enter searches · empty query ends the search · esc cancels[/]")
            if not self._is_text_search_available:
                yield Label('[dim]searching by content ("beach sunset") needs the AI extra[/]')

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


class HelpScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape,q,question_mark", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="dialog help"):
            yield Static(HELP_TEXT)

    def action_close(self) -> None:
        self.dismiss(None)
