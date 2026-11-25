"""Modal screens: format selection, rating filter, export, help."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, SelectionList, Static
from textual.widgets.selection_list import Selection

from lazyrate.export import ExportOptions, validate_destination
from lazyrate.media import IMAGE_EXTENSIONS, RAW_EXTENSIONS, VIDEO_EXTENSIONS
from lazyrate.ratings import ALL_BUCKETS, RatingBucket, RatingFilter, build_custom_filter, describe_bucket

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
            if extension in IMAGE_EXTENSIONS:
                selection_list.select(extension)

    def action_select_all(self) -> None:
        self.query_one(SelectionList).select_all()

    def action_select_none(self) -> None:
        self.query_one(SelectionList).deselect_all()


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
  pgdn / pgup    jump 10 (grid: page)
  ctrl+↑ / ↓     first / last (also ctrl+k / j, home / end)
  n              next unrated
  J / K          next / previous section   (also ] . and \\[ ,)

[b]Review[/b]
  v / V          cycle rating view: all · unrated · not rejected · ≥★ … ★★★★★ · rejected
  f              choose rating view (any combination)
  s              cycle sort: rating (best first, default) · date taken · path — also sets the sections
  S              re-sort / re-filter now (re-rated files move to their new section)
  g              toggle thumbnail grid (default) / single image
  p              grid: right pane shows the highlighted image large / metadata
  enter          full view of the image; enter / esc returns to grid or single view

[b]Near duplicates[/b]  (fingerprints are computed in the background, once)
  d              toggle duplicates mode: groups of similar shots taken ≤ 15 min apart
  b              keep this one, reject the rest of its group, go to next group
  v / f          in duplicates mode: show groups with at least one matching member

[b]Other[/b]
  E              export: choose ratings, destination, flat / symlink
  e              choose file formats (e.g. exclude .mp4)
  o              open in system viewer
  c              copy full path to clipboard
  a              toggle auto-advance
  r              rescan folder
  q              quit

Ratings are saved immediately to the SQLite file shown in the sidebar.
Export from the shell:  lazyrate export ROOT DEST --rating 5   (repeatable; or --min-rating 4)
"""


class HelpScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape,q,question_mark", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog help"):
            yield Static(HELP_TEXT)

    def action_close(self) -> None:
        self.dismiss(None)
