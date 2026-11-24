"""Modal screens: format selection, rating filter, help."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, SelectionList, Static
from textual.widgets.selection_list import Selection

from imgrate.media import IMAGE_EXTENSIONS, RAW_EXTENSIONS, VIDEO_EXTENSIONS
from imgrate.ratings import ALL_BUCKETS, RatingBucket, RatingFilter, build_custom_filter, describe_bucket

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
  home / end     first / last
  n              next unrated

[b]Review[/b]
  v / V          cycle rating view: all · unrated · not rejected · ≥★ … ★★★★★ · rejected
  f              choose rating view (any combination)
  g              toggle thumbnail grid
  enter          grid → open highlighted image

[b]Other[/b]
  e              choose file formats (e.g. exclude .mp4)
  o              open in system viewer
  a              toggle auto-advance
  r              rescan folder
  q              quit

Ratings are saved immediately to the SQLite file shown in the sidebar.
Export the keepers:  imgrate export ROOT DEST --min-rating 5
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
