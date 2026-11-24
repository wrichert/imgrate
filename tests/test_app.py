from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Input, Static

from imgrate.app import ImageRaterApp, ThumbnailCell, fit_image_to_cells
from imgrate.media import scan_media_files
from imgrate.ratings import REJECTED
from imgrate.screens import ExportScreen
from imgrate.session import RatingSession, SortOrder
from imgrate.store import RatingStore

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_fit_image_to_cells_keeps_aspect_ratio() -> None:
    cell_box = fit_image_to_cells(
        image_width_px=4000,
        image_height_px=3000,
        max_columns=100,
        max_rows=50,
        cell_width_px=10,
        cell_height_px=20,
    )

    assert (cell_box.columns, cell_box.rows) == (100, 37)


async def test_rating_keys_store_ratings_and_advance(session: RatingSession, store: RatingStore) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("5", "3", "x")
        await pilot.press("z")
        await pilot.pause()

    assert store.load_ratings() == {"2023/a.jpg": 5, "2023/b.JPG": 3}
    assert session.current_file is not None
    assert session.current_file.relative_path == "2023/sub/c.png"


async def test_view_cycling_and_grid(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("4", "0")
        await pilot.press("v")  # all -> unrated
        assert session.visible_count == 3
        await pilot.pause(0.3)
        await pilot.press("right", "2")
        await pilot.pause(0.3)
        assert len(app.query(".thumbnail")) > 1

    assert session.get_rating(session.visible_files[1]) == 2
    assert session.count_ratings()[REJECTED] == 1


async def test_extension_dialog_enables_video(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("e")
        await pilot.press("a", "enter")
        await pilot.pause()

    assert ".mp4" in session.enabled_extensions
    assert session.visible_count == 6


async def test_background_indexing_then_duplicates_mode_and_keep_best(
    burst_root: Path, store: RatingStore, tmp_path: Path
) -> None:
    session = RatingSession(scan_result=scan_media_files(burst_root), store=store)
    app = ImageRaterApp(session=session, database_path=store.database_path)

    async with app.run_test(size=(140, 40)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert session.duplicate_group_count == 2
        await pilot.press("d")
        await pilot.pause(0.3)
        assert len([cell for cell in app.query(ThumbnailCell) if cell.relative_path]) == 5
        assert len(app.query(".section-header")) == 2
        await pilot.press("b")
        await pilot.pause()
        section_position = session.current_section_position
        assert section_position is not None
        assert section_position.section_number == 2

    ratings = store.load_ratings()
    assert ratings == {"burst/sunrise_2.jpg": REJECTED, "burst/sunrise_3.jpg": REJECTED}
    assert len(RatingStore(store.database_path).load_fingerprints()) == 7


async def test_sort_key_cycles(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        assert session.sort_order is SortOrder.RATING  # default
        await pilot.press("s")
        assert session.sort_order is SortOrder.DATE
        await pilot.press("s", "s")
        assert session.sort_order is SortOrder.RATING


async def test_shift_j_k_jump_sections_and_sidebar_lists_them(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("s", "s")  # rating -> date -> path
        assert session.sort_order is SortOrder.PATH
        await pilot.press("J")
        assert session.current_index == 2  # path sort: folders 2023 (2) | 2023/sub (1) | 2024 (2)
        await pilot.press("J", "K")
        assert session.current_index == 2
        sections_text = str(app.query_one("#sections", Static).render())
        assert "2023/sub" in sections_text
        assert "▶ 2023/sub" in sections_text


async def test_export_dialog_exports_selected_ratings_and_remembers_options(
    session: RatingSession, tmp_path: Path
) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")
    destination = tmp_path / "calendar"

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("5", "4", "5")  # a, b, c.png
        await pilot.press("E")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ExportScreen)
        screen.query_one("#export-destination", Input).value = str(destination)
        await pilot.press("ctrl+s")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert sorted(path.name for path in destination.iterdir()) == ["a.jpg", "c.png"]
    assert str(destination) in (session.load_setting(ImageRaterApp.EXPORT_OPTIONS_SETTING_KEY) or "")


async def test_export_dialog_refuses_destination_inside_photo_folder(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("5", "E")
        await pilot.pause()
        app.screen.query_one("#export-destination", Input).value = str(session.root / "picked")
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.screen, ExportScreen)

    assert not (session.root / "picked").exists()


async def test_starts_in_rating_sorted_grid_and_capital_s_resorts(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    # Tall enough for three sections (header + one thumbnail row each) on one grid page.
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause(0.3)
        assert app.query_one("#grid").display
        assert not app.query_one("#viewer").display
        assert [section.label for section in session.sections] == ["unrated"]

        await pilot.press("5", "x")
        assert [section.label for section in session.sections] == ["unrated"]  # stable while rating
        await pilot.press("S")
        await pilot.pause(0.3)
        assert [section.label for section in session.sections] == ["★★★★★", "unrated", "rejected"]
        assert len(app.query(".section-header")) == 3


async def test_c_copies_full_path_to_clipboard(session: RatingSession, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("imgrate.app.shutil.which", lambda _command: None)  # no clipboard tool -> OSC 52 fallback
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("c")
        await pilot.pause()
        expected_path = str(session.resolve_path(session.current_file).resolve())

    assert app.clipboard == expected_path
