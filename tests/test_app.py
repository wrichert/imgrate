from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from textual.content import Content
from textual.widgets import Input, Static

from lazyrate.app import ImageRaterApp, SectionHeader, SidePaneMode, ThumbnailCell, fit_image_to_cells
from lazyrate.geo_map import BROWSER_MAP_FILE_NAME, Basemap
from lazyrate.media import scan_media_files
from lazyrate.ratings import REJECTED
from lazyrate.screens import HELP_TEXT, ExportScreen
from lazyrate.session import RatingSession, SortOrder
from lazyrate.store import RatingStore

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from conftest import ColorEmbedder
    from textual.pilot import Pilot


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
    monkeypatch.setattr("lazyrate.app.shutil.which", lambda _command: None)  # no clipboard tool -> OSC 52 fallback
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("c")
        await pilot.pause()
        expected_path = str(session.resolve_path(session.current_file).resolve())

    assert app.clipboard == expected_path


async def test_ctrl_arrows_and_ctrl_j_k_jump_to_last_and_first(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("ctrl+down")
        index_after_ctrl_down = session.current_index
        await pilot.press("ctrl+up")
        index_after_ctrl_up = session.current_index
        await pilot.press("ctrl+j")
        index_after_ctrl_j = session.current_index
        await pilot.press("ctrl+k")
        index_after_ctrl_k = session.current_index

    last_index = session.visible_count - 1
    assert (index_after_ctrl_down, index_after_ctrl_up) == (last_index, 0)
    assert (index_after_ctrl_j, index_after_ctrl_k) == (last_index, 0)


async def test_g_toggles_grid(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("g")
        await pilot.pause()
        is_grid_after_one_toggle = app._is_grid_mode

    assert is_grid_after_one_toggle is False


async def test_fast_scrolling_through_grid_pages_does_not_crash(tmp_path: Path) -> None:
    for photo_index in range(120):
        Image.new("RGB", (640, 480), (photo_index * 2, 100, 200 - photo_index)).save(
            tmp_path / f"p{photo_index:03d}.jpg"
        )
    store = RatingStore(tmp_path / RatingStore.DEFAULT_FILE_NAME)
    session = RatingSession(scan_result=scan_media_files(tmp_path), store=store)
    app = ImageRaterApp(session=session, database_path=tmp_path / RatingStore.DEFAULT_FILE_NAME)

    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.press("ctrl+down")
        for _ in range(4):
            await pilot.press(*["k"] * 15)
            await pilot.pause(0.05)
        await pilot.press("ctrl+up")
        await pilot.pause(0.5)
        index_at_end = session.current_index

    assert index_at_end == 0


async def test_rejecting_updates_both_section_counts_in_sidebar_and_grid_headers(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("x", "S")  # creates a "rejected" section
        await pilot.pause(0.3)
        await pilot.press("ctrl+up", "x")
        await pilot.pause(0.3)
        sections_text = str(app.query_one("#sections", Static).render())
        header_texts = [str(header.render()) for header in app.query(SectionHeader)]

    unrated_count = session.visible_count - 2
    assert re.search(rf"unrated\s+{unrated_count} ↻", sections_text)
    assert re.search(r"rejected\s+2 ↻", sections_text)
    assert any(text.startswith("rejected   2 files") and "1 in, 0 out" in text for text in header_texts)


async def test_starts_on_first_unrated_file(session: RatingSession) -> None:
    session.rate(session.visible_files[0], rating=4)
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)):
        current_rating = session.current_rating
        current_index = session.current_index

    assert current_rating is None
    assert current_index == 1  # rating sort: the 4-star file first, then the unrated ones


async def test_p_switches_right_pane_to_large_preview_and_is_remembered(
    session: RatingSession, store: RatingStore
) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("p")
        await pilot.pause(0.5)
        is_preview_pane = app.query_one("#sidebar").has_class("-preview")
        is_image_loaded = app.query_one("#side-preview-image").image is not None
        is_sections_hidden = not app.query_one("#sections").display or not app.query_one("#sections").visible
        await pilot.press("g")
        is_preview_pane_in_single_view = app.query_one("#sidebar").has_class("-preview")

    assert is_preview_pane and is_image_loaded
    assert is_sections_hidden
    assert not is_preview_pane_in_single_view
    assert store.load_setting(ImageRaterApp.SIDE_PANE_SETTING_KEY) == SidePaneMode.PREVIEW


def test_help_text_is_valid_markup() -> None:
    content = Content.from_markup(HELP_TEXT)

    assert "(also ] . and [ ,)" in content.plain


async def test_question_mark_opens_help(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.press("question_mark")
        await pilot.pause()
        screen_name = type(app.screen).__name__

    assert screen_name == "HelpScreen"


async def test_copying_a_path_with_brackets_does_not_break_markup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("lazyrate.app.shutil.which", lambda _command: None)  # no clipboard tool -> OSC 52 fallback
    photo_folder = tmp_path / "[2024] summer"
    photo_folder.mkdir()
    Image.new("RGB", (64, 48), (200, 120, 40)).save(photo_folder / "[b]beach.jpg")
    store = RatingStore(tmp_path / RatingStore.DEFAULT_FILE_NAME)
    session = RatingSession(scan_result=scan_media_files(tmp_path), store=store)
    app = ImageRaterApp(session=session, database_path=tmp_path / RatingStore.DEFAULT_FILE_NAME)

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("c", "3", "z")
        await pilot.pause(0.3)
        copied_path = app.clipboard

    assert copied_path.endswith("[2024] summer/[b]beach.jpg")


async def test_enter_opens_full_view_and_returns_to_previous_view(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_path=session.root / "db.sqlite")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("enter")
        await pilot.pause()
        is_full_view_from_grid = app.query_one("#viewer").display and not app.query_one("#sidebar").display
        await pilot.press("enter")
        await pilot.pause()
        is_back_in_grid = app.query_one("#grid").display and app.query_one("#sidebar").display
        await pilot.press("g", "enter")
        await pilot.pause()
        is_full_view_from_single = not app.query_one("#sidebar").display
        await pilot.press("escape")
        await pilot.pause()
        is_back_in_single_view = app.query_one("#viewer").display and app.query_one("#sidebar").display

    assert is_full_view_from_grid
    assert is_back_in_grid
    assert is_full_view_from_single
    assert is_back_in_single_view


def build_mixed_session(root: Path) -> RatingSession:
    """Half of the receipts rejected and half of the trips starred, so the junk model can train."""
    store = RatingStore(root / RatingStore.DEFAULT_FILE_NAME)
    for index in range(12):
        store.set_rating(relative_path=f"receipts/r{index:02d}.jpg", rating=REJECTED)
        store.set_rating(relative_path=f"trips/p{index:02d}.jpg", rating=3 + index % 3)
    return RatingSession(scan_result=scan_media_files(root), store=store)


async def wait_for_suggestions(app: ImageRaterApp, pilot: Pilot[None], session: RatingSession) -> None:
    for _ in range(100):
        await pilot.pause(0.05)
        if session.suggestion_count and not app.query(".progress-row"):
            return
    raise AssertionError("no suggestions after 5 s")


async def test_background_learning_suggests_junk_and_hides_progress_when_done(
    mixed_root: Path, color_embedder: ColorEmbedder
) -> None:
    session = build_mixed_session(mixed_root)
    app = ImageRaterApp(
        session=session,
        database_path=mixed_root / RatingStore.DEFAULT_FILE_NAME,
        embedder_factory=lambda: color_embedder,
    )

    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_suggestions(app, pilot, session)
        receipt_label = str(
            app._build_thumbnail_label(
                next(file for file in session.all_files if file.relative_path == "receipts/r20.jpg"),
                is_suggested_best=False,
            )
        )
        settings_text = str(app.query_one("#settings", Static).render())

    assert receipt_label.startswith("junk?")
    assert "junk ✓" in settings_text


async def test_y_accepts_and_capital_y_accepts_the_section(mixed_root: Path, color_embedder: ColorEmbedder) -> None:
    session = build_mixed_session(mixed_root)
    app = ImageRaterApp(
        session=session,
        database_path=mixed_root / RatingStore.DEFAULT_FILE_NAME,
        embedder_factory=lambda: color_embedder,
    )

    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_suggestions(app, pilot, session)
        session.set_sort_order(SortOrder.SUGGESTED)
        junk_section = next(section for section in session.sections if section.label == "suggested junk")
        session.move_to(junk_section.start_index)
        await pilot.press("y")
        await pilot.press("Y")
        await pilot.pause()

    receipt_ratings = [rating for path, rating in session.ratings.items() if path.startswith("receipts/")]
    assert receipt_ratings == [REJECTED] * 24


async def test_slash_searches_by_content_and_escape_ends_it(mixed_root: Path, color_embedder: ColorEmbedder) -> None:
    session = build_mixed_session(mixed_root)
    app = ImageRaterApp(
        session=session,
        database_path=mixed_root / RatingStore.DEFAULT_FILE_NAME,
        embedder_factory=lambda: color_embedder,
    )

    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_suggestions(app, pilot, session)
        await pilot.press("slash", *"white", "enter")
        for _ in range(40):
            await pilot.pause(0.05)
            if session.search_query is not None:
                break
        first_match = session.visible_files[0].relative_path
        await pilot.press("escape")
        query_after_escape = session.search_query

    assert first_match.startswith("receipts/")
    assert query_after_escape is None


async def test_ten_new_ratings_retrain_the_model(mixed_root: Path, color_embedder: ColorEmbedder) -> None:
    session = build_mixed_session(mixed_root)
    app = ImageRaterApp(
        session=session,
        database_path=mixed_root / RatingStore.DEFAULT_FILE_NAME,
        embedder_factory=lambda: color_embedder,
    )

    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_suggestions(app, pilot, session)
        session.set_sort_order(SortOrder.PATH)
        session.move_to(
            next(index for index, file in enumerate(session.visible_files) if file.relative_path == "receipts/r12.jpg")
        )
        await pilot.press(*["x"] * ImageRaterApp.RETRAIN_AFTER_RATING_COUNT)
        for _ in range(40):
            await pilot.pause(0.05)
            if app._learning_result is not None and app._learning_result.readiness.rejected_count > 12:
                break
        rejected_count = app._learning_result.readiness.rejected_count if app._learning_result else 0

    assert rejected_count == 12 + ImageRaterApp.RETRAIN_AFTER_RATING_COUNT


async def test_without_ai_search_still_finds_places_and_explains_text_search(geo_root: Path) -> None:
    store = RatingStore(geo_root / RatingStore.DEFAULT_FILE_NAME)
    session = RatingSession(scan_result=scan_media_files(geo_root), store=store)
    app = ImageRaterApp(session=session, database_path=store.database_path)

    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_locations(pilot, session)
        await search_for(pilot, "sunset")
        query_after_text_search = session.search_query
        await search_for(pilot, "madeira")
        section_labels = [section.label for section in session.sections]
        settings_text = str(app.query_one("#settings", Static).render())
    store.close()

    assert query_after_text_search is None
    assert session.search_query == "madeira"
    assert section_labels[:2] == ["Portugal › Madeira › Funchal", "Portugal › Madeira › São Jorge"]  # noqa: RUF001
    assert section_labels[-1] == "other photos"
    assert "AI  off" in settings_text


async def wait_for_locations(pilot: Pilot[None], session: RatingSession) -> None:
    for _ in range(100):
        await pilot.pause(0.05)
        if not session.find_files_missing_locations():
            return
    raise AssertionError("GPS locations not read after 5 s")


async def search_for(pilot: Pilot[None], query: str) -> None:
    await pilot.press("slash")
    await pilot.pause()
    pilot.app.screen.query_one(Input).value = query
    await pilot.press("enter")
    await pilot.pause()


async def test_radius_search_and_info_panel_show_the_place(geo_root: Path) -> None:
    store = RatingStore(geo_root / RatingStore.DEFAULT_FILE_NAME)
    session = RatingSession(scan_result=scan_media_files(geo_root), store=store)
    app = ImageRaterApp(session=session, database_path=store.database_path)

    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_locations(pilot, session)
        await search_for(pilot, "@32.6657,-16.9255 5km")
        current_path = session.current_file.relative_path if session.current_file else None
        info_text = str(app.query_one("#info", Static).render())
    store.close()

    assert current_path == "madeira/funchal_1.jpg"
    assert session.sections[0].label == "< 1 km"
    assert "Funchal, Madeira, Portugal" in info_text
    assert "32.66570, -16.92550" in info_text


async def test_place_plus_text_ranks_within_the_place_with_ai(geo_root: Path, color_embedder: ColorEmbedder) -> None:
    store = RatingStore(geo_root / RatingStore.DEFAULT_FILE_NAME)
    session = RatingSession(scan_result=scan_media_files(geo_root), store=store)
    app = ImageRaterApp(session=session, database_path=store.database_path, embedder_factory=lambda: color_embedder)

    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_locations(pilot, session)
        for _ in range(100):
            await pilot.pause(0.05)
            if app._is_ai_search_ready:
                break
        await search_for(pilot, "white madeira")
        for _ in range(40):
            await pilot.pause(0.05)
            if session.search_query is not None:
                break
        first_paths = [media_file.relative_path for media_file in session.visible_files[:3]]
    store.close()

    assert first_paths[0] == "madeira/funchal_1.jpg"
    assert set(first_paths) == {"madeira/funchal_1.jpg", "madeira/funchal_2.jpg", "madeira/north_coast.jpg"}
    assert session.sections[1].label == "other photos"


async def test_m_shows_the_map_of_search_matches_and_capital_m_writes_the_browser_map(
    geo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RatingStore(geo_root / RatingStore.DEFAULT_FILE_NAME)
    session = RatingSession(scan_result=scan_media_files(geo_root), store=store)
    app = ImageRaterApp(session=session, database_path=store.database_path)
    opened_paths: list[Path] = []
    monkeypatch.setattr(app, "_open_with_default_app", opened_paths.append)

    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_locations(pilot, session)
        await search_for(pilot, "madeira")
        await pilot.press("m")
        for _ in range(60):
            await pilot.pause(0.05)
            if app.query_one("#side-preview-image").image is not None:
                break
        map_image = app.query_one("#side-preview-image").image
        await pilot.press("M")
        for _ in range(60):
            await pilot.pause(0.05)
            if opened_paths:
                break
    store.close()

    assert map_image is not None
    assert Basemap.COLORS.current in {color for _count, color in map_image.getcolors(maxcolors=1_000_000)}
    assert opened_paths == [geo_root / BROWSER_MAP_FILE_NAME]
    page = opened_paths[0].read_text()
    assert page.count('"path": ') == 3
    assert "munich" not in page
    assert store.database_path.exists()


async def test_map_zooms_pans_and_fits_again_and_resets_for_a_new_search(geo_root: Path) -> None:
    store = RatingStore(geo_root / RatingStore.DEFAULT_FILE_NAME)
    session = RatingSession(scan_result=scan_media_files(geo_root), store=store)
    app = ImageRaterApp(session=session, database_path=store.database_path)

    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_locations(pilot, session)
        await search_for(pilot, "madeira")
        await pilot.press("m")
        await pilot.pause()
        await pilot.press("plus", "plus", "shift+right")
        zoomed_viewport = app._map_viewport
        await pilot.click("#side-preview", offset=(2, 2))  # water, no dot: centers the map there
        clicked_viewport = app._map_viewport
        await pilot.press("asterisk")
        fitted_viewport = app._map_viewport
        await pilot.press("plus")
        await search_for(pilot, "portugal")
        await pilot.pause()
        viewport_after_search = app._map_viewport
    store.close()

    assert zoomed_viewport.zoom_level == 2
    assert zoomed_viewport.center_x_ratio > 0
    assert clicked_viewport.center_x_ratio < zoomed_viewport.center_x_ratio
    assert fitted_viewport.is_fitted
    assert viewport_after_search.is_fitted


async def test_every_terminal_resize_relayouts_the_grid(session: RatingSession) -> None:
    """Regression: the app's own debounce timer shadowed Textual's ``App._resize_timer``, so after the first resize
    the screen was never laid out again."""
    app = ImageRaterApp(session=session, database_path=session.root / RatingStore.DEFAULT_FILE_NAME)
    layout_keys = []

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for width, height in [(160, 50), (100, 30), (140, 45)]:
            await pilot.resize_terminal(width, height)
            await pilot.pause(ImageRaterApp.RESIZE_DEBOUNCE_SECONDS + 0.1)
            layout_keys.append(app._grid_layout_key)

    statusbar_and_footer_height = 2
    assert [key.available_height for key in layout_keys] == [
        height - statusbar_and_footer_height for height in (50, 30, 45)
    ]


async def test_clicking_a_dot_on_the_map_selects_its_photo_and_dragging_pans(geo_root: Path) -> None:
    store = RatingStore(geo_root / RatingStore.DEFAULT_FILE_NAME)
    session = RatingSession(scan_result=scan_media_files(geo_root), store=store)
    app = ImageRaterApp(session=session, database_path=store.database_path)

    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_locations(pilot, session)
        await search_for(pilot, "@48.1374,11.5755 500km")  # Munich only
        await pilot.press("m")
        for _ in range(60):
            await pilot.pause(0.05)
            if app._side_map_base is not None and app._side_map_points:
                break
        pane_size = app.query_one("#side-preview").content_size
        munich_point = next(point for point in app._side_map_points if "munich" in point.relative_path)
        ((x_px, y_px),) = app._side_map_base.extent.project(
            longitudes_deg=np.array([munich_point.coordinates.longitude_deg]),
            latitudes_deg=np.array([munich_point.coordinates.latitude_deg]),
            width_px=pane_size.width,
            height_px=pane_size.height,
        ).tolist()
        session.move_to(session.visible_count - 1)  # an "other photo", not Munich
        await pilot.click("#side-preview", offset=(int(x_px), int(y_px)))
        selected_path = session.current_file.relative_path
        await pilot.mouse_down("#side-preview", offset=(10, 10))
        await pilot.hover("#side-preview", offset=(10, 5))
        await pilot.mouse_up("#side-preview", offset=(10, 5))
        dragged_viewport = app._map_viewport
    store.close()

    assert selected_path == munich_point.relative_path
    assert dragged_viewport.center_y_ratio > 0  # pulled up: the map moves south
    assert dragged_viewport.zoom_level == 0
