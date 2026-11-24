from __future__ import annotations

from typing import TYPE_CHECKING

from imgrate.app import ImageRaterApp, fit_image_to_cells
from imgrate.ratings import REJECTED

if TYPE_CHECKING:
    from imgrate.session import RatingSession
    from imgrate.store import RatingStore


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
    app = ImageRaterApp(session=session, database_label="db: test")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("5", "3", "x")
        await pilot.press("z")
        await pilot.pause()

    assert store.load_ratings() == {"2023/a.jpg": 5, "2023/b.JPG": 3}
    assert session.current_file is not None
    assert session.current_file.relative_path == "2023/sub/c.png"


async def test_view_cycling_and_grid(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_label="db: test")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("4", "0")
        await pilot.press("v")  # all -> unrated
        assert session.visible_count == 3
        await pilot.press("g")
        await pilot.pause(0.3)
        await pilot.press("right", "2")
        await pilot.pause(0.3)
        assert len(app.query(".thumbnail")) > 1

    assert session.get_rating(session.visible_files[1]) == 2
    assert session.count_ratings()[REJECTED] == 1


async def test_extension_dialog_enables_video(session: RatingSession) -> None:
    app = ImageRaterApp(session=session, database_label="db: test")

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("e")
        await pilot.press("a", "enter")
        await pilot.pause()

    assert ".mp4" in session.enabled_extensions
    assert session.visible_count == 6
