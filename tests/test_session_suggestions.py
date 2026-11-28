from __future__ import annotations

from typing import TYPE_CHECKING

from lazyrate.ratings import REJECTED
from lazyrate.session import SortOrder
from lazyrate.suggestions import Suggestion

if TYPE_CHECKING:
    from lazyrate.session import RatingSession

FIVE_STARS = Suggestion(junk_probability=0.05, star_probabilities=(0.0, 0.0, 0.05, 0.15, 0.8))
THREE_STARS = Suggestion(junk_probability=0.1, star_probabilities=(0.05, 0.1, 0.7, 0.1, 0.05))
JUNK = Suggestion(junk_probability=0.95, star_probabilities=(0.6, 0.2, 0.1, 0.05, 0.05))
UNSURE = Suggestion(junk_probability=0.45, star_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2))

SUGGESTIONS = {
    "2023/a.jpg": THREE_STARS,
    "2023/b.JPG": JUNK,
    "2023/sub/c.png": FIVE_STARS,
    "2024/d.jpg": UNSURE,
}


def visible_paths(session: RatingSession) -> list[str]:
    return [media_file.relative_path for media_file in session.visible_files]


def test_suggestion_sorts_appear_only_with_suggestions(session: RatingSession) -> None:
    sort_orders_before = session.available_sort_orders

    session.set_suggestions(SUGGESTIONS)

    assert SortOrder.SUGGESTED not in sort_orders_before
    assert SortOrder.SUGGESTED in session.available_sort_orders
    assert SortOrder.RELEVANCE not in session.available_sort_orders


def test_suggested_sort_puts_best_first_junk_last_and_rated_after(session: RatingSession) -> None:
    session.set_suggestions(SUGGESTIONS)
    session.rate(session.visible_files[4], rating=2)  # 2024/e.heic

    session.set_sort_order(SortOrder.SUGGESTED)

    assert visible_paths(session) == ["2023/sub/c.png", "2023/a.jpg", "2024/d.jpg", "2023/b.JPG", "2024/e.heic"]
    assert [section.label for section in session.sections] == [
        "suggested ★★★★★",
        "suggested ★★★",
        "suggested ★",
        "suggested rejected",
        "★★",
    ]


def test_accepting_suggestions_rates_and_moves_counts(session: RatingSession) -> None:
    session.set_suggestions(SUGGESTIONS)
    session.set_sort_order(SortOrder.SUGGESTED)
    session.move_to(3)  # suggested junk

    change = session.accept_suggestion_for_current()
    junk_section_counts = session.count_section_members(session.sections[3])

    assert change is not None
    assert change.new_rating == REJECTED
    assert junk_section_counts.moved_out_count == 1


def test_accepting_a_section_is_one_undo_step(session: RatingSession) -> None:
    session.set_suggestions(dict.fromkeys(visible_paths(session), JUNK))
    session.set_sort_order(SortOrder.SUGGESTED)

    changes = session.accept_suggestions_in_current_section()
    session.undo()

    assert len(changes) == 5
    assert session.ratings == {}


def test_keeper_without_stars_is_not_accepted(session: RatingSession) -> None:
    session.set_suggestions({"2023/a.jpg": Suggestion(junk_probability=0.1, star_probabilities=None)})
    session.set_sort_order(SortOrder.PATH)

    change = session.accept_suggestion_for_current()

    assert change is None


def test_informative_sort_shows_least_certain_first(session: RatingSession) -> None:
    session.set_suggestions(SUGGESTIONS)

    session.set_sort_order(SortOrder.INFORMATIVE)

    assert visible_paths(session)[0] == "2024/d.jpg"
    assert session.sections[-1].label == "no suggestion yet"


def test_new_suggestions_do_not_reshuffle_until_resort(session: RatingSession) -> None:
    session.set_suggestions(SUGGESTIONS)
    session.set_sort_order(SortOrder.SUGGESTED)
    order_before = visible_paths(session)

    session.set_suggestions({path: JUNK for path in SUGGESTIONS})
    order_after_update = visible_paths(session)
    session.refresh_view()

    assert order_after_update == order_before
    assert session.sections[0].label == "suggested rejected"


def test_search_shows_best_matches_first_and_clearing_restores_sort(session: RatingSession) -> None:
    session.set_sort_order(SortOrder.DATE)

    session.set_search(query="sunset", scores={"2024/d.jpg": 0.9, "2023/a.jpg": 0.5})
    paths_during_search = visible_paths(session)
    first_label_during_search = session.sections[0].label
    session.clear_search()

    assert paths_during_search[:2] == ["2024/d.jpg", "2023/a.jpg"]
    assert first_label_during_search == "best 20 matches"
    assert session.sort_order is SortOrder.DATE
    assert session.search_query is None


def test_suggested_best_of_a_section_is_the_highest_expected_rating(session: RatingSession) -> None:
    session.set_suggestions(SUGGESTIONS)
    session.set_sort_order(SortOrder.PATH)
    folder_2023 = session.sections[0]

    best_index = session.find_suggested_best_index(folder_2023)

    assert best_index is not None
    assert session.visible_files[best_index].relative_path == "2023/a.jpg"
