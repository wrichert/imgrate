from __future__ import annotations

import pytest

from lazyrate.pace import RatingPace, describe_pace, format_duration


def test_pace_ignores_long_pauses() -> None:
    pace = RatingPace()
    for index in range(10):
        pace.record_rating(now_monotonic=index * 3.0)  # one rating every 3 s
    pace.record_rating(now_monotonic=27.0 + 600)  # back after a 10 minute break

    assert pace.ratings_per_minute == pytest.approx(11 / 27 * 60)


def test_no_estimate_before_enough_ratings() -> None:
    pace = RatingPace()
    for index in range(RatingPace.MIN_RATINGS_FOR_ESTIMATE - 1):
        pace.record_rating(now_monotonic=float(index))

    assert pace.ratings_per_minute is None


@pytest.mark.parametrize(
    ("seconds", "expected_text"), [(20, "< 1 min"), (40 * 60, "~40 min"), (130 * 60, "~2 h 10 min"), (7200, "~2 h")]
)
def test_durations_are_rounded_to_minutes(seconds: float, expected_text: str) -> None:
    assert format_duration(seconds) == expected_text


def test_pace_says_all_rated_when_nothing_is_left() -> None:
    description = describe_pace(ratings_per_minute=12.0, unrated_count=0)

    assert description == "12/min · all rated"


def test_pace_estimates_time_for_the_unrated_rest() -> None:
    description = describe_pace(ratings_per_minute=12.0, unrated_count=340)

    assert description == "12/min · 340 unrated · ~28 min left"
