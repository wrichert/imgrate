"""How fast you rate, and how long the unrated rest will take at that pace."""

from __future__ import annotations

import time


class RatingPace:
    """Counts ratings given by key and the time spent on them; pauses longer than ``MAX_GAP_SECONDS`` (a coffee, a
    phone call) do not count as rating time."""

    MAX_GAP_SECONDS = 120.0
    MIN_RATINGS_FOR_ESTIMATE = 10

    def __init__(self) -> None:
        self._rating_count = 0
        self._active_seconds = 0.0
        self._last_rating_monotonic: float | None = None

    def __repr__(self) -> str:
        return f"RatingPace(ratings={self._rating_count}, active_seconds={self._active_seconds:.0f})"

    @property
    def ratings_per_minute(self) -> float | None:
        """``None`` until there are enough ratings for a meaningful pace."""
        if self._rating_count < self.MIN_RATINGS_FOR_ESTIMATE or self._active_seconds <= 0:
            return None
        return self._rating_count / self._active_seconds * 60

    def record_rating(self, *, now_monotonic: float | None = None) -> None:
        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        if self._last_rating_monotonic is not None:
            gap_seconds = now_monotonic - self._last_rating_monotonic
            if gap_seconds <= self.MAX_GAP_SECONDS:
                self._active_seconds += gap_seconds
        self._rating_count += 1
        self._last_rating_monotonic = now_monotonic

    def estimate_remaining_seconds(self, remaining_count: int) -> float | None:
        ratings_per_minute = self.ratings_per_minute
        if ratings_per_minute is None:
            return None
        return remaining_count / ratings_per_minute * 60


def format_duration(seconds: float) -> str:
    """``~40 min``, ``~2 h 10 min``, ``< 1 min``."""
    minutes = round(seconds / 60)
    if minutes < 1:
        return "< 1 min"
    if minutes < 60:
        return f"~{minutes} min"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"~{hours} h" if remaining_minutes == 0 else f"~{hours} h {remaining_minutes} min"
