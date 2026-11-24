"""Rating values, rating filters and their presets."""

from __future__ import annotations

from dataclasses import dataclass

REJECTED = 0
MAX_STARS = 5

RatingBucket = int | None
"""``None`` = unrated, ``0`` = rejected, ``1..5`` = stars."""

ALL_BUCKETS: tuple[RatingBucket, ...] = (None, REJECTED, 1, 2, 3, 4, 5)


def describe_bucket(bucket: RatingBucket) -> str:
    if bucket is None:
        return "unrated"
    if bucket == REJECTED:
        return "rejected"
    return "★" * bucket


def format_stars(rating: int | None) -> str:
    """Render a rating as ``★★★☆☆``, ``✗ rejected`` or ``unrated``."""
    if rating is None:
        return "· unrated"
    if rating == REJECTED:
        return "✗ rejected"
    return "★" * rating + "☆" * (MAX_STARS - rating)


@dataclass(frozen=True, slots=True)
class RatingFilter:
    """Which rating buckets are visible."""

    name: str
    buckets: frozenset[RatingBucket]

    def matches(self, rating: int | None) -> bool:
        return rating in self.buckets


def build_minimum_stars_filter(minimum_stars: int) -> RatingFilter:
    if not 1 <= minimum_stars <= MAX_STARS:
        raise ValueError(f"minimum_stars must be between 1 and {MAX_STARS}, got {minimum_stars}")
    name = "★★★★★ only" if minimum_stars == MAX_STARS else f"≥ {'★' * minimum_stars}"
    return RatingFilter(name=name, buckets=frozenset(range(minimum_stars, MAX_STARS + 1)))


def build_custom_filter(buckets: frozenset[RatingBucket]) -> RatingFilter:
    for preset in FILTER_PRESETS:
        if preset.buckets == buckets:
            return preset
    ordered_buckets = [bucket for bucket in ALL_BUCKETS if bucket in buckets]
    return RatingFilter(name=", ".join(describe_bucket(bucket) for bucket in ordered_buckets), buckets=buckets)


ALL_FILTER = RatingFilter(name="all", buckets=frozenset(ALL_BUCKETS))
UNRATED_FILTER = RatingFilter(name="unrated", buckets=frozenset({None}))
REJECTED_FILTER = RatingFilter(name="rejected", buckets=frozenset({REJECTED}))
NOT_REJECTED_FILTER = RatingFilter(name="all but rejected", buckets=frozenset(ALL_BUCKETS) - {REJECTED})

FILTER_PRESETS: tuple[RatingFilter, ...] = (
    ALL_FILTER,
    UNRATED_FILTER,
    NOT_REJECTED_FILTER,
    *(build_minimum_stars_filter(stars) for stars in range(1, MAX_STARS + 1)),
    REJECTED_FILTER,
)
"""Cycled through with ``v`` / ``V`` in the TUI."""
