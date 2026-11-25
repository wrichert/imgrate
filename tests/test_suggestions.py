from __future__ import annotations

import numpy as np
import pytest

from lazyrate.ratings import REJECTED
from lazyrate.suggestions import RatingModel, Suggestion, assess_readiness, evaluate_rating_model


def build_synthetic_ratings(*, photo_count: int = 600, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Features with a hidden quality axis (-> 1..5 stars) and a hidden junk axis (-> 0)."""
    rng = np.random.default_rng(seed)
    quality = rng.normal(size=photo_count)
    junkness = rng.normal(size=photo_count)
    dimension_count = 32
    quality_direction = rng.normal(size=dimension_count)
    junk_direction = rng.normal(size=dimension_count)
    features = (
        quality[:, None] * quality_direction
        + junkness[:, None] * junk_direction
        + rng.normal(scale=0.5, size=(photo_count, dimension_count))
    )
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    ratings = np.where(junkness > 1.0, REJECTED, np.clip(np.round(quality + 3), 1, 5)).astype(np.int64)
    return features, ratings


def test_model_learns_junk_and_ordered_stars() -> None:
    features, ratings = build_synthetic_ratings()

    model = RatingModel(features=features, ratings=ratings)
    suggestions = model.predict(features)

    predicted_junk = np.array([suggestion.is_junk for suggestion in suggestions])
    assert (predicted_junk == (ratings == REJECTED)).mean() > 0.85
    kept = ratings != REJECTED
    expected_stars = np.array([suggestion.expected_stars for suggestion in suggestions])
    assert np.corrcoef(expected_stars[kept], ratings[kept])[0, 1] > 0.7


def test_cross_validation_reports_star_accuracy_within_one() -> None:
    features, ratings = build_synthetic_ratings()

    evaluation = evaluate_rating_model(features=features, ratings=ratings)

    assert evaluation.star_within_one_accuracy is not None
    assert evaluation.star_within_one_accuracy > 0.85
    assert evaluation.junk_recall is not None
    assert evaluation.junk_recall > 0.7


def test_no_suggestions_until_enough_rejected_and_starred() -> None:
    features, _ = build_synthetic_ratings(photo_count=15)
    ratings = np.array([REJECTED] * 5 + [4] * 10)

    model = RatingModel(features=features, ratings=ratings)

    assert not model.is_ready
    assert model.predict(features) == []
    assert model.readiness.missing_rejected_count == 5


def test_star_model_waits_for_enough_starred_photos() -> None:
    readiness = assess_readiness([REJECTED] * 12 + [3] * 12 + [5] * 5)

    assert readiness.is_junk_model_ready
    assert not readiness.is_star_model_ready
    assert readiness.missing_starred_count == RatingModel.MIN_STARRED_FOR_STARS - 17


def test_suggestion_combines_junk_and_stars() -> None:
    keeper = Suggestion(junk_probability=0.1, star_probabilities=(0.0, 0.1, 0.2, 0.6, 0.1))
    junk = Suggestion(junk_probability=0.9, star_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2))
    keeper_without_stars = Suggestion(junk_probability=0.2, star_probabilities=None)

    assert (keeper.suggested_rating, keeper.confidence) == (4, pytest.approx(0.54))
    assert keeper.expected_stars == pytest.approx(0.9 * 3.7)
    assert junk.suggested_rating == REJECTED
    assert keeper_without_stars.suggested_rating is None
    assert 0.0 <= keeper.uncertainty <= 1.0
    assert junk.uncertainty < Suggestion(junk_probability=0.5, star_probabilities=(0.2,) * 5).uncertainty
