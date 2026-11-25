"""Suggested ratings learned from the user's own ratings, on top of image embeddings.

Two small linear models share one embedding per photo:

* a junk model (logistic regression): rejected (0) vs. kept (1-5 stars). Junk is a *kind* of picture (receipts,
  screenshots, pocket shots), not the bottom of the quality scale, so it gets its own yes / no model;
* a star model (ordinal logistic regression, cumulative-link "CORAL" style) for 1-5 stars on kept photos. Stars are
  ordered, so predicting 4 for a 5 is a small error and 1 for a 5 a big one.

Unrated photos never act as training data. Pure numpy: training on 10,000 x 512 takes well under a second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lazyrate.ratings import MAX_STARS, REJECTED

STAR_LEVELS = tuple(range(1, MAX_STARS + 1))
RATING_LEVEL_COUNT = MAX_STARS + 1
"""Rejected plus 1-5 stars."""


@dataclass(frozen=True, slots=True)
class Suggestion:
    """Predicted rating distribution for one photo.

    ``star_probabilities`` (for 1..5 stars, given the photo is kept) is None while there are too few starred
    examples to train the star model.
    """

    junk_probability: float
    star_probabilities: tuple[float, ...] | None
    JUNK_THRESHOLD = 0.5

    @property
    def is_junk(self) -> bool:
        return self.junk_probability >= self.JUNK_THRESHOLD

    @property
    def suggested_rating(self) -> int | None:
        """0 for junk, else the most likely star count; None if the photo is a keeper but stars are unknown yet."""
        if self.is_junk:
            return REJECTED
        if self.star_probabilities is None:
            return None
        return STAR_LEVELS[int(np.argmax(self.star_probabilities))]

    @property
    def confidence(self) -> float:
        """Probability of the suggested outcome."""
        if self.is_junk:
            return self.junk_probability
        if self.star_probabilities is None:
            return 1.0 - self.junk_probability
        return (1.0 - self.junk_probability) * max(self.star_probabilities)

    @property
    def expected_stars(self) -> float:
        """Expected rating with junk counted as 0; ranks photos best first. Neutral 3 stars if stars are unknown."""
        keep_probability = 1.0 - self.junk_probability
        if self.star_probabilities is None:
            return keep_probability * (MAX_STARS + 1) / 2
        return keep_probability * sum(
            star * probability for star, probability in zip(STAR_LEVELS, self.star_probabilities, strict=True)
        )

    @property
    def uncertainty(self) -> float:
        """Normalised entropy in [0, 1] of the full 0-5 distribution (or of junk / keep without a star model)."""
        if self.star_probabilities is None:
            return compute_normalized_entropy([self.junk_probability, 1.0 - self.junk_probability])
        keep_probability = 1.0 - self.junk_probability
        return compute_normalized_entropy(
            [self.junk_probability, *(keep_probability * probability for probability in self.star_probabilities)]
        )


@dataclass(frozen=True, slots=True)
class ModelReadiness:
    """How many labelled examples exist, and how many more each model needs."""

    rejected_count: int
    starred_count: int
    distinct_star_count: int
    missing_rejected_count: int
    missing_starred_count: int

    @property
    def is_junk_model_ready(self) -> bool:
        return self.missing_rejected_count == 0 and self.starred_count >= RatingModel.MIN_EXAMPLES_PER_SIDE

    @property
    def is_star_model_ready(self) -> bool:
        return self.missing_starred_count == 0 and self.distinct_star_count >= RatingModel.MIN_DISTINCT_STARS


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Cross-validated quality: every rated photo is predicted by a model that never saw it."""

    junk_evaluated_count: int
    junk_precision: float | None
    """Of the photos predicted junk, the share you actually rejected."""
    junk_recall: float | None
    """Of the photos you rejected, the share predicted junk."""
    star_evaluated_count: int
    star_exact_accuracy: float | None
    star_within_one_accuracy: float | None
    star_mean_absolute_error: float | None


def compute_normalized_entropy(probabilities: list[float]) -> float:
    positive_probabilities = [probability for probability in probabilities if probability > 0]
    if len(probabilities) < 2:
        return 0.0
    entropy = -sum(probability * math.log(probability) for probability in positive_probabilities)
    return entropy / math.log(len(probabilities))


def assess_readiness(ratings: list[int]) -> ModelReadiness:
    rejected_count = sum(1 for rating in ratings if rating == REJECTED)
    starred_ratings = [rating for rating in ratings if rating != REJECTED]
    return ModelReadiness(
        rejected_count=rejected_count,
        starred_count=len(starred_ratings),
        distinct_star_count=len(set(starred_ratings)),
        missing_rejected_count=max(0, RatingModel.MIN_EXAMPLES_PER_SIDE - rejected_count),
        missing_starred_count=max(0, RatingModel.MIN_STARRED_FOR_STARS - len(starred_ratings)),
    )


class _Standardizer:
    """Per-dimension z-scoring fitted on the training features (better conditioned than raw unit-norm CLIP)."""

    def __init__(self, features: np.ndarray) -> None:
        self._mean = features.mean(axis=0)
        self._scale = features.std(axis=0) + 1e-6

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self._mean) / self._scale


class _AdamOptimizer:
    """Full-batch Adam over a list of parameter arrays (updated in place)."""

    def __init__(self, parameters: list[np.ndarray], *, learning_rate: float) -> None:
        self._parameters = parameters
        self._learning_rate = learning_rate
        self._first_moments = [np.zeros_like(parameter) for parameter in parameters]
        self._second_moments = [np.zeros_like(parameter) for parameter in parameters]
        self._step_count = 0

    def step(self, gradients: list[np.ndarray]) -> None:
        self._step_count += 1
        beta_first, beta_second, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, first_moment, second_moment in zip(
            self._parameters, gradients, self._first_moments, self._second_moments, strict=True
        ):
            first_moment *= beta_first
            first_moment += (1 - beta_first) * gradient
            second_moment *= beta_second
            second_moment += (1 - beta_second) * gradient**2
            corrected_first = first_moment / (1 - beta_first**self._step_count)
            corrected_second = second_moment / (1 - beta_second**self._step_count)
            parameter -= self._learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def _compute_balanced_weights(labels: np.ndarray) -> np.ndarray:
    """Each class contributes equally to the loss, so 40 five-star photos are not drowned by 2,000 three-star ones."""
    classes, counts = np.unique(labels, return_counts=True)
    weight_by_class = {
        label: len(labels) / (len(classes) * count) for label, count in zip(classes, counts, strict=True)
    }
    return np.array([weight_by_class[label] for label in labels], dtype=np.float64)


class JunkModel:
    """L2-regularised, class-balanced logistic regression: probability that a photo is junk."""

    STEP_COUNT = 300
    LEARNING_RATE = 0.05
    L2_PENALTY = 1e-2

    def __init__(self, *, features: np.ndarray, is_junk: np.ndarray) -> None:
        self._standardizer = _Standardizer(features)
        standardized_features = self._standardizer.transform(features)
        targets = is_junk.astype(np.float64)
        sample_weights = _compute_balanced_weights(is_junk)
        self._weights = np.zeros(features.shape[1])
        self._bias = np.zeros(1)
        optimizer = _AdamOptimizer([self._weights, self._bias], learning_rate=self.LEARNING_RATE)
        for _ in range(self.STEP_COUNT):
            probabilities = _sigmoid(standardized_features @ self._weights + self._bias)
            residuals = sample_weights * (probabilities - targets) / len(targets)
            weight_gradient = standardized_features.T @ residuals + self.L2_PENALTY * self._weights
            optimizer.step([weight_gradient, np.array([residuals.sum()])])

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        return _sigmoid(self._standardizer.transform(features) @ self._weights + self._bias)


class StarModel:
    """Ordinal logistic regression for 1..5 stars: shared direction, one threshold per step (P(stars > k))."""

    STEP_COUNT = 400
    LEARNING_RATE = 0.05
    L2_PENALTY = 1e-2

    def __init__(self, *, features: np.ndarray, stars: np.ndarray) -> None:
        self._standardizer = _Standardizer(features)
        standardized_features = self._standardizer.transform(features)
        step_levels = np.array(STAR_LEVELS[:-1])
        # targets[i, k] = 1 if photo i has more than k+1 stars.
        targets = (stars[:, None] > step_levels[None, :]).astype(np.float64)
        sample_weights = _compute_balanced_weights(stars)[:, None]
        self._weights = np.zeros(features.shape[1])
        self._thresholds = np.linspace(1.0, -1.0, len(step_levels))
        optimizer = _AdamOptimizer([self._weights, self._thresholds], learning_rate=self.LEARNING_RATE)
        for _ in range(self.STEP_COUNT):
            logits = (standardized_features @ self._weights)[:, None] + self._thresholds[None, :]
            residuals = sample_weights * (_sigmoid(logits) - targets) / len(stars)
            weight_gradient = standardized_features.T @ residuals.sum(axis=1) + self.L2_PENALTY * self._weights
            optimizer.step([weight_gradient, residuals.sum(axis=0)])

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Shape (n, 5): probability of 1..5 stars."""
        scores = self._standardizer.transform(features) @ self._weights
        # Monotone thresholds keep P(stars > k) non-increasing in k, so class probabilities stay non-negative.
        thresholds = np.minimum.accumulate(self._thresholds)
        exceed_probabilities = _sigmoid(scores[:, None] + thresholds[None, :])
        upper_bounds = np.concatenate([np.ones((len(scores), 1)), exceed_probabilities], axis=1)
        lower_bounds = np.concatenate([exceed_probabilities, np.zeros((len(scores), 1))], axis=1)
        probabilities = np.clip(upper_bounds - lower_bounds, 0.0, None)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


class RatingModel:
    """Junk model plus (once enough stars exist) star model, trained on the rated photos only."""

    MIN_EXAMPLES_PER_SIDE = 10
    """Rejected and starred photos each, before junk is suggested."""
    MIN_STARRED_FOR_STARS = 30
    MIN_DISTINCT_STARS = 2

    def __init__(self, *, features: np.ndarray, ratings: np.ndarray) -> None:
        if len(features) != len(ratings):
            raise ValueError(f"{len(features)} feature rows but {len(ratings)} ratings")
        self.readiness = assess_readiness([int(rating) for rating in ratings])
        is_junk = ratings == REJECTED
        self._junk_model = JunkModel(features=features, is_junk=is_junk) if self.readiness.is_junk_model_ready else None
        self._star_model = (
            StarModel(features=features[~is_junk], stars=ratings[~is_junk])
            if self.readiness.is_star_model_ready
            else None
        )

    @property
    def is_ready(self) -> bool:
        return self._junk_model is not None

    def predict(self, features: np.ndarray) -> list[Suggestion]:
        """One suggestion per row; empty while the junk model is not ready (too few rejected / starred photos)."""
        if self._junk_model is None or len(features) == 0:
            return []
        junk_probabilities = self._junk_model.predict_probabilities(features)
        star_probabilities = None if self._star_model is None else self._star_model.predict_probabilities(features)
        return [
            Suggestion(
                junk_probability=float(junk_probabilities[row]),
                star_probabilities=None
                if star_probabilities is None
                else tuple(float(value) for value in star_probabilities[row]),
            )
            for row in range(len(features))
        ]


def evaluate_rating_model(*, features: np.ndarray, ratings: np.ndarray, fold_count: int = 5) -> ModelEvaluation:
    """k-fold cross-validation: each rated photo is predicted by a model trained on the other folds."""
    fold_ids = np.random.default_rng(seed=0).permutation(len(ratings)) % fold_count
    predicted_junk: list[bool] = []
    actual_junk: list[bool] = []
    predicted_stars: list[int] = []
    actual_stars: list[int] = []
    for fold_id in range(fold_count):
        is_held_out = fold_ids == fold_id
        if not is_held_out.any():
            continue
        fold_model = RatingModel(features=features[~is_held_out], ratings=ratings[~is_held_out])
        if not fold_model.is_ready:
            # Too few examples left after holding out this fold; it cannot be evaluated.
            continue
        held_out_ratings = ratings[is_held_out]
        for suggestion, rating in zip(fold_model.predict(features[is_held_out]), held_out_ratings, strict=True):
            predicted_junk.append(suggestion.is_junk)
            actual_junk.append(int(rating) == REJECTED)
            if int(rating) != REJECTED and suggestion.star_probabilities is not None:
                predicted_stars.append(STAR_LEVELS[int(np.argmax(suggestion.star_probabilities))])
                actual_stars.append(int(rating))
    return _summarize_evaluation(
        predicted_junk=predicted_junk,
        actual_junk=actual_junk,
        predicted_stars=predicted_stars,
        actual_stars=actual_stars,
    )


def _summarize_evaluation(
    *, predicted_junk: list[bool], actual_junk: list[bool], predicted_stars: list[int], actual_stars: list[int]
) -> ModelEvaluation:
    true_positive_count = sum(
        1 for predicted, actual in zip(predicted_junk, actual_junk, strict=True) if predicted and actual
    )
    predicted_positive_count = sum(predicted_junk)
    actual_positive_count = sum(actual_junk)
    star_errors = [abs(predicted - actual) for predicted, actual in zip(predicted_stars, actual_stars, strict=True)]
    return ModelEvaluation(
        junk_evaluated_count=len(actual_junk),
        junk_precision=true_positive_count / predicted_positive_count if predicted_positive_count else None,
        junk_recall=true_positive_count / actual_positive_count if actual_positive_count else None,
        star_evaluated_count=len(star_errors),
        star_exact_accuracy=sum(1 for error in star_errors if error == 0) / len(star_errors) if star_errors else None,
        star_within_one_accuracy=sum(1 for error in star_errors if error <= 1) / len(star_errors)
        if star_errors
        else None,
        star_mean_absolute_error=sum(star_errors) / len(star_errors) if star_errors else None,
    )
