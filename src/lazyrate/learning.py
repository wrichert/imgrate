"""Background pipeline shared by the TUI and the CLI: embed new photos, train on the ratings, predict the rest."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lazyrate.embeddings import EmbeddingMatrix, StoredEmbedding, load_image_for_embedding
from lazyrate.media import PHOTO_EXTENSIONS
from lazyrate.suggestions import RatingModel, evaluate_rating_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from lazyrate.embeddings import ImageEmbedder
    from lazyrate.media import MediaFile
    from lazyrate.store import RatingStore
    from lazyrate.suggestions import ModelEvaluation, ModelReadiness, Suggestion

EMBEDDING_BATCH_SIZE = 32
DECODE_WORKER_COUNT = 4


@dataclass(frozen=True, slots=True)
class LearningResult:
    suggestions: dict[str, Suggestion]
    """For every embedded photo (rated ones too, e.g. to pick the best of a duplicate group)."""
    readiness: ModelReadiness
    evaluation: ModelEvaluation | None
    """None while the model is not ready."""
    embedded_count: int


def is_embedding_current(embedding: StoredEmbedding, media_file: MediaFile) -> bool:
    return (
        embedding.size_bytes == media_file.size_bytes
        and embedding.modified_timestamp_utc == media_file.modified_timestamp_utc
    )


def find_files_missing_embeddings(*, files: list[MediaFile], embeddings: dict[str, StoredEmbedding]) -> list[MediaFile]:
    """Images without an embedding, or whose file changed since it was computed."""
    return [
        media_file
        for media_file in files
        if media_file.extension in PHOTO_EXTENSIONS
        and (
            (embedding := embeddings.get(media_file.relative_path)) is None
            or not is_embedding_current(embedding, media_file)
        )
    ]


def build_current_embedding_matrix(
    *, files: list[MediaFile], embeddings: dict[str, StoredEmbedding]
) -> EmbeddingMatrix:
    """Embeddings of the files that still exist unchanged, in scan order."""
    return EmbeddingMatrix.from_embeddings(
        [
            embedding
            for media_file in files
            if (embedding := embeddings.get(media_file.relative_path)) is not None
            and is_embedding_current(embedding, media_file)
        ]
    )


def compute_embeddings(
    *,
    root: Path,
    files: list[MediaFile],
    embedder: ImageEmbedder,
    store: RatingStore,
    on_progress: Callable[[int, int], None],
    should_stop: Callable[[], bool],
) -> int:
    """Embed ``files`` in batches (decoding in parallel), saving each batch. Returns how many were embedded.

    Undecodable files are skipped; they are retried on the next start.
    """
    embedded_count = 0
    with ThreadPoolExecutor(max_workers=DECODE_WORKER_COUNT) as executor:
        for batch_start in range(0, len(files), EMBEDDING_BATCH_SIZE):
            if should_stop():
                break
            batch_files = files[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
            images = list(
                executor.map(load_image_for_embedding, [root / media_file.relative_path for media_file in batch_files])
            )
            decoded_pairs = [
                (media_file, image) for media_file, image in zip(batch_files, images, strict=True) if image
            ]
            if decoded_pairs:
                vectors = embedder.embed_images([image for _, image in decoded_pairs])
                store.save_embeddings(
                    StoredEmbedding(
                        relative_path=media_file.relative_path,
                        size_bytes=media_file.size_bytes,
                        modified_timestamp_utc=media_file.modified_timestamp_utc,
                        model_name=embedder.model_name,
                        vector=vector,
                    )
                    for (media_file, _), vector in zip(decoded_pairs, vectors, strict=True)
                )
                embedded_count += len(decoded_pairs)
            on_progress(min(batch_start + EMBEDDING_BATCH_SIZE, len(files)), len(files))
    return embedded_count


def train_and_predict(*, embedding_matrix: EmbeddingMatrix, ratings: dict[str, int]) -> LearningResult:
    """Train on the rated embedded photos, evaluate by cross-validation, and predict every embedded photo."""
    labelled_rows = [
        row for row, relative_path in enumerate(embedding_matrix.relative_paths) if relative_path in ratings
    ]
    labelled_features = embedding_matrix.vectors[labelled_rows] if labelled_rows else np.zeros((0, 0))
    labelled_ratings = np.array(
        [ratings[embedding_matrix.relative_paths[row]] for row in labelled_rows], dtype=np.int64
    )
    model = RatingModel(features=labelled_features, ratings=labelled_ratings)
    if not model.is_ready:
        return LearningResult(
            suggestions={},
            readiness=model.readiness,
            evaluation=None,
            embedded_count=len(embedding_matrix.relative_paths),
        )
    suggestions = dict(zip(embedding_matrix.relative_paths, model.predict(embedding_matrix.vectors), strict=True))
    return LearningResult(
        suggestions=suggestions,
        readiness=model.readiness,
        evaluation=evaluate_rating_model(features=labelled_features, ratings=labelled_ratings),
        embedded_count=len(embedding_matrix.relative_paths),
    )
