from __future__ import annotations

import os
from typing import TYPE_CHECKING

from lazyrate.learning import (
    build_current_embedding_matrix,
    compute_embeddings,
    find_files_missing_embeddings,
    train_and_predict,
)
from lazyrate.media import scan_media_files
from lazyrate.ratings import REJECTED

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import ColorEmbedder

    from lazyrate.store import RatingStore


def embed_all(*, root: Path, store: RatingStore, embedder: ColorEmbedder) -> list[tuple[int, int]]:
    scan_result = scan_media_files(root)
    progress_reports: list[tuple[int, int]] = []
    compute_embeddings(
        root=root,
        files=scan_result.files,
        embedder=embedder,
        store=store,
        on_progress=lambda done_count, total_count: progress_reports.append((done_count, total_count)),
        should_stop=lambda: False,
    )
    return progress_reports


def test_embeddings_are_cached_and_recomputed_only_for_changed_files(
    mixed_root: Path, store: RatingStore, color_embedder: ColorEmbedder
) -> None:
    progress_reports = embed_all(root=mixed_root, store=store, embedder=color_embedder)
    changed_path = mixed_root / "trips" / "p00.jpg"
    os.utime(changed_path, (1_000_000, 1_000_000))

    pending_files = find_files_missing_embeddings(
        files=scan_media_files(mixed_root).files, embeddings=store.load_embeddings(color_embedder.model_name)
    )

    assert progress_reports[-1] == (48, 48)
    assert [media_file.relative_path for media_file in pending_files] == ["trips/p00.jpg"]


def test_undecodable_files_are_skipped(mixed_root: Path, store: RatingStore, color_embedder: ColorEmbedder) -> None:
    (mixed_root / "broken.jpg").write_bytes(b"not a jpeg")

    embed_all(root=mixed_root, store=store, embedder=color_embedder)

    embeddings = store.load_embeddings(color_embedder.model_name)
    assert len(embeddings) == 48
    assert "broken.jpg" not in embeddings


def test_train_and_predict_suggests_junk_for_unrated_receipts(
    mixed_root: Path, store: RatingStore, color_embedder: ColorEmbedder
) -> None:
    embed_all(root=mixed_root, store=store, embedder=color_embedder)
    ratings = {f"receipts/r{index:02d}.jpg": REJECTED for index in range(12)} | {
        f"trips/p{index:02d}.jpg": 3 + index % 3 for index in range(12)
    }
    embedding_matrix = build_current_embedding_matrix(
        files=scan_media_files(mixed_root).files, embeddings=store.load_embeddings(color_embedder.model_name)
    )

    result = train_and_predict(embedding_matrix=embedding_matrix, ratings=ratings)

    unrated_receipts = [f"receipts/r{index:02d}.jpg" for index in range(12, 24)]
    unrated_trips = [f"trips/p{index:02d}.jpg" for index in range(12, 24)]
    assert all(result.suggestions[path].is_junk for path in unrated_receipts)
    assert not any(result.suggestions[path].is_junk for path in unrated_trips)
    assert result.evaluation is not None


def test_search_ranks_matching_photos_first(
    mixed_root: Path, store: RatingStore, color_embedder: ColorEmbedder
) -> None:
    embed_all(root=mixed_root, store=store, embedder=color_embedder)
    embedding_matrix = build_current_embedding_matrix(
        files=scan_media_files(mixed_root).files, embeddings=store.load_embeddings(color_embedder.model_name)
    )

    scores = embedding_matrix.rank_by_similarity(color_embedder.embed_text("white paper"))

    best_ten = sorted(scores, key=lambda relative_path: -scores[relative_path])[:10]
    assert all(relative_path.startswith("receipts/") for relative_path in best_ten)
