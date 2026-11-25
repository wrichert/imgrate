"""Image embeddings ("what is in this photo" as 512 numbers): computed once per file in the background, cached.

The CLIP model is optional (``pip install 'lazyrate[ml]'``) and imported lazily, so the core app never pays for torch.
"""

from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from lazyrate.preview import load_preview

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

EMBEDDING_DECODE_EDGE_PX = 448
"""Decode size for embedding; CLIP itself looks at 224 x 224."""


@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    relative_path: str
    size_bytes: int
    modified_timestamp_utc: float
    model_name: str
    vector: np.ndarray
    """L2-normalised float32."""


@dataclass(frozen=True, slots=True)
class EmbeddingMatrix:
    """All embeddings of one model as a matrix (row ``i`` belongs to ``relative_paths[i]``)."""

    relative_paths: list[str]
    vectors: np.ndarray

    @classmethod
    def from_embeddings(cls, embeddings: list[StoredEmbedding]) -> EmbeddingMatrix:
        if not embeddings:
            return cls(relative_paths=[], vectors=np.zeros((0, 0), dtype=np.float32))
        return cls(
            relative_paths=[embedding.relative_path for embedding in embeddings],
            vectors=np.stack([embedding.vector for embedding in embeddings]).astype(np.float32),
        )

    def rank_by_similarity(self, query_vector: np.ndarray) -> dict[str, float]:
        """Cosine similarity of every photo to ``query_vector`` (both sides are unit length)."""
        if not self.relative_paths:
            return {}
        similarities = self.vectors @ query_vector.astype(np.float32)
        return {
            relative_path: float(similarity)
            for relative_path, similarity in zip(self.relative_paths, similarities, strict=True)
        }


class ImageEmbedder(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Shape (n, d), rows L2-normalised."""
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """Shape (d,), L2-normalised, comparable with image embeddings."""
        ...


def is_embedding_available() -> bool:
    """Whether the optional CLIP dependencies are installed."""
    return importlib.util.find_spec("open_clip") is not None and importlib.util.find_spec("torch") is not None


def load_image_for_embedding(path: Path) -> Image.Image | None:
    return load_preview(path, max_edge_px=EMBEDDING_DECODE_EDGE_PX).image


class ClipEmbedder:
    """OpenCLIP ViT-B/32 (LAION-2B weights, the base of the LAION aesthetic predictor). Thread-safe.

    Construction loads the model (and downloads ~600 MB of weights on first use), so do it off the UI thread.
    """

    ARCHITECTURE = "ViT-B-32"
    PRETRAINED_TAG = "laion2b_s34b_b79k"
    MODEL_NAME = f"{ARCHITECTURE}/{PRETRAINED_TAG}"
    """Key of the cached embeddings; known without importing torch (e.g. for ``lazyrate stats``)."""

    def __init__(self) -> None:
        import open_clip
        import torch

        self._torch = torch
        self._device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(self.ARCHITECTURE, pretrained=self.PRETRAINED_TAG)
        self._model = model.to(self._device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.ARCHITECTURE)
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = self._torch.stack([self._preprocess(image) for image in images]).to(self._device)
        with self._lock, self._torch.no_grad():
            features = self._model.encode_image(batch)
        return _normalize_rows(features.float().cpu().numpy())

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenizer([text]).to(self._device)
        with self._lock, self._torch.no_grad():
            features = self._model.encode_text(tokens)
        return _normalize_rows(features.float().cpu().numpy())[0]


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)
