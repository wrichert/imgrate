"""Persistent rating storage in SQLite."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from lazyrate.embeddings import StoredEmbedding
from lazyrate.fingerprints import Fingerprint
from lazyrate.locations import GeoCoordinates, PhotoLocation, Place
from lazyrate.ratings import MAX_STARS, REJECTED

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


SIGNED_64_BIT_OFFSET = 1 << 64
MAX_SIGNED_64_BIT = (1 << 63) - 1


def to_signed_64_bit(value: int) -> int:
    """SQLite integers are signed 64 bit; hashes are unsigned."""
    return value - SIGNED_64_BIT_OFFSET if value > MAX_SIGNED_64_BIT else value


def to_unsigned_64_bit(value: int) -> int:
    return value + SIGNED_64_BIT_OFFSET if value < 0 else value


LEGACY_DATABASE_FILE_NAME = ".imgrate.sqlite"
"""Database name before the rename to lazyrate."""
SQLITE_SIDECAR_SUFFIXES = ("", "-wal", "-shm")


def find_legacy_database(root: Path) -> Path | None:
    """The pre-rename database in ``root``, if there is one and no lazyrate database exists yet."""
    legacy_path = root / LEGACY_DATABASE_FILE_NAME
    if not legacy_path.is_file() or (root / RatingStore.DEFAULT_FILE_NAME).exists():
        return None
    return legacy_path


def migrate_legacy_database(root: Path) -> None:
    """Rename ``.imgrate.sqlite`` (plus its WAL / SHM files) to ``.lazyrate.sqlite``; no-op if there is none."""
    legacy_path = find_legacy_database(root)
    if legacy_path is None:
        return
    for suffix in SQLITE_SIDECAR_SUFFIXES:
        legacy_file = legacy_path.with_name(legacy_path.name + suffix)
        if legacy_file.exists():
            legacy_file.rename(root / (RatingStore.DEFAULT_FILE_NAME + suffix))


def build_photo_location(row: tuple) -> PhotoLocation:
    relative_path, size_bytes, modified_timestamp_utc, latitude_deg, longitude_deg, town, region, country = row
    has_coordinates = latitude_deg is not None and longitude_deg is not None
    return PhotoLocation(
        relative_path=relative_path,
        size_bytes=size_bytes,
        modified_timestamp_utc=modified_timestamp_utc,
        coordinates=GeoCoordinates(latitude_deg=latitude_deg, longitude_deg=longitude_deg) if has_coordinates else None,
        place=Place(town=town or "", region=region or "", country=country or "") if has_coordinates else None,
    )


class RatingStore:
    """Ratings keyed by path relative to the photo root, so the folder can be moved.

    Rating values: ``0`` = rejected, ``1..5`` = stars. Unrated files have no row.
    """

    DEFAULT_FILE_NAME = ".lazyrate.sqlite"
    _ENABLED_EXTENSIONS_KEY = "enabled_extensions"
    _WAL_SWITCH_ATTEMPTS = 50
    _WAL_SWITCH_RETRY_DELAY_SECONDS = 0.05
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS ratings (
            relative_path TEXT PRIMARY KEY,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 0 AND 5),
            updated_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fingerprints (
            relative_path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            modified_timestamp_utc REAL NOT NULL,
            difference_hash INTEGER,
            capture_timestamp REAL
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            relative_path TEXT NOT NULL,
            model_name TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            modified_timestamp_utc REAL NOT NULL,
            vector BLOB NOT NULL,
            PRIMARY KEY (relative_path, model_name)
        );
        CREATE TABLE IF NOT EXISTS locations (
            relative_path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            modified_timestamp_utc REAL NOT NULL,
            latitude_deg REAL,
            longitude_deg REAL,
            town TEXT,
            region TEXT,
            country TEXT
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        # timeout: the background indexers (fingerprints, GPS, embeddings) write through their own connections.
        self._connection = sqlite3.connect(database_path, timeout=10)
        self._switch_to_write_ahead_log()
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.executescript(self._SCHEMA)
        self._connection.commit()

    def __repr__(self) -> str:
        return f"RatingStore({self._database_path!s})"

    @property
    def database_path(self) -> Path:
        return self._database_path

    def load_ratings(self) -> dict[str, int]:
        """Return all ratings as ``{relative_path: rating}``."""
        rows = self._connection.execute("SELECT relative_path, rating FROM ratings").fetchall()
        return dict(rows)

    def set_rating(self, *, relative_path: str, rating: int | None) -> None:
        """Store a rating; ``None`` removes it (file becomes unrated)."""
        if rating is None:
            self._connection.execute("DELETE FROM ratings WHERE relative_path = ?", (relative_path,))
        else:
            if not REJECTED <= rating <= MAX_STARS:
                raise ValueError(f"Rating must be between {REJECTED} and {MAX_STARS}, got {rating}")
            self._connection.execute(
                "INSERT INTO ratings (relative_path, rating, updated_at_utc) VALUES (?, ?, ?) "
                "ON CONFLICT(relative_path) DO UPDATE SET rating = excluded.rating, "
                "updated_at_utc = excluded.updated_at_utc",
                (relative_path, rating, datetime.now(UTC).isoformat()),
            )
        self._connection.commit()

    def load_enabled_extensions(self) -> set[str] | None:
        """Return the persisted extension selection, or ``None`` if never saved."""
        row = self._connection.execute(
            "SELECT value FROM settings WHERE key = ?", (self._ENABLED_EXTENSIONS_KEY,)
        ).fetchone()
        return None if row is None else set(json.loads(row[0]))

    def save_enabled_extensions(self, extensions: Iterable[str]) -> None:
        self._connection.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (self._ENABLED_EXTENSIONS_KEY, json.dumps(sorted(extensions))),
        )
        self._connection.commit()

    def load_fingerprints(self) -> dict[str, Fingerprint]:
        rows = self._connection.execute(
            "SELECT relative_path, size_bytes, modified_timestamp_utc, difference_hash, capture_timestamp "
            "FROM fingerprints"
        ).fetchall()
        return {
            relative_path: Fingerprint(
                relative_path=relative_path,
                size_bytes=size_bytes,
                modified_timestamp_utc=modified_timestamp_utc,
                difference_hash=None if difference_hash is None else to_unsigned_64_bit(difference_hash),
                capture_timestamp=capture_timestamp,
            )
            for relative_path, size_bytes, modified_timestamp_utc, difference_hash, capture_timestamp in rows
        }

    def save_fingerprints(self, fingerprints: Iterable[Fingerprint]) -> None:
        self._connection.executemany(
            "INSERT OR REPLACE INTO fingerprints "
            "(relative_path, size_bytes, modified_timestamp_utc, difference_hash, capture_timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    fingerprint.relative_path,
                    fingerprint.size_bytes,
                    fingerprint.modified_timestamp_utc,
                    None if fingerprint.difference_hash is None else to_signed_64_bit(fingerprint.difference_hash),
                    fingerprint.capture_timestamp,
                )
                for fingerprint in fingerprints
            ],
        )
        self._connection.commit()

    def load_embeddings(self, model_name: str) -> dict[str, StoredEmbedding]:
        rows = self._connection.execute(
            "SELECT relative_path, size_bytes, modified_timestamp_utc, vector FROM embeddings WHERE model_name = ?",
            (model_name,),
        ).fetchall()
        return {
            relative_path: StoredEmbedding(
                relative_path=relative_path,
                size_bytes=size_bytes,
                modified_timestamp_utc=modified_timestamp_utc,
                model_name=model_name,
                vector=np.frombuffer(vector, dtype=np.float16).astype(np.float32),
            )
            for relative_path, size_bytes, modified_timestamp_utc, vector in rows
        }

    def save_embeddings(self, embeddings: Iterable[StoredEmbedding]) -> None:
        """Vectors are stored as float16 (1 KB for 512 dimensions); plenty for cosine similarity and linear models."""
        self._connection.executemany(
            "INSERT OR REPLACE INTO embeddings "
            "(relative_path, model_name, size_bytes, modified_timestamp_utc, vector) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    embedding.relative_path,
                    embedding.model_name,
                    embedding.size_bytes,
                    embedding.modified_timestamp_utc,
                    embedding.vector.astype(np.float16).tobytes(),
                )
                for embedding in embeddings
            ],
        )
        self._connection.commit()

    def load_locations(self) -> dict[str, PhotoLocation]:
        """GPS position and nearest town per file; rows without coordinates mark files that have no GPS data."""
        rows = self._connection.execute(
            "SELECT relative_path, size_bytes, modified_timestamp_utc, latitude_deg, longitude_deg, town, region, "
            "country FROM locations"
        ).fetchall()
        return {row[0]: build_photo_location(row) for row in rows}

    def save_locations(self, locations: Iterable[PhotoLocation]) -> None:
        self._connection.executemany(
            "INSERT OR REPLACE INTO locations (relative_path, size_bytes, modified_timestamp_utc, latitude_deg, "
            "longitude_deg, town, region, country) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    location.relative_path,
                    location.size_bytes,
                    location.modified_timestamp_utc,
                    None if location.coordinates is None else location.coordinates.latitude_deg,
                    None if location.coordinates is None else location.coordinates.longitude_deg,
                    None if location.place is None else location.place.town,
                    None if location.place is None else location.place.region,
                    None if location.place is None else location.place.country,
                )
                for location in locations
            ],
        )
        self._connection.commit()

    def load_setting(self, key: str) -> str | None:
        row = self._connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def save_setting(self, *, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def _switch_to_write_ahead_log(self) -> None:
        """WAL mode is stored in the file, so usually only a new database switches. Switching ignores the busy
        timeout and fails with "database is locked" while another connection (a background indexer opening the
        same new file) is busy, hence the retries."""
        for _ in range(self._WAL_SWITCH_ATTEMPTS):
            try:
                if self._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal":
                    return
                self._connection.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                time.sleep(self._WAL_SWITCH_RETRY_DELAY_SECONDS)
        self._connection.execute("PRAGMA journal_mode=WAL")
