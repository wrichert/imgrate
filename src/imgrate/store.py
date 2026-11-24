"""Persistent rating storage in SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from imgrate.ratings import MAX_STARS, REJECTED

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class RatingStore:
    """Ratings keyed by path relative to the photo root, so the folder can be moved.

    Rating values: ``0`` = rejected, ``1..5`` = stars. Unrated files have no row.
    """

    DEFAULT_FILE_NAME = ".imgrate.sqlite"
    _ENABLED_EXTENSIONS_KEY = "enabled_extensions"
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS ratings (
            relative_path TEXT PRIMARY KEY,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 0 AND 5),
            updated_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._connection = sqlite3.connect(database_path)
        self._connection.execute("PRAGMA journal_mode=WAL")
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

    def close(self) -> None:
        self._connection.close()
