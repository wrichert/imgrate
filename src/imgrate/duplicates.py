"""Grouping of near-duplicate images (bursts, re-saved copies, resized versions)."""

from __future__ import annotations

from dataclasses import dataclass

from imgrate.fingerprints import count_differing_bits


@dataclass(frozen=True, slots=True)
class HashedFile:
    relative_path: str
    difference_hash: int
    sort_timestamp: float
    """Capture time if known, otherwise file modification time."""


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Two or more similar files, ordered by time."""

    relative_paths: list[str]


def build_time_order_key(hashed_file: HashedFile) -> tuple[float, str]:
    return (hashed_file.sort_timestamp, hashed_file.relative_path)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self._parents = list(range(size))

    def find(self, index: int) -> int:
        while self._parents[index] != index:
            self._parents[index] = self._parents[self._parents[index]]
            index = self._parents[index]
        return index

    def union(self, first_index: int, second_index: int) -> None:
        self._parents[self.find(first_index)] = self.find(second_index)


class DuplicateFinder:
    """Links files whose hashes differ in at most ``max_differing_bits`` and that were taken close in time.

    The time window keeps the comparison near-linear and stops unrelated, merely similar-looking
    photos (e.g. two sunsets a year apart) from being chained into one giant group.
    """

    DEFAULT_MAX_DIFFERING_BITS = 10
    DEFAULT_MAX_TIME_GAP_SECONDS = 15 * 60
    MAX_COMPARISONS_PER_FILE = 300
    """Bounds the work when many files share one timestamp (e.g. copies that lost their dates)."""

    def __init__(
        self,
        *,
        max_differing_bits: int = DEFAULT_MAX_DIFFERING_BITS,
        max_time_gap_seconds: float = DEFAULT_MAX_TIME_GAP_SECONDS,
    ) -> None:
        if max_differing_bits < 0:
            raise ValueError(f"max_differing_bits must be >= 0, got {max_differing_bits}")
        self._max_differing_bits = max_differing_bits
        self._max_time_gap_seconds = max_time_gap_seconds

    def __repr__(self) -> str:
        return (
            f"DuplicateFinder(max_differing_bits={self._max_differing_bits}, "
            f"max_time_gap_seconds={self._max_time_gap_seconds})"
        )

    def find_groups(self, hashed_files: list[HashedFile]) -> list[DuplicateGroup]:
        """Return groups of size >= 2, ordered by the time of their first member."""
        ordered_files = sorted(hashed_files, key=build_time_order_key)
        disjoint_set = _DisjointSet(len(ordered_files))
        for first_index, first_file in enumerate(ordered_files):
            last_index = min(len(ordered_files), first_index + 1 + self.MAX_COMPARISONS_PER_FILE)
            for second_index in range(first_index + 1, last_index):
                second_file = ordered_files[second_index]
                if second_file.sort_timestamp - first_file.sort_timestamp > self._max_time_gap_seconds:
                    break
                differing_bits = count_differing_bits(first_file.difference_hash, second_file.difference_hash)
                if differing_bits <= self._max_differing_bits:
                    disjoint_set.union(first_index, second_index)

        members_by_root: dict[int, list[str]] = {}
        for index, hashed_file in enumerate(ordered_files):
            members_by_root.setdefault(disjoint_set.find(index), []).append(hashed_file.relative_path)
        # dict preserves insertion order, and members were appended in time order.
        return [DuplicateGroup(relative_paths=members) for members in members_by_root.values() if len(members) > 1]
