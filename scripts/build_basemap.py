"""Build ``src/lazyrate/data/basemap.npz`` from Natural Earth (public domain) for the offline map pane.

Usage::

    BASE=https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson
    curl -O $BASE/ne_10m_land.geojson
    curl -O $BASE/ne_50m_admin_0_boundary_lines_land.geojson
    uv run python scripts/build_basemap.py <folder with the two files>

Land outlines (10m: islands like Madeira keep their shape) and country borders (50m), as integer
longitude / latitude in 1e-3 degrees (~100 m), consecutive duplicate points removed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "src" / "lazyrate" / "data" / "basemap.npz"
COORDINATE_SCALE = 1000
"""Stored value = degrees x 1000."""


def collect_lines(geojson_path: Path) -> tuple[list[list[list[float]]], list[bool]]:
    """Every polygon ring or line string of the file, and per line whether it is a polygon hole (a lake)."""
    lines: list[list[list[float]]] = []
    hole_flags: list[bool] = []
    for feature in json.loads(geojson_path.read_text())["features"]:
        geometry = feature["geometry"]
        if geometry["type"] in {"Polygon", "MultiPolygon"}:
            polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
            for polygon in polygons:
                lines.extend(polygon)
                hole_flags.extend(ring_index > 0 for ring_index in range(len(polygon)))
        elif geometry["type"] in {"LineString", "MultiLineString"}:
            line_strings = [geometry["coordinates"]] if geometry["type"] == "LineString" else geometry["coordinates"]
            lines.extend(line_strings)
            hole_flags.extend(False for _ in line_strings)
    return lines, hole_flags


def quantize_lines(lines: list[list[list[float]]], hole_flags: list[bool]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All points as one ``(N, 2)`` int32 array, each line's start offset (plus the total), and the hole flags."""
    quantized_lines: list[np.ndarray] = []
    kept_hole_flags: list[bool] = []
    for line, is_hole in zip(lines, hole_flags, strict=True):
        points = np.round(np.asarray(line, dtype=np.float64)[:, :2] * COORDINATE_SCALE).astype(np.int32)
        is_new_point = np.ones(len(points), dtype=bool)
        is_new_point[1:] = np.any(points[1:] != points[:-1], axis=1)
        points = points[is_new_point]
        if len(points) >= 2:
            quantized_lines.append(points)
            kept_hole_flags.append(is_hole)
    offsets = np.cumsum([0] + [len(points) for points in quantized_lines]).astype(np.int64)
    return np.concatenate(quantized_lines), offsets, np.array(kept_hole_flags, dtype=bool)


def main() -> None:
    source_folder = Path(sys.argv[1])
    land_points, land_offsets, land_hole_flags = quantize_lines(*collect_lines(source_folder / "ne_10m_land.geojson"))
    border_points, border_offsets, _ = quantize_lines(
        *collect_lines(source_folder / "ne_50m_admin_0_boundary_lines_land.geojson")
    )
    np.savez_compressed(
        OUTPUT_PATH,
        land_points=land_points,
        land_offsets=land_offsets,
        land_hole_flags=land_hole_flags,
        border_points=border_points,
        border_offsets=border_offsets,
    )
    print(
        f"wrote {len(land_offsets) - 1:,} land rings ({land_hole_flags.sum():,} lakes, {len(land_points):,} points), "
        f"{len(border_offsets) - 1:,} borders to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
