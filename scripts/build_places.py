"""Build ``src/lazyrate/data/places.tsv.gz`` from a GeoNames dump (https://download.geonames.org/export/dump/).

Usage::

    curl -O https://download.geonames.org/export/dump/cities1000.zip   # then unzip
    curl -O https://download.geonames.org/export/dump/admin1CodesASCII.txt
    curl -O https://download.geonames.org/export/dump/countryInfo.txt
    uv run python scripts/build_places.py <folder with the three files>

GeoNames data is licensed CC BY 4.0. One row per town with at least 1,000 inhabitants:
``latitude  longitude  population  town  region  country``.
"""

from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "src" / "lazyrate" / "data" / "places.tsv.gz"
COORDINATE_DECIMALS = 4
"""1e-4 degrees is about 11 m: far finer than "nearest town" needs."""
SKIPPED_FEATURE_CODES = {"PPLX", "PPLH", "PPLQ", "PPLW"}
"""City districts (a Funchal photo should say Funchal, not its parish) and historical, abandoned, destroyed places."""


def load_country_names(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as country_file:
        rows = csv.reader((line for line in country_file if not line.startswith("#")), delimiter="\t")
        return {row[0]: row[4] for row in rows}


def load_region_names(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as region_file:
        return {row[0]: row[1] for row in csv.reader(region_file, delimiter="\t", quoting=csv.QUOTE_NONE)}


def main() -> None:
    source_folder = Path(sys.argv[1])
    country_names = load_country_names(source_folder / "countryInfo.txt")
    region_names = load_region_names(source_folder / "admin1CodesASCII.txt")
    row_count = 0
    with (
        (source_folder / "cities1000.txt").open(encoding="utf-8") as city_file,
        gzip.open(OUTPUT_PATH, "wt", encoding="utf-8", compresslevel=9) as output_file,
    ):
        for row in csv.reader(city_file, delimiter="\t", quoting=csv.QUOTE_NONE):
            name, latitude_deg, longitude_deg, feature_code = row[1], row[4], row[5], row[7]
            country_code, admin1_code, population = row[8], row[10], row[14] or "0"
            if feature_code in SKIPPED_FEATURE_CODES:
                continue
            region = region_names.get(f"{country_code}.{admin1_code}", "")
            country = country_names.get(country_code, country_code)
            latitude_text = f"{float(latitude_deg):.{COORDINATE_DECIMALS}f}"
            longitude_text = f"{float(longitude_deg):.{COORDINATE_DECIMALS}f}"
            output_file.write(f"{latitude_text}\t{longitude_text}\t{population}\t{name}\t{region}\t{country}\n")
            row_count += 1
    print(f"wrote {row_count:,} places to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
