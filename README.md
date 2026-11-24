# imgrate

Terminal UI to rate thousands of photos fast and keep only the best ones (e.g. for a year calendar).

- Recursively scans a folder (hidden folders and NAS metadata like `@eaDir` are skipped)
- Real image previews in the terminal: Kitty graphics protocol (Ghostty, Kitty, WezTerm), Sixel (iTerm2, foot, …), unicode fallback elsewhere
- Background decoding + prefetch of the next images, JPEG draft-mode decoding (24 MP → ~25 ms), EXIF orientation, HEIC/HEIF
- Ratings stored immediately in SQLite (`ROOT/.imgrate.sqlite`), keyed by path relative to the root, so the folder can be moved
- Sort by path, rating (best first) or date taken
- Near-duplicate detection (bursts, re-saved or resized copies): perceptual hash computed once in the background
  and cached in SQLite; review one group at a time and keep the best with a single key
- Review any rating bucket (unrated, rejected, ≥ N stars, any combination) in single view or as a thumbnail grid
- Export from the app (`E`) or the shell, by copy or symlink, flat or with sub folders
- Choose which file formats take part (images on by default; `.mp4`, `.mov`, RAW etc. off unless you enable them)
- Export the keepers by copy or symlink

## Install

```sh
uv tool install git+https://github.com/wrichert/imgrate
# or, from a checkout
uv sync && uv run imgrate ~/Pictures/2025
```

## Usage

```sh
imgrate ~/Pictures/2025                       # rate (TUI)
imgrate stats ~/Pictures/2025                 # counts per rating
imgrate export ~/Pictures/2025 ~/calendar --min-rating 4 [--flat] [--symlink] [--dry-run]  # 4 and 5 stars
imgrate export ~/Pictures/2025 ~/maybe --rating 3            # exactly 3 stars (repeatable; 0 = rejected)
imgrate rate ~/Pictures/2025 --db ~/ratings.sqlite   # keep the DB elsewhere (e.g. read-only folder)
```

### Keys

| Key | Action |
| --- | --- |
| `1`–`5` | rate (auto-advances; toggle with `a`) |
| `0` / `x` | reject |
| `u` | clear rating |
| `z` | undo |
| `→` `l` `space` / `←` `h` | next / previous |
| `↓` `↑` | next / previous (grid: row) |
| `pgdn` `pgup`, `home` `end` | jump 10 (grid: page), first / last |
| `J` / `K` | next / previous section (folder, rating, month or duplicate group, depending on sort); also `]` `.` / `[` `,` |
| `n` | next unrated |
| `v` / `V` | cycle view: all · unrated · not rejected · ≥★ … ★★★★★ · rejected |
| `f` | choose any combination of ratings to view |
| `s` | cycle sort: rating (best first, default) · date taken · path |
| `S` | re-sort / re-filter now (re-rated photos move to their new section) |
| `g` | toggle thumbnail grid (default) / single image (`enter` in the grid opens the highlighted image) |
| `d` | toggle duplicates mode (each group is a section) |
| `b` | duplicates mode: keep this one, reject the rest of the group, jump to next group |
| `E` | export: pick ratings, destination, flat / symlink (remembered for next time) |
| `e` | choose file formats |
| `o` | open in the system viewer (also for videos) |
| `c` | copy the full path to the clipboard |
| `r` | rescan folder |
| `?` | help |

### Near duplicates

Each image gets a 64-bit difference hash (grayscale 9×8 thumbnail). Two images are grouped when their hashes
differ in ≤ 10 bits and they were taken ≤ 15 minutes apart (EXIF capture time, else file time). This reliably
catches bursts, re-saved and resized copies; it misses the same scene with a different crop or zoom.
Indexing takes about 3–7 ms per photo, runs in the background on start, and is reused on later runs
(recomputed only for changed files).

### Sections

The list is split into sections depending on the sort order: folders (path), ratings (rating), months
(date taken) or duplicate groups. The grid shows a header line per section and starts each section on a new row;
`J`/`K` (or `]`/`.` and `[`/`,`) jump between sections, also in the single-image view. The sidebar lists the
sections around the current one with their sizes.

The app starts in the grid, sorted by rating. Within a view the list stays stable while you rate, so rating
doesn't make the cursor jump; press `S` to re-sort and re-filter (re-rated photos move to their new section).

### Suggested workflow for a calendar

1. Press `d` and `g`: go through the duplicate groups, `b` on the best shot of each burst.
   Press `d` again to leave.
2. View **unrated**, go through quickly: `x` for rejects, `3` for "maybe", `4`/`5` for good ones.
3. View **≥ ★★★★** in grid mode (`g`), compare, promote the best to `5`, demote the rest.
4. View **★★★★★ only**, `s` to sort by date taken (one section per month), until 12–13 remain, then `E` to export.

## Development

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format .
```
