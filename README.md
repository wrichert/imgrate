# imgrate

Terminal UI to rate thousands of photos fast and keep only the best ones (e.g. for a year calendar).

- Recursively scans a folder (hidden folders and NAS metadata like `@eaDir` are skipped)
- Real image previews in the terminal: Kitty graphics protocol (Ghostty, Kitty, WezTerm), Sixel (iTerm2, foot, …), unicode fallback elsewhere
- Background decoding + prefetch of the next images, JPEG draft-mode decoding (24 MP → ~25 ms), EXIF orientation, HEIC/HEIF
- Ratings stored immediately in SQLite (`ROOT/.imgrate.sqlite`), keyed by path relative to the root, so the folder can be moved
- Review any rating bucket (unrated, rejected, ≥ N stars, any combination) in single view or as a thumbnail grid
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
imgrate export ~/Pictures/2025 ~/calendar --min-rating 5 [--flat] [--symlink] [--dry-run]
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
| `n` | next unrated |
| `v` / `V` | cycle view: all · unrated · not rejected · ≥★ … ★★★★★ · rejected |
| `f` | choose any combination of ratings to view |
| `g` | toggle thumbnail grid (`enter` opens the highlighted image) |
| `e` | choose file formats |
| `o` | open in the system viewer (also for videos) |
| `r` | rescan folder |
| `?` | help |

Within a view the list stays stable while you rate, so rating through "unrated" doesn't make the cursor jump;
press `v`/`f` again to re-apply the filter.

### Suggested workflow for a calendar

1. View **unrated**, go through quickly: `x` for rejects, `3` for "maybe", `4`/`5` for good ones.
2. View **≥ ★★★★** in grid mode (`g`), compare, promote the best to `5`, demote the rest.
3. View **★★★★★ only** until 12–13 remain, then `imgrate export ROOT ~/calendar --min-rating 5 --flat`.

## Development

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format .
```
