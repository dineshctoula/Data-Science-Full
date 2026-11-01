# Day 90 — Portfolio pass

No new model today. With both projects closed, the repo got the same treatment the projects did: one convention, tools that enforce it, and a health check.

## One naming pattern for every folder

Days 1–21 had been named five different ways: `day_01_setup`, `day_09_Exception handling`, `day_11. numpy`, `day 12. numpy arrays`, and `day-19. pandas dataframe`. Spaces and dots meant every shell command needed quotes, and no script could find "Day 12" reliably. All 20 are now `day-NN-short-name`, like Days 22–89.

- Whole folders were moved, so ignored outputs came along, and git records pure renames (42 files, 0 lines changed).
- Nothing in the repo referred to the old names.
- Two `housing.csv` files had been committed before `.gitignore` covered CSVs. They're re-added explicitly, because Days 20–21 read them.
- A stale `numpy.cpython-312.pyc` in Day 13, left over from a long-gone `numpy.py` that would have shadowed the real NumPy, is no longer tracked.

## Tools (`tools/`)

| tool | what it does |
|---|---|
| `daycatalog.py` | finds every `day-NN-name` folder (refusing any that break the pattern or repeat a number), and records its walkthrough, tests, notes, phase, and project; titles and dates come from `PROGRESS.md` |
| `run_day.py` | `python3 tools/run_day.py 87` runs that day's `main.py`, `--tests` runs its suite, and anything after `--` goes to `main.py`; it always runs inside the day's folder |
| `check_all.py` | every suite in its own process, with a timeout, written up as a Markdown report; exits non-zero on any failure |
| `build_index.py` | writes `INDEX.md`; `--check` fails if it's stale |
| `test_tools.py` | 9 tests on a throwaway folder tree, plus checks that the real repo has no off-pattern folder, no gap in day numbers, and no dead index links |

One bug was caught while trying the runner: `run_day.py 79 --tests` ran the walkthrough instead of the tests. `argparse.REMAINDER` had swallowed the `--tests` flag. The arguments are now split at `--` by hand, and a test covers it.

## Health

[`HEALTH.md`](HEALTH.md) is a full sweep: **53 of 53 suites pass, 449 tests**. Days 1–36 have no suites; they're scripts and notebooks from before the testing habit started on Day 37. Day 90's own tests live in `tools/` (`cd tools && python3 -m unittest test_tools`).

## Documents

- [`INDEX.md`](../../INDEX.md): generated; both projects first, then every day by phase, with links to its folder, tests, and notes.
- [`LESSONS.md`](../../LESSONS.md): eleven habits both projects relied on, each with the numbers that taught it.
- [`README.md`](../../README.md): a quick start with the tools, a real workspace tree, and the five phases the repo actually follows. The dashboard's `curriculum.json` still carries the original seven-phase syllabus, and the README says so rather than pretending they match.
