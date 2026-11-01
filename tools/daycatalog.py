"""Discover every day folder and describe it.

Every day lives in phase_1_python_programming/day-NN-short-name (Day 90
renamed the early ones to match). A Day records what a reader or a script
needs: where it is, how to run it, where its tests and notes are, and which
phase or project it belongs to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DAYS_DIR = REPO / "phase_1_python_programming"
FOLDER = re.compile(r"^day-(\d{2,3})-([a-z0-9]+(?:-[a-z0-9]+)*)$")

# phase boundaries as PROGRESS.md lists them
PHASES = (
    (1, 15, "Phase 1 — Python & NumPy Fundamentals"),
    (16, 35, "Phase 2 — Data Manipulation & Visualization"),
    (36, 60, "Phase 3 — Math & Statistics for Data Science"),
    (61, 80, "Phase 4 — Machine Learning Fundamentals"),
    (81, 100, "Phase 5 — Projects & Portfolio Development"),
)
PROJECTS = (
    (81, 85, "Project 1 — Telco churn call list", 85),
    (86, 89, "Project 2 — Day-ahead bike demand", 89),
)
# notes files are named after the day's topic (DROP_NOTES.md, PROJECT.md, ...);
# README.md is the project write-up on a project's closing day
NOTE_FILE = re.compile(r"^([A-Z_]+NOTES|PROJECT|README)\.md$")


PROGRESS = REPO / "PROGRESS.md"
# a daily-log row: | **79** | 2026-10-18 | Phase 4 | Dropout | ... |
PROGRESS_ROW = re.compile(r"^\|\s*\*\*(\d+)\*\*\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|[^|]*\|\s*([^|]+?)\s*\|")


def progress_log(path: Path = PROGRESS) -> dict:
    """{day number: (date, topic)} from the daily table in PROGRESS.md, if it's there."""
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PROGRESS_ROW.match(line)
        if match:
            rows[int(match.group(1))] = (match.group(2), match.group(3))
    return rows


@dataclass
class Day:
    number: int
    slug: str
    path: Path
    title: str
    main: Path | None
    date: str | None = None
    tests: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    notebooks: list = field(default_factory=list)

    @property
    def phase(self) -> str:
        for lo, hi, name in PHASES:
            if lo <= self.number <= hi:
                return name
        return "unplaced"

    @property
    def project(self) -> str | None:
        for lo, hi, name, _ in PROJECTS:
            if lo <= self.number <= hi:
                return name
        return None

    @property
    def runnable(self) -> bool:
        return self.main is not None

    def rel(self, path: Path | None = None) -> str:
        """A path relative to the repo root, with forward slashes (for links)."""
        return (path or self.path).relative_to(REPO).as_posix()


def _title_from_notes(notes) -> str | None:
    """The first '# ' heading of the first notes file, minus a leading 'Day NN —'."""
    for note in notes:
        for line in note.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("# "):
                heading = line[2:].strip()
                # "Day 79 — Dropout" -> "Dropout"; "Project 2 — ..." stays as it is
                return re.sub(r"^Day\s+\d+\s*[—–-]\s*", "", heading)
    return None


def _title_from_slug(slug: str) -> str:
    small = {"and", "of", "vs", "to", "the"}
    words = slug.split("-")
    return " ".join(w if (i and w in small) else w.capitalize() for i, w in enumerate(words))


def discover(days_dir: Path = DAYS_DIR, progress: Path = PROGRESS) -> list:
    """Every day folder, sorted by day number. A folder off the pattern is an error.

    The title comes from PROGRESS.md's topic column when the day is logged
    there (short and consistent), else the notes heading, else the folder name.
    """
    log = progress_log(progress)
    days = []
    for folder in days_dir.iterdir():
        if not folder.is_dir() or folder.name.startswith((".", "__")):
            continue
        match = FOLDER.match(folder.name)
        if not match:
            # the naming rule is what lets every script and link find a day
            raise ValueError(f"{folder.name!r} doesn't follow day-NN-short-name")
        notes = sorted(p for p in folder.iterdir() if NOTE_FILE.match(p.name))
        main = folder / "main.py"
        number = int(match.group(1))
        date, topic = log.get(number, (None, None))
        days.append(
            Day(
                number=number,
                slug=match.group(2),
                path=folder,
                title=topic or _title_from_notes(notes) or _title_from_slug(match.group(2)),
                main=main if main.exists() else None,
                date=date,
                tests=sorted(folder.glob("test_*.py")),
                notes=notes,
                notebooks=sorted(folder.glob("*.ipynb")),
            )
        )
    numbers = [d.number for d in days]
    if len(numbers) != len(set(numbers)):
        raise ValueError("two folders claim the same day number")
    return sorted(days, key=lambda d: d.number)


def find(number: int, days_dir: Path = DAYS_DIR) -> Day:
    for day in discover(days_dir):
        if day.number == number:
            return day
    raise KeyError(f"no folder for day {number}")
