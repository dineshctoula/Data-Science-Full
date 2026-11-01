"""Tests for the repo tools: the catalog, the runner, the health sweep, and the index.

Run from this folder: python3 -m unittest test_tools
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path

from build_index import render
from check_all import parse_range, parse_unittest
from daycatalog import DAYS_DIR, FOLDER, discover, progress_log
from run_day import command_for, split_extra

PROGRESS_SNIPPET = """
| Day | Date | Phase | Topic | Key Accomplishments & Deliverables | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **01** | 2026-06-15 | Phase 1 | Python Setup | Did things. | ✅ |
| **79** | 2026-10-18 | Phase 4 | Dropout | Built a net. | ✅ |
"""


def _tree(tmp: Path, folders: dict) -> Path:
    """folders: {name: {filename: text}} under tmp/days."""
    days = tmp / "days"
    for name, files in folders.items():
        (days / name).mkdir(parents=True)
        for filename, text in files.items():
            (days / name / filename).write_text(text)
    return days


class CatalogTests(unittest.TestCase):
    def test_progress_rows_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PROGRESS.md"
            path.write_text(PROGRESS_SNIPPET)
            self.assertEqual(progress_log(path), {1: ("2026-06-15", "Python Setup"), 79: ("2026-10-18", "Dropout")})

    def test_titles_fall_back_from_log_to_notes_to_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "PROGRESS.md").write_text(PROGRESS_SNIPPET)
            days_dir = _tree(
                tmp,
                {
                    "day-01-setup": {},
                    "day-02-basic-types": {"TYPE_NOTES.md": "# Day 2 — Numbers and strings\n\ntext"},
                    "day-03-control-flow": {"main.py": "print(1)", "test_flow.py": ""},
                },
            )
            days = discover(days_dir, tmp / "PROGRESS.md")
            self.assertEqual([d.title for d in days], ["Python Setup", "Numbers and strings", "Control Flow"])
            self.assertEqual(days[0].date, "2026-06-15")
            self.assertTrue(days[2].runnable)
            self.assertEqual([t.name for t in days[2].tests], ["test_flow.py"])
            self.assertEqual(days[1].phase, "Phase 1 — Python & NumPy Fundamentals")

    def test_bad_names_and_repeated_numbers_are_refused(self):
        for folders in ({"day 12. numpy arrays": {}}, {"day-05-a": {}, "day-05-b": {}}):
            with tempfile.TemporaryDirectory() as tmp:
                days_dir = _tree(Path(tmp), folders)
                with self.assertRaises(ValueError):
                    discover(days_dir, Path(tmp) / "none.md")

    def test_every_real_folder_follows_the_pattern(self):
        names = [p.name for p in DAYS_DIR.iterdir() if p.is_dir() and not p.name.startswith((".", "__"))]
        self.assertGreaterEqual(len(names), 89)
        self.assertEqual([n for n in names if not FOLDER.match(n)], [])
        numbers = sorted(d.number for d in discover())
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))


class RunnerTests(unittest.TestCase):
    def test_arguments_after_a_bare_dash_dash_go_to_main(self):
        self.assertEqual(split_extra(["85", "--tests"]), (["85", "--tests"], []))
        self.assertEqual(split_extra(["85", "--", "--rehearse", "-v"]), (["85"], ["--rehearse", "-v"]))

    def test_commands_and_a_missing_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            days_dir = _tree(Path(tmp), {"day-04-loop": {"day_04_loop.py": ""}, "day-05-x": {"main.py": "", "test_x.py": ""}})
            loop, x = discover(days_dir, Path(tmp) / "none.md")
            self.assertEqual(command_for(x, tests=True), [sys.executable, "-m", "unittest", "test_x"])
            self.assertEqual(command_for(x, tests=False, extra=["--fast"]), [sys.executable, "main.py", "--fast"])
            with self.assertRaisesRegex(SystemExit, "day_04_loop.py"):
                command_for(loop, tests=False)
            with self.assertRaisesRegex(SystemExit, "no test_"):
                command_for(loop, tests=True)


class SweepTests(unittest.TestCase):
    def test_unittest_endings_are_read(self):
        self.assertEqual(parse_unittest("....\nRan 4 tests in 0.1s\n\nOK\n", 0), (4, "ok", ""))
        self.assertEqual(parse_unittest("Ran 5 tests in 1s\n\nOK (skipped=2)\n", 0), (5, "ok", "skipped=2"))
        self.assertEqual(parse_unittest("Ran 3 tests in 1s\n\nFAILED (failures=1)\n", 1), (3, "failed", "failures=1"))
        tests, status, detail = parse_unittest("Traceback...\nModuleNotFoundError: No module named 'x'\n", 1)
        self.assertEqual((tests, status), (None, "error"))
        self.assertIn("ModuleNotFoundError", detail)

    def test_day_ranges(self):
        self.assertEqual(parse_range("81-89"), (81, 89))
        self.assertEqual(parse_range("7"), (7, 7))
        for bad in ("9-3", "a-b", "1,2"):
            with self.assertRaises(ValueError):
                parse_range(bad)


class IndexTests(unittest.TestCase):
    def test_every_day_is_listed_with_only_real_links(self):
        text = render(discover())
        for day in discover():
            self.assertIn(f"| {day.number} |", text)
        repo = DAYS_DIR.parent
        links = re.findall(r"\]\(([^)]+)\)", text)
        self.assertTrue(links)
        self.assertEqual([l for l in links if not (repo / l).exists()], [])


if __name__ == "__main__":
    unittest.main()
