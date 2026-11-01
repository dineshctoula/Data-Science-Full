"""Run any day's walkthrough or tests from anywhere in the repo.

    python3 tools/run_day.py --list              every day, and what it can run
    python3 tools/run_day.py 87                  that day's main.py
    python3 tools/run_day.py 87 --tests          that day's test suite
    python3 tools/run_day.py 85 -- --rehearse    extra arguments go to main.py

Each day runs inside its own folder, because the walkthroughs read and
write relative paths (output/, data/, split files) next to themselves.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from daycatalog import discover, find


def list_days(days) -> str:
    lines = [f"{'day':>3}  {'main':<4}  {'tests':<5}  topic"]
    for d in days:
        tests = str(sum(1 for _ in d.tests)) if d.tests else "-"
        lines.append(f"{d.number:>3}  {'yes' if d.runnable else '-':<4}  {tests:<5}  {d.title}")
    return "\n".join(lines)


def command_for(day, tests: bool, extra=()) -> list:
    """The exact command to run, so callers (and tests) can see it without running it."""
    if tests:
        if not day.tests:
            raise SystemExit(f"day {day.number} has no test_*.py files")
        # module names, not paths: unittest imports them from the day's folder
        return [sys.executable, "-m", "unittest", *[t.stem for t in day.tests]]
    if not day.runnable:
        scripts = sorted(p.name for p in day.path.glob("*.py"))
        hint = f"; its scripts are: {', '.join(scripts)}" if scripts else ""
        raise SystemExit(f"day {day.number} has no main.py{hint}")
    return [sys.executable, "main.py", *extra]


def split_extra(argv):
    """(runner args, main.py args): everything after the first bare '--' belongs to main.py.

    argparse's REMAINDER would also swallow flags like --tests that come after
    the day number, so the split is done by hand before parsing.
    """
    argv = list(argv)
    if "--" in argv:
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1 :]
    return argv, []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("day", nargs="?", type=int, help="day number, e.g. 87")
    parser.add_argument("--tests", action="store_true", help="run the day's test suite instead of main.py")
    parser.add_argument("--list", action="store_true", help="list every day")
    own, extra = split_extra(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(own)

    if args.list or args.day is None:
        print(list_days(discover()))
        return 0
    day = find(args.day)
    cmd = command_for(day, args.tests, extra)
    print(f"[day {day.number}: {day.title}] {' '.join(cmd[1:])}  (in {day.rel()})", flush=True)
    return subprocess.run(cmd, cwd=day.path).returncode


if __name__ == "__main__":
    sys.exit(main())
