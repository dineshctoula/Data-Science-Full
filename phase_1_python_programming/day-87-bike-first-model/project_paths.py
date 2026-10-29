"""Point Day 87 at the Day 86 project files instead of copying them.

The grid rebuild, the frozen time split, the metric, and the baselines
all live in day-86-bike-kickoff.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KICKOFF = HERE.parent / "day-86-bike-kickoff"
DATA_DIR = KICKOFF / "data"
SPLIT_FILE = KICKOFF / "split_dates.tsv"

if not KICKOFF.is_dir():
    raise ImportError(f"expected the Day 86 project at {KICKOFF}")
# appended, so this day's visualizer.py and main.py win the name lookup
if str(KICKOFF) not in sys.path:
    sys.path.append(str(KICKOFF))


def load_kickoff(module: str):
    """Import a same-named Day 86 module (e.g. visualizer) by file path."""
    spec = importlib.util.spec_from_file_location(f"bike_kickoff_{module}", KICKOFF / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
