"""Point Day 82 at the Day 81 project files instead of copying them.

The loader, the frozen split, and the metric live in day-81-churn-kickoff.
Importing them (rather than pasting a second copy) means there's one
definition of "the test rows" and one definition of top-20% recall.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KICKOFF = HERE.parent / "day-81-churn-kickoff"
DATA_DIR = KICKOFF / "data"
SPLIT_FILE = KICKOFF / "split_ids.tsv"

if not KICKOFF.is_dir():
    raise ImportError(f"expected the Day 81 project at {KICKOFF}")
# appended, not inserted first: both days have a visualizer.py and a main.py,
# and this day's own files must win the name lookup
if str(KICKOFF) not in sys.path:
    sys.path.append(str(KICKOFF))


def load_kickoff(module: str):
    """Import a Day 81 module whose name this day also uses (e.g. visualizer)."""
    path = KICKOFF / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"kickoff_{module}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
