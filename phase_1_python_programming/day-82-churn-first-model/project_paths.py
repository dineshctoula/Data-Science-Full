"""Point Day 82 at the Day 81 project files instead of copying them.

The loader, the frozen split, and the metric live in day-81-churn-kickoff.
Importing them (rather than pasting a second copy) means there's one
definition of "the test rows" and one definition of top-20% recall.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KICKOFF = HERE.parent / "day-81-churn-kickoff"
DATA_DIR = KICKOFF / "data"
SPLIT_FILE = KICKOFF / "split_ids.tsv"

if not KICKOFF.is_dir():
    raise ImportError(f"expected the Day 81 project at {KICKOFF}")
if str(KICKOFF) not in sys.path:
    sys.path.insert(0, str(KICKOFF))
