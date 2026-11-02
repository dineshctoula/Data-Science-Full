"""Point Day 91 at the bike-project modules instead of copying them.

Day 86 has the hourly grid and the frozen split. Day 87 has the
no-weather design and the ridge that corrects the baseline.

Day 89's recipe and test gate are not imported. This server must not
be a second way to open the test quarter.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KICKOFF = HERE.parent / "day-86-bike-kickoff"
FIRST_MODEL = HERE.parent / "day-87-bike-first-model"
DATA_DIR = KICKOFF / "data"
SPLIT_FILE = KICKOFF / "split_dates.tsv"

for folder in (KICKOFF, FIRST_MODEL):
    if not folder.is_dir():
        raise ImportError(f"expected an earlier project day at {folder}")
    # appended, so this day's modules win any name they share
    if str(folder) not in sys.path:
        sys.path.append(str(folder))
