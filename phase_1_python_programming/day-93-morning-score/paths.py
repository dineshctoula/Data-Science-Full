"""Point Day 93 at Day 92's client and Day 91's server.

The score is a join: a staging row from Day 92, and the morning rides
that actually happened. This folder must not ship a project_paths.py.
Day 91's fit does `import project_paths` and expects that name to add
the Day 86 grid. Day 92 already solved that by loading Day 91 under a
different bootstrap name; this day reuses that bootstrap.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REBALANCE = HERE.parent / "day-92-morning-rebalance"

if not REBALANCE.is_dir():
    raise ImportError(f"expected the rebalancing client at {REBALANCE}")
# appended, so this day's modules win any name they share
if str(REBALANCE) not in sys.path:
    sys.path.append(str(REBALANCE))

import server_path  # noqa: F401  (puts Day 91 on the path, then Day 86)
