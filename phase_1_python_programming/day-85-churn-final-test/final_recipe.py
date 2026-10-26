"""Day 85 — every choice for the final test, written down before test is opened.

This file is committed on its own, ahead of any code that reads the
test rows, so the git history shows the recipe was fixed first. The test
gate fingerprints this file (and final_engine.py) the first time test is
opened and refuses to run again if either has changed since.

Where each choice came from:
- model and penalty: Day 82, picked on validation log loss
- boosting: tried on Day 83, tied on ranking, lost on log loss -> not used
- call-list rule and economics: Day 84
"""

from __future__ import annotations

import project_paths  # noqa: F401
from economics import Economics

# the champion: L2 logistic (Newton), mirrors collapsed, log1p(tenure) added
VARIANT = "collapsed, + log tenure"
LAM = 1e-3

# With every choice made, validation has no job left, so the final model
# is refit on train + validation (5986 rows). The train-only model is also
# scored on test, as a check on how much the extra 1056 rows changed.
FIT_ON = ("train", "validation")

# the Day 81 lookup, refit the same way, as the floor
BASELINE_KEYS = ("Contract", "tenure_band", "InternetService")

# Day 84's placeholder economics and its two lists
ECONOMICS = Economics(call_cost=10.0, credit=20.0, save_rate=0.25, months_kept=12.0, margin=0.5)
TOP_FRAC = 0.20

# bootstrap settings for every interval on the test card
N_BOOT = 2000
SEED = 85
