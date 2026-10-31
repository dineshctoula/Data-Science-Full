"""Point Day 89 at the earlier project days.

Day 86: grid, split, metric, baselines. Day 87: correction features and
the ridge. Day 88: forecast-grade weather. Their modules run
`import project_paths`, which inside this day resolves to this file, so
all three go on the path here. Day 85's test gate is reused by file path
(its folder holds a final_engine.py and final_recipe.py of its own, which
must never shadow this day's).
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KICKOFF = HERE.parent / "day-86-bike-kickoff"
FIRST_MODEL = HERE.parent / "day-87-bike-first-model"
FORECAST_WEATHER = HERE.parent / "day-88-bike-forecast-weather"
CHURN_FINAL = HERE.parent / "day-85-churn-final-test"
DATA_DIR = KICKOFF / "data"
SPLIT_FILE = KICKOFF / "split_dates.tsv"

for folder in (KICKOFF, FIRST_MODEL, FORECAST_WEATHER):
    if not folder.is_dir():
        raise ImportError(f"expected the earlier project day at {folder}")
    # appended, so this day's visualizer.py and main.py win the name lookup
    if str(folder) not in sys.path:
        sys.path.append(str(folder))


def load_from(folder: Path, module: str):
    """Import a module from an earlier day by file path, under a name of its own."""
    spec = importlib.util.spec_from_file_location(f"{folder.name.replace('-', '_')}_{module}", folder / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
