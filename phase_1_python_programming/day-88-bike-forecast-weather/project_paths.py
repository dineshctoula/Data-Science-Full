"""Point Day 88 at the Day 86 and Day 87 project files.

Day 86 owns the grid, the split, the metric, and the baselines; Day 87
owns the correction features and the ridge. Day 87's modules run
`import project_paths` themselves, which inside this day resolves to this
file, so Day 86 goes on the path here as well.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KICKOFF = HERE.parent / "day-86-bike-kickoff"
FIRST_MODEL = HERE.parent / "day-87-bike-first-model"
DATA_DIR = KICKOFF / "data"
SPLIT_FILE = KICKOFF / "split_dates.tsv"

for folder in (KICKOFF, FIRST_MODEL):
    if not folder.is_dir():
        raise ImportError(f"expected the earlier project day at {folder}")
    # appended, so this day's visualizer.py and main.py win the name lookup
    if str(folder) not in sys.path:
        sys.path.append(str(folder))


def load_from(folder: Path, module: str):
    """Import a same-named module (e.g. an earlier day's visualizer) by file path."""
    spec = importlib.util.spec_from_file_location(f"{folder.name.replace('-', '_')}_{module}", folder / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
