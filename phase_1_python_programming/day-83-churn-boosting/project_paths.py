"""Point Day 83 at the Day 81 and Day 82 project files.

Day 81 owns the data, the frozen split, and the metric. Day 82 owns the
feature sets, the Newton logistic, and the selection helpers. Both are
imported, not copied.

Day 82's own modules also run `import project_paths`; inside this day that
resolves to this file, so it has to put Day 81 on the path as well.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KICKOFF = HERE.parent / "day-81-churn-kickoff"
FIRST_MODEL = HERE.parent / "day-82-churn-first-model"
DATA_DIR = KICKOFF / "data"
SPLIT_FILE = KICKOFF / "split_ids.tsv"

for folder in (KICKOFF, FIRST_MODEL):
    if not folder.is_dir():
        raise ImportError(f"expected the earlier project day at {folder}")
    # appended, so this day's visualizer.py and main.py win the name lookup
    if str(folder) not in sys.path:
        sys.path.append(str(folder))


def load_from(folder: Path, module: str):
    """Import a same-named module (e.g. visualizer) from an earlier day by file path."""
    path = folder / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"{folder.name.replace('-', '_')}_{module}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_kickoff(module: str):
    return load_from(KICKOFF, module)
