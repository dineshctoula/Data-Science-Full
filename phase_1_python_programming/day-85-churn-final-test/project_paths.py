"""Point Day 85 at the earlier project days.

Day 81: data, frozen split, metric. Day 82: the champion logistic.
Day 84: the call-list economics and policies. Their own modules run
`import project_paths`, which inside this day resolves to this file, so
every earlier folder they lean on goes on the path here.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KICKOFF = HERE.parent / "day-81-churn-kickoff"
FIRST_MODEL = HERE.parent / "day-82-churn-first-model"
CALL_LIST = HERE.parent / "day-84-churn-call-list"
DATA_DIR = KICKOFF / "data"
SPLIT_FILE = KICKOFF / "split_ids.tsv"

for folder in (KICKOFF, FIRST_MODEL, CALL_LIST):
    if not folder.is_dir():
        raise ImportError(f"expected the earlier project day at {folder}")
    # appended, so this day's visualizer.py and main.py win the name lookup
    if str(folder) not in sys.path:
        sys.path.append(str(folder))


def load_from(folder: Path, module: str):
    """Import a same-named module (e.g. an earlier day's visualizer) by file path."""
    path = folder / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"{folder.name.replace('-', '_')}_{module}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
