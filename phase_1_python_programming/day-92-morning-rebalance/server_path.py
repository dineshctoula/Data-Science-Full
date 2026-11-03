"""Put Day 91 on the import path, then load *its* project_paths.

Day 91's fit does `import project_paths` and expects that module to add
the Day 86 grid and the Day 87 model. This folder is first on sys.path
when the walkthrough or the tests run. A project_paths.py here would
bind that import, and the grid would not be found. The bootstrap lives
under a different name so the name `project_paths` stays Day 91's.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

API = Path(__file__).resolve().parent.parent / "day-91-bike-forecast-api"

if not API.is_dir():
    raise ImportError(f"expected the forecast server at {API}")
if str(API) not in sys.path:
    sys.path.append(str(API))

# A leftover binding (an earlier draft of this day, or another copy)
# has to go before Day 91's module is loaded under this name.
current = sys.modules.get("project_paths")
current_file = getattr(current, "__file__", None)
if current is not None and (current_file is None or Path(current_file).resolve().parent != API):
    del sys.modules["project_paths"]

importlib.import_module("project_paths")
