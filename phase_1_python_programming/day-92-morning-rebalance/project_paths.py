"""Point Day 92 at the Day 91 forecast server instead of copying it.

The client speaks HTTP. The walkthrough still starts that server
in-process, because nothing here assumes a leftover process is already
listening on port 8765, and this day should not take that port from
someone following the Day 91 notes.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent / "day-91-bike-forecast-api"

if not API.is_dir():
    raise ImportError(f"expected the forecast server at {API}")
# appended, so this day's modules win any name they share
if str(API) not in sys.path:
    sys.path.append(str(API))
