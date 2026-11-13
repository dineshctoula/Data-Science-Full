"""Day 93 — seven staging rows, scored one morning at a time.

The week is the same seven calendar days Day 92 asked for. A date
inside the test quarter is still present so a server that had started
answering October cannot hide behind a local skip. Those rows are not
joined to actuals. They stay blank, and they stay out of the MAE.

MAE is the mean absolute morning error over scored days only. Adding
a refused day as a zero error would credit the model for a morning it
did not forecast.

Shortfall is bikes the rush needed that were not staged
(`actual - staged` when that is positive). Surplus is the other
direction. They add up to the sum of absolute errors, so a small MAE
can still be a week that was always slightly short.
"""

from __future__ import annotations

import paths  # noqa: F401

from actuals import ClosedQuarter
from score import score_row
from week import CLOSED_FROM, closed_quarter_is_blank

_COLUMNS = (
    ("date", 12, False),
    ("weekday", 11, False),
    ("staged", 8, False),
    ("actual", 8, False),
    ("error", 7, True),
    ("extra", 7, True),
    ("base_err", 9, False),
)


def score_week(rows, actuals) -> list[dict]:
    """Score each open day. A closed date must already be a blank staging row.

    `closed_quarter_is_blank` is the Day 92 check. Running it here means
    a served October row fails before `score_row` would also refuse it.
    """
    leaked = closed_quarter_is_blank(rows)
    if leaked:
        raise ClosedQuarter(f"the test quarter was staged: {', '.join(leaked)}")
    scored = []
    for row in rows:
        if row["date"] >= CLOSED_FROM:
            # not score_row: that raises, and the walkthrough needs the blank
            # line so a reader can see October was asked for and not judged
            scored.append(
                {
                    "date": row["date"],
                    "weekday": row.get("weekday"),
                    "kind": row.get("kind"),
                    "status": "unscored",
                    "reason": row.get("reason") or "closed test quarter",
                    "morning_staged": None,
                    "morning_actual": None,
                    "morning_error": None,
                    "abs_error": None,
                    "morning_baseline": None,
                    "baseline_abs_error": None,
                    "morning_extra": None,
                    "slice": None,
                }
            )
            continue
        scored.append(score_row(row, actuals))
    return scored


def summarize(rows) -> dict:
    """Totals over scored days only. Refused mornings are not zeros."""
    scored = [row for row in rows if row["status"] == "scored"]
    n = len(scored)
    abs_errors = [row["abs_error"] for row in scored]
    baseline_abs = [row["baseline_abs_error"] for row in scored]
    shortfall = sum(max(0, -row["morning_error"]) for row in scored)
    surplus = sum(max(0, row["morning_error"]) for row in scored)
    return {
        "n_days": len(rows),
        "n_scored": n,
        "n_unscored": len(rows) - n,
        "morning_staged": sum(row["morning_staged"] for row in scored),
        "morning_actual": sum(row["morning_actual"] for row in scored),
        "morning_extra": sum(row["morning_extra"] for row in scored),
        "mae": (sum(abs_errors) / n) if n else None,
        "baseline_mae": (sum(baseline_abs) / n) if n else None,
        "shortfall": shortfall,
        "surplus": surplus,
    }


def _cell(value, width: int, signed: bool) -> str:
    if value is None:
        text = "-"
    elif signed:
        text = f"{value:+d}"
    elif isinstance(value, float):
        text = f"{value:.1f}"
    else:
        text = str(value)
    return f"{text:<{width}}" if width > 8 else f"{text:>{width}}"


def render(rows) -> str:
    """One line per day. A closed morning is a dash, not a zero error."""
    header = ""
    names = {
        "date": "date",
        "weekday": "weekday",
        "staged": "staged",
        "actual": "actual",
        "error": "error",
        "extra": "extra",
        "base_err": "base err",
    }
    for key, width, _signed in _COLUMNS:
        piece = names[key]
        header += f"{piece:<{width}}" if width > 8 else f"{piece:>{width}}"
    lines = [header]
    for row in rows:
        values = {
            "date": row["date"],
            "weekday": row["weekday"],
            "staged": row["morning_staged"],
            "actual": row["morning_actual"],
            "error": row["morning_error"],
            "extra": row["morning_extra"],
            "base_err": row["baseline_abs_error"],
        }
        line = ""
        for key, width, signed in _COLUMNS:
            line += _cell(values[key], width, signed)
        lines.append(line)
        if row["status"] == "unscored" and row.get("reason"):
            lines.append(f"{'':12}{row['reason']}")
    totals = summarize(rows)
    if totals["n_scored"]:
        mae = totals["mae"]
        base = totals["baseline_mae"]
        lines.append(
            f"scored {totals['n_scored']} of {totals['n_days']}; "
            f"morning MAE {mae:.1f} "
            f"(baseline {base:.1f}). "
            f"Staged {totals['morning_staged']} against {totals['morning_actual']} actual. "
            f"Short {totals['shortfall']}, surplus {totals['surplus']}. "
            "Unscored days are blank, not zero error."
        )
    else:
        lines.append("no scored mornings; unscored days are blank, not zero error.")
    return "\n".join(lines)
