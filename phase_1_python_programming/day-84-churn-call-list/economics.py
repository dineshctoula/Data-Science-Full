"""Day 84 — what a retention call is worth, written down as assumptions.

None of these numbers come from the data. They are placeholders a
retention team would replace with its own, and every result downstream
is shown for a range of save rates because that one is the least known.

For a customer with churn probability p and monthly bill m:

    value if kept  = m * months_kept * margin
    expected gain  = p * save_rate * value       (only a leaver can be saved)
    cost           = call_cost + credit          (paid for everyone called)
    expected profit = expected gain - cost

Calling is worth it when expected profit > 0, i.e. when
p > cost / (save_rate * value). That cut-off differs per customer: a
customer with a bigger bill is worth calling at a lower risk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Economics:
    call_cost: float = 10.0  # agent time for one call
    credit: float = 20.0  # goodwill credit offered on every call, leaver or not
    save_rate: float = 0.25  # share of called leavers who stay because of the call
    months_kept: float = 12.0  # how long a saved customer is assumed to stay
    margin: float = 0.5  # share of the bill that is profit

    def __post_init__(self):
        if self.call_cost < 0 or self.credit < 0:
            raise ValueError("costs should be zero or more")
        if not 0 < self.save_rate <= 1:
            raise ValueError("save_rate should be in (0, 1]")
        if self.months_kept <= 0 or not 0 < self.margin <= 1:
            raise ValueError("months_kept should be positive and margin in (0, 1]")

    @property
    def cost(self) -> float:
        return self.call_cost + self.credit

    def with_save_rate(self, save_rate: float) -> "Economics":
        return replace(self, save_rate=save_rate)

    def value(self, df: pd.DataFrame) -> np.ndarray:
        """Margin kept over months_kept if this customer is saved."""
        return df["MonthlyCharges"].to_numpy(dtype=float) * self.months_kept * self.margin

    def expected_profit(self, p, df: pd.DataFrame) -> np.ndarray:
        """What calling each customer is worth before we know who leaves."""
        p = np.asarray(p, dtype=float).reshape(-1)
        if len(p) != len(df):
            raise ValueError("one probability per customer")
        return p * self.save_rate * self.value(df) - self.cost

    def break_even(self, df: pd.DataFrame) -> np.ndarray:
        """The churn probability above which a call pays for itself, per customer."""
        return self.cost / (self.save_rate * self.value(df))

    def realized_profit(self, y, df: pd.DataFrame) -> np.ndarray:
        """What calling each customer turned out to be worth, once we know who left.

        A real save is still a coin flip at save_rate, so this is the
        average over that flip, not one draw of it: y * save_rate * value - cost.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != len(df):
            raise ValueError("one outcome per customer")
        return y * self.save_rate * self.value(df) - self.cost
