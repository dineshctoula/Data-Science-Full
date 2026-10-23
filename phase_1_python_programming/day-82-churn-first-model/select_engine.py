"""Day 82 — choose a model on validation, then ask whether the win is real.

Selection uses validation log loss, not the headline top-20% recall.
With 280 validation leavers, top-20% recall moves in steps of 1/280 and
jumps around from one setting to the next; log loss uses every row's
probability and is far steadier. The headline is reported, not chased.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import project_paths  # noqa: F401
from baseline_engine import log_loss, recall_at_top, roc_auc
from churn_data import TARGET
from features import VARIANTS, Encoder, apply_steps
from logit_engine import NewtonLogistic

# no lam = 0: the "as-is" variant has exactly repeated columns (the mirrored
# levels), and with no penalty at all its Hessian is singular
LAMS = (1e-4, 1e-3, 1e-2, 1e-1)


@dataclass
class Trial:
    variant: str
    lam: float
    n_columns: int
    val_log_loss: float
    val_auc: float
    val_top20: float
    model: NewtonLogistic
    encoder: Encoder

    def scores(self, df):
        """Probabilities for any slice, run through this trial's own prep."""
        steps, _ = VARIANTS[self.variant]
        return self.model.predict_proba(self.encoder.transform(apply_steps(df, steps)))


def run_grid(train, val, variants=None, lams=LAMS):
    """Every variant × every penalty, fit on train, scored on validation."""
    variants = list(VARIANTS) if variants is None else list(variants)
    if not variants or not lams:
        raise ValueError("need at least one variant and one lam")
    trials = []
    for name in variants:
        steps, numeric = VARIANTS[name]
        tr, va = apply_steps(train, steps), apply_steps(val, steps)
        # the encoder is fit once per variant, on train only
        encoder = Encoder(numeric).fit(tr)
        X_tr, X_va = encoder.transform(tr), encoder.transform(va)
        y_va = va[TARGET].to_numpy()
        for lam in lams:
            model = NewtonLogistic(lam=lam).fit(X_tr, tr[TARGET])
            p = model.predict_proba(X_va)
            trials.append(
                Trial(name, float(lam), X_tr.shape[1], log_loss(y_va, p), roc_auc(y_va, p), recall_at_top(y_va, p), model, encoder)
            )
    return trials


def pick_simplest(trials, tol: float = 0.001) -> Trial:
    """Lowest validation log loss, but treat anything within `tol` as a tie.

    Among the tied trials, take the fewest columns, then the strongest
    penalty. A gain smaller than `tol` isn't worth a bigger model.
    """
    if not trials:
        raise ValueError("no trials")
    if tol < 0:
        raise ValueError("tol should be zero or more")
    best = min(t.val_log_loss for t in trials)
    tied = [t for t in trials if t.val_log_loss <= best + tol]
    return min(tied, key=lambda t: (t.n_columns, -t.lam, t.val_log_loss))


def paired_bootstrap(y, score_a, score_b, metric=recall_at_top, n_boot: int = 2000, seed: int = 82):
    """Interval for metric(a) - metric(b) on the same resampled rows.

    Resampling the rows once and scoring both lists on the same draw is
    what makes it paired: a lucky draw lifts both, so the difference is
    much steadier than either number alone.
    """
    if n_boot < 100:
        raise ValueError("n_boot should be at least 100")
    y = np.asarray(y).reshape(-1)
    a = np.asarray(score_a, dtype=float).reshape(-1)
    b = np.asarray(score_b, dtype=float).reshape(-1)
    if not len(y) == len(a) == len(b):
        raise ValueError("y and both scores should be the same length")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        take = rng.integers(0, len(y), size=len(y))
        diffs[i] = metric(y[take], a[take]) - metric(y[take], b[take])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "diff": float(metric(y, a) - metric(y, b)),
        "lo": float(lo),
        "hi": float(hi),
        "share_above_zero": float(np.mean(diffs > 0)),
        "samples": diffs,
    }
