"""Day 83 — tune the booster with cross-validation on the training rows only.

Day 82's logistic had one knob (the penalty) and barely cared about it.
The booster has depth, learning rate, and tree count, and tree count alone
has hundreds of settings. Tuning all of that on the 1056 validation rows
would leave them too worn to judge the booster against the logistic
fairly. So the knobs are set by 5-fold CV inside train, and validation
is used once, for the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import project_paths  # noqa: F401
from baseline_engine import log_loss
from boost_engine import BoostedTrees, sigmoid
from churn_data import TARGET
from features import Encoder, apply_steps, collapse_mirrors

# trees split on order, so log tenure would repeat tenure; the mirrors still go
TREE_STEPS = (collapse_mirrors,)
TREE_NUMERIC = ("tenure", "MonthlyCharges", "TotalCharges")


@dataclass(frozen=True)
class Config:
    max_depth: int
    learning_rate: float

    def label(self) -> str:
        return f"depth {self.max_depth}, lr {self.learning_rate:g}"


def stratified_folds(y, k: int = 5, seed: int = 83):
    """k index arrays that partition the rows, each with about the same churn rate."""
    y = np.asarray(y).reshape(-1)
    if k < 2:
        raise ValueError("k should be at least 2")
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        # deal the rows of each class out round-robin, like cards
        for i, chunk in enumerate(np.array_split(idx, k)):
            folds[i].extend(chunk.tolist())
    return [np.sort(np.array(f, dtype=np.int64)) for f in folds]


def encode(train_df, other_df):
    """Tree columns for one train / held-out pair, with the encoder fit on the train part."""
    tr, ot = apply_steps(train_df, TREE_STEPS), apply_steps(other_df, TREE_STEPS)
    enc = Encoder(TREE_NUMERIC).fit(tr)
    return enc.transform(tr), enc.transform(ot), enc


def cv_curves(train_df, configs, max_trees: int = 500, k: int = 5, seed: int = 83, **boost_kwargs):
    """Held-out log loss after every tree count, for each config and fold.

    Returns {config: array of shape (k, max_trees)}. One fit per fold and
    config covers every tree count, because staged_decision replays the
    ensemble one tree at a time.
    """
    y = train_df[TARGET].to_numpy()
    folds = stratified_folds(y, k=k, seed=seed)
    curves = {cfg: np.empty((k, max_trees)) for cfg in configs}
    for f, held in enumerate(folds):
        fit_rows = np.setdiff1d(np.arange(len(y)), held)
        X_fit, X_held, _ = encode(train_df.iloc[fit_rows], train_df.iloc[held])
        for cfg in configs:
            model = BoostedTrees(
                n_trees=max_trees, learning_rate=cfg.learning_rate, max_depth=cfg.max_depth, **boost_kwargs
            ).fit(X_fit, y[fit_rows])
            staged = sigmoid(model.staged_decision(X_held))
            curves[cfg][f] = [log_loss(y[held], p) for p in staged]
    return curves


def cv_logistic(train_df, variant: str = "collapsed, + log tenure", lam: float = 1e-3, k: int = 5, seed: int = 83):
    """Day 82's picked logistic on the very same folds, one log loss per fold.

    Without this, the booster's CV number has nothing to stand next to:
    CV losses and validation losses come from different rows and aren't
    comparable with each other.
    """
    from features import VARIANTS
    from logit_engine import NewtonLogistic

    steps, numeric = VARIANTS[variant]
    y = train_df[TARGET].to_numpy()
    losses = []
    for held in stratified_folds(y, k=k, seed=seed):
        fit_rows = np.setdiff1d(np.arange(len(y)), held)
        fit_df = apply_steps(train_df.iloc[fit_rows], steps)
        held_df = apply_steps(train_df.iloc[held], steps)
        enc = Encoder(numeric).fit(fit_df)
        model = NewtonLogistic(lam=lam).fit(enc.transform(fit_df), y[fit_rows])
        losses.append(log_loss(y[held], model.predict_proba(enc.transform(held_df))))
    return np.array(losses)


@dataclass
class Pick:
    config: Config
    n_trees: int
    cv_loss: float
    best_loss: float


def pick_config(curves, tol: float = 0.001) -> Pick:
    """Lowest mean CV log loss; anything within `tol` ties, and ties go simplest.

    Simplest means shallower trees first, then fewer trees. Shallow trees
    can only combine a few columns at once, so they are the smaller model
    even when they need more rounds.
    """
    if not curves:
        raise ValueError("no curves")
    means = {cfg: c.mean(axis=0) for cfg, c in curves.items()}
    best = min(float(m.min()) for m in means.values())
    candidates = []
    for cfg, m in means.items():
        within = np.flatnonzero(m <= best + tol)
        if len(within):
            # within one config, the fewest trees that already reach the tie band
            n = int(within[0]) + 1
            candidates.append((cfg.max_depth, n, cfg, float(m[n - 1])))
    depth, n, cfg, loss = min(candidates, key=lambda c: (c[0], c[1]))
    return Pick(cfg, n, loss, best)
