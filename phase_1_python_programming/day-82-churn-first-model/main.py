"""Run the Day 82 first churn model."""

import numpy as np

from project_paths import DATA_DIR, HERE, SPLIT_FILE, load_kickoff
from baseline_engine import TOP_FRAC, RateTable, add_tenure_band, recall_at_top, roc_auc
from churn_data import TARGET, load_clean
from select_engine import paired_bootstrap, pick_simplest, run_grid
from split_engine import load_split
from visualizer import ModelPlots

KickoffPlots = load_kickoff("visualizer").KickoffPlots

BASELINE_KEYS = ["Contract", "tenure_band", "InternetService"]


def run_pipeline():
    print("=" * 72)
    print("DAY 82: FIRST CHURN MODEL")
    print("=" * 72)

    parts = load_split(load_clean(DATA_DIR), SPLIT_FILE)
    train, val = parts["train"], parts["validation"]
    y_val = val[TARGET].to_numpy()
    print(f"\n1) frozen Day 81 split: train {len(train)}, validation {len(val)}; test stays closed")

    # the bar from Day 81, rebuilt exactly the same way
    base_p = RateTable(BASELINE_KEYS).fit(add_tenure_band(train)).predict(add_tenure_band(val))
    print(f"   bar to beat (contract × tenure band × internet lookup): top-20% {recall_at_top(y_val, base_p):.3f}")

    trials = run_grid(train, val)
    print("\n2) logistic (Newton, L2) — every feature set × every penalty, scored on validation")
    print(f"   {'feature set':<30} {'lam':>6} {'cols':>4}  {'log loss':>8}  {'AUC':>6}  {'top20':>6}")
    for t in trials:
        print(f"   {t.variant:<30} {t.lam:6g} {t.n_columns:4d}  {t.val_log_loss:8.4f}  {t.val_auc:6.3f}  {t.val_top20:6.3f}")

    picked = pick_simplest(trials, tol=0.001)
    best_loss = min(t.val_log_loss for t in trials)
    print("\n3) pick: lowest validation log loss, ties within 0.001 go to fewer columns, then more penalty")
    print(f"   best log loss {best_loss:.4f}; picked '{picked.variant}', lam {picked.lam:g} ({picked.val_log_loss:.4f})")
    print(f"   {picked.model.n_iter} Newton steps")

    p_val = picked.scores(val)
    boot = paired_bootstrap(y_val, p_val, base_p, n_boot=2000, seed=82)
    auc_boot = paired_bootstrap(y_val, p_val, base_p, metric=roc_auc, n_boot=500, seed=82)
    print("\n4) is the win over the lookup real? (paired bootstrap on validation rows)")
    print(
        f"   top-20% recall {picked.val_top20:.3f} vs {recall_at_top(y_val, base_p):.3f}: "
        f"diff {boot['diff']:+.3f}, 95% [{boot['lo']:+.3f}, {boot['hi']:+.3f}], "
        f"above zero in {boot['share_above_zero']:.1%} of resamples"
    )
    leavers = int(y_val.sum())
    print(f"   that is about {boot['diff'] * leavers:.0f} more of the {leavers} validation leavers on the call list")
    print(
        f"   AUC {picked.val_auc:.3f} vs {roc_auc(y_val, base_p):.3f}: "
        f"diff {auc_boot['diff']:+.3f}, 95% [{auc_boot['lo']:+.3f}, {auc_boot['hi']:+.3f}]"
    )

    enc = picked.encoder
    # coef / std turns a per-std coefficient back into "switch this level on vs the reference"
    raw = picked.model.coef / enc.std
    print("\n5) what the model leans on (one-hot levels, vs the most common level)")
    onehot = [(n, r) for n, r in zip(enc.names, raw) if "=" in n]
    for name, r in sorted(onehot, key=lambda nr: -abs(nr[1]))[:6]:
        col = name.split("=")[0]
        print(f"   {name:<34} odds × {np.exp(r):4.2f}   (vs {col}={enc.reference(col)})")

    calibration_gap = abs(float(p_val.mean()) - float(y_val.mean()))
    print(f"\n6) mean predicted churn {p_val.mean():.3f} vs actual {y_val.mean():.3f} (gap {calibration_gap:.3f})")

    out = HERE / "output"
    gains = KickoffPlots(out).gains(
        y_val,
        {
            "contract × tenure × internet lookup": base_p,
            f"logistic, {picked.variant}": p_val,
            "perfect list": y_val.astype(float),
        },
        frac=TOP_FRAC,
        title="validation gains: first model vs the Day 81 bar",
    )
    plots = ModelPlots(out)
    paths = [
        gains,
        plots.calibration(y_val, p_val),
        plots.coefficients(enc.names, picked.model.coef),
        plots.bootstrap_diff(boot["samples"], boot["lo"], boot["hi"]),
    ]
    print("\n7) plots saved")
    for p in paths:
        print("  ", p.relative_to(HERE))
    print(f"\nDay 82 done. New bar on validation: top-20% {picked.val_top20:.3f}, log loss {picked.val_log_loss:.4f}.")


if __name__ == "__main__":
    run_pipeline()
