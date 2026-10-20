"""Run the Day 80 phase-4 wrap-up."""

import numpy as np

from wrap_engine import (
    Prep,
    log_loss,
    make_churn_table,
    permutation_importance,
    pick_decay,
    pick_threshold,
    roc_auc,
    scores_at,
    search_decay,
    split_three,
)
from visualizer import WrapPlots

DECAYS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3]


def _print_scores(label, s):
    print(
        f"   {label:<22} acc={s['accuracy']:.3f}  precision={s['precision']:.3f}  "
        f"recall={s['recall']:.3f}  f1={s['f1']:.3f}"
    )


def run_pipeline():
    print("=" * 72)
    print("DAY 80: PHASE 4 WRAP-UP")
    print("=" * 72)

    table, y = make_churn_table(n=900, seed=80, missing_rate=0.10)
    (tr, y_tr), (va, y_va), (te, y_te) = split_three(table, y, fracs=(0.6, 0.2, 0.2), seed=80)
    print("\n1) 900 customers, split 60/20/20 before anything else")
    print(f"   rows: train {len(y_tr)}, validation {len(y_va)}, test {len(y_te)}")
    print(f"   churn rate: train {y_tr.mean():.3f}, validation {y_va.mean():.3f}, test {y_te.mean():.3f}")
    spend = table["spend"]
    print(
        f"   spend is blank for {np.isnan(spend).mean():.1%} of rows: "
        f"{np.isnan(spend[y == 1]).mean():.1%} of leavers, {np.isnan(spend[y == 0]).mean():.1%} of stayers"
    )

    # every statistic the preprocessing needs comes from the 540 training rows
    prep = Prep().fit(tr)
    X_tr, X_va, X_te = prep.transform(tr), prep.transform(va), prep.transform(te)
    print("\n2) preprocessing fit on the training rows only")
    print(f"   spend median used for blanks: {prep.medians['spend']:.2f}")
    print(f"   {len(prep.names)} model columns: {', '.join(prep.names)}")

    trials = search_decay(X_tr, y_tr, X_va, y_va, DECAYS)
    picked = pick_decay(trials)
    print("\n3) logistic, mini-batch Adam (batch 32, lr 0.005), early stopping (patience 25)")
    print(f"   {'decay':>6}  {'kept epoch':>10}  {'train':>7}  {'val':>7}")
    for t in trials:
        print(f"   {t.decay:6g}  {t.best_epoch:10d}  {t.train_loss:7.4f}  {t.val_loss:7.4f}")
    print(f"   validation picks decay {picked.decay:g}")

    model = picked.model
    p_va = model.predict_proba(X_va)
    threshold, val_f1 = pick_threshold(y_va, p_va)
    print(f"\n4) threshold picked on validation F1: {threshold:g} (val F1 {val_f1:.3f})")

    # the test slice is opened here and nowhere earlier
    p_te = model.predict_proba(X_te)
    base_p = np.full(len(y_te), y_tr.mean())
    print("\n5) test slice, scored once")
    print(f"   log loss {log_loss(y_te, p_te):.4f}  vs  {log_loss(y_te, base_p):.4f} for 'always the train rate'")
    print(f"   AUC {roc_auc(y_te, p_te):.3f}")
    _print_scores("always 'stays'", scores_at(y_te, np.zeros(len(y_te))))
    _print_scores("model, threshold 0.5", scores_at(y_te, p_te, 0.5))
    _print_scores(f"model, threshold {threshold:g}", scores_at(y_te, p_te, threshold))
    no_decay = trials[0].model.predict_proba(X_te)
    print(f"   decay 0 for comparison: test log loss {log_loss(y_te, no_decay):.4f}")

    base_loss, rises = permutation_importance(model, prep, va, y_va, n_repeats=10, seed=80)
    print(f"\n6) permutation importance on validation (base log loss {base_loss:.4f})")
    for col, rise in sorted(rises.items(), key=lambda kv: -kv[1]):
        print(f"   {col:<7} +{rise:.4f}" if rise >= 0 else f"   {col:<7} {rise:.4f}")

    plots = WrapPlots()
    p1 = plots.loss_curves(model.history, title=f"decay {picked.decay:g} — training stops on validation")
    p2 = plots.decay_grid(trials, picked)
    p3 = plots.importance(rises)

    print("\n7) plots saved")
    for p in (p1, p2, p3):
        print("  ", p)
    print("\nDay 80 done. Phase 4 closes here.")


if __name__ == "__main__":
    run_pipeline()
