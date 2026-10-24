"""Run the Day 83 boosting day: tune by CV inside train, then one head-to-head on validation."""

import numpy as np

from project_paths import DATA_DIR, HERE, SPLIT_FILE, load_kickoff
from baseline_engine import TOP_FRAC, RateTable, add_tenure_band, log_loss, recall_at_top, roc_auc
from boost_engine import BoostedTrees
from churn_data import TARGET, load_clean
from cv_engine import Config, cv_curves, cv_logistic, encode, pick_config
from features import VARIANTS, apply_steps
from select_engine import paired_bootstrap, run_grid
from split_engine import load_split
from visualizer import BoostPlots

KickoffPlots = load_kickoff("visualizer").KickoffPlots

CONFIGS = [Config(depth, lr) for depth in (1, 2, 3, 4) for lr in (0.05, 0.1)]
MAX_TREES = 500


def run_pipeline():
    print("=" * 72)
    print("DAY 83: GRADIENT-BOOSTED TREES VS THE LOGISTIC")
    print("=" * 72)

    parts = load_split(load_clean(DATA_DIR), SPLIT_FILE)
    train, val = parts["train"], parts["validation"]
    y_val = val[TARGET].to_numpy()
    print(f"\n1) frozen Day 81 split: train {len(train)}, validation {len(val)}; test stays closed")

    print(f"\n2) 5-fold CV inside train: {len(CONFIGS)} settings × up to {MAX_TREES} trees (about a minute)")
    curves = cv_curves(train, CONFIGS, max_trees=MAX_TREES, k=5, seed=83)
    print(f"   {'setting':<20} {'best trees':>10}  {'CV log loss':>11}  {'fold sd':>7}  {'at 500':>7}")
    for cfg, c in curves.items():
        mean = c.mean(axis=0)
        at = int(mean.argmin())
        print(f"   {cfg.label():<20} {at + 1:10d}  {mean[at]:11.4f}  {c[:, at].std():7.4f}  {mean[-1]:7.4f}")
    logit_cv = cv_logistic(train, k=5, seed=83)
    print(f"   {'Day 82 logistic':<20} {'—':>10}  {logit_cv.mean():11.4f}  {logit_cv.std():7.4f}")

    pick = pick_config(curves, tol=0.001)
    print("\n3) pick: lowest CV log loss, ties within 0.001 go to shallower trees, then fewer trees")
    print(f"   best {pick.best_loss:.4f}; picked {pick.config.label()}, {pick.n_trees} trees ({pick.cv_loss:.4f})")

    # refit on all 4930 training rows with the picked setting; validation is opened here
    X_tr, X_va, enc = encode(train, val)
    booster = BoostedTrees(
        n_trees=pick.n_trees, learning_rate=pick.config.learning_rate, max_depth=pick.config.max_depth
    ).fit(X_tr, train[TARGET].to_numpy())
    p_boost = booster.predict_proba(X_va)
    logit = run_grid(train, val, variants=["collapsed, + log tenure"], lams=[1e-3])[0]
    p_logit = logit.scores(val)
    base_p = RateTable(["Contract", "tenure_band", "InternetService"]).fit(add_tenure_band(train)).predict(add_tenure_band(val))

    print("\n4) validation, once")
    print(f"   {'model':<34} {'log loss':>8}  {'AUC':>6}  {'top20':>6}")
    for name, p in (("Day 81 lookup", base_p), ("Day 82 logistic", p_logit), (f"boosted stumps ({pick.n_trees})", p_boost)):
        print(f"   {name:<34} {log_loss(y_val, p):8.4f}  {roc_auc(y_val, p):6.3f}  {recall_at_top(y_val, p):6.3f}")

    top = paired_bootstrap(y_val, p_boost, p_logit, n_boot=2000, seed=83)
    auc = paired_bootstrap(y_val, p_boost, p_logit, metric=roc_auc, n_boot=500, seed=83)
    neg_loss = lambda y, p: -log_loss(y, p)  # noqa: E731  (higher is better, like the other two)
    loss = paired_bootstrap(y_val, p_boost, p_logit, metric=neg_loss, n_boot=500, seed=83)
    print("\n5) booster minus logistic (paired bootstrap on validation rows)")
    for name, b in (("top-20% recall", top), ("AUC", auc), ("-log loss", loss)):
        print(f"   {name:<15} {b['diff']:+.4f}   95% [{b['lo']:+.4f}, {b['hi']:+.4f}]   above zero {b['share_above_zero']:.0%}")
    agree = np.corrcoef(p_boost, p_logit)[0, 1]
    print(f"   the two models' probabilities correlate at {agree:.3f}")

    names = enc.names
    shapes = []
    if pick.config.max_depth == 1:
        boost_parts, _ = booster.additive_parts(X_va)
        logit_enc = logit.encoder
        steps, _ = VARIANTS[logit.variant]
        X_logit = logit_enc.transform(apply_steps(val, steps))
        # the logistic's tenure curve is its tenure term plus its log-tenure term
        coef = dict(zip(logit_enc.names, logit.model.coef))
        col = dict(zip(logit_enc.names, range(len(logit_enc.names))))
        tenure_logit = coef["tenure"] * X_logit[:, col["tenure"]] + coef["log_tenure"] * X_logit[:, col["log_tenure"]]
        charge_logit = coef["MonthlyCharges"] * X_logit[:, col["MonthlyCharges"]]
        shapes = [
            ("tenure (months)", val["tenure"].to_numpy(), {"stumps": boost_parts[:, names.index("tenure")], "logistic": tenure_logit}),
            ("MonthlyCharges", val["MonthlyCharges"].to_numpy(), {"stumps": boost_parts[:, names.index("MonthlyCharges")], "logistic": charge_logit}),
            ("TotalCharges", val["TotalCharges"].to_numpy(), {"stumps": boost_parts[:, names.index("TotalCharges")]}),
        ]

    gains = booster.gain_by_column(len(names))
    print("\n6) where the booster spent its splits (share of total gain)")
    for j in np.argsort(-gains)[:5]:
        print(f"   {names[j]:<34} {gains[j] / gains.sum():.1%}")

    out = HERE / "output"
    plots = BoostPlots(out)
    paths = [
        plots.cv_curves(curves, pick, logit_cv),
        KickoffPlots(out).gains(
            y_val,
            {"Day 81 lookup": base_p, "Day 82 logistic": p_logit, "boosted stumps": p_boost, "perfect list": y_val.astype(float)},
            frac=TOP_FRAC,
            title="validation gains: lookup, logistic, boosted stumps",
        ),
        plots.gain_importance(names, gains),
    ]
    if shapes:
        paths.append(plots.shapes(shapes))
    print("\n7) plots saved")
    for p in paths:
        print("  ", p.relative_to(HERE))
    print("\nDay 83 done.")


if __name__ == "__main__":
    run_pipeline()
