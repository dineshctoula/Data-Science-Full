"""Run the Day 52 gradient boosting walkthrough."""

from boost_engine import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    make_regression_wave,
    make_two_blobs_clf,
    train_test_split,
)
from visualizer import BoostPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 52: GRADIENT BOOSTING")
    print("=" * 72)

    X, y = make_regression_wave(n=140, noise=0.3, seed=52)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.3, seed=52)

    print("\n1) regress the sine-ish wave")
    reg = GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=2).fit(Xtr, ytr)
    print("   train MSE:", round(reg.score(Xtr, ytr).mse, 4))
    print("   test  MSE:", round(reg.score(Xte, yte).mse, 4))
    print("   first→last train loss:", round(reg.train_loss_[0], 4), "→", round(reg.train_loss_[-1], 4))

    print("\n2) learning-rate shootout (same n_estimators)")
    curves = {}
    for lr in (0.05, 0.1, 0.3):
        m = GradientBoostingRegressor(n_estimators=40, learning_rate=lr, max_depth=2).fit(Xtr, ytr)
        curves[f"lr={lr}"] = m.train_loss_
        print(f"   lr={lr}: final train MSE={m.train_loss_[-1]:.4f}")

    print("\n3) binary blobs via L2 boost + 0.5 threshold")
    Xc, yc = make_two_blobs_clf(n_per=55, seed=52)
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(Xc, yc, test_frac=0.3, seed=53)
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.15, max_depth=2).fit(Xtr_c, ytr_c)
    print("  ", clf.score(Xte_c, yte_c).summary())

    plots = BoostPlots()
    p1 = plots.loss_curve(reg.train_loss_, title="wave: train MSE per round")
    p2 = plots.regression_fit(reg, X, y, title="boosted fit on noisy sine")
    p3 = plots.lr_compare(curves, title="learning rate effect")
    p4 = plots.decision_map(clf, Xc, yc, title="boosted blob regions")

    print("\n4) plots saved")
    for p in (p1, p2, p3, p4):
        print("  ", p)
    print("\nDay 52 done.")


if __name__ == "__main__":
    run_pipeline()
