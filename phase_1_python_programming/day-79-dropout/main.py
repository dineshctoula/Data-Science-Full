"""Run the Day 79 dropout walkthrough."""

import numpy as np

from drop_engine import TinyNet, make_curve, pick_by_val, sweep
from visualizer import DropPlots

DROPS = [0.0, 0.1, 0.2, 0.3, 0.5]


def _run(n_val):
    data = make_curve(n_train=16, n_val=n_val, n_test=300, noise=0.35, seed=79)
    trials = sweep(data, DROPS, n_seeds=5)
    return data, trials, pick_by_val(trials)


def _print_table(trials):
    print(f"   {'drop':>5}  {'train':>7}  {'val':>7}  {'test':>7}  {'worst test':>10}")
    for t in trials:
        print(f"   {t.drop:5g}  {t.train_mse:7.3f}  {t.val_mse:7.3f}  {t.test_mse:7.3f}  {t.worst_test:10.3f}")


def run_pipeline():
    print("=" * 72)
    print("DAY 79: DROPOUT")
    print("=" * 72)

    print("\n1) sin(1.6 x) plus noise 0.35, 16 training points, 300 test points")
    print("   one tanh layer of 64 units, lr 0.04, 8000 full-batch steps")
    print("   every number is a mean over 5 starting seeds")
    print("   noise alone puts the best possible MSE near 0.35^2 = 0.123")

    # the training points come out of the generator first, so both runs
    # train on the exact same 16 rows; only the held-out draws differ
    small_data, small_trials, small_pick = _run(n_val=40)
    print("\n2) 40 validation rows")
    _print_table(small_trials)
    print(f"   validation picks dropout {small_pick.drop:g}")

    big_data, big_trials, big_pick = _run(n_val=200)
    print("\n3) same 16 training rows, 200 validation rows")
    _print_table(big_trials)
    print(f"   validation picks dropout {big_pick.drop:g}")
    plain = big_trials[0]
    print(f"   test MSE {big_pick.test_mse:.3f} vs {plain.test_mse:.3f} with no dropout")

    (x_tr, y_tr), _, _ = big_data
    grid = np.linspace(-2.0, 2.0, 300)
    curves = []
    for drop in (0.0, big_pick.drop):
        net = TinyNet(drop=drop, seed=0).fit(x_tr, y_tr)
        curves.append((f"dropout {drop:g}", grid, net.predict(grid)))

    plots = DropPlots()
    p1 = plots.fits(x_tr, y_tr, curves, title="seed 0 — with and without dropout")
    p2 = plots.mse_bars(big_trials, title="200 validation rows — error by dropout rate")

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 79 done.")


if __name__ == "__main__":
    run_pipeline()
