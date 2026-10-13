"""Run the Day 73 learning-rate walkthrough."""

from rate_engine import fit_rate, make_line
from visualizer import RatePlots


def _print_run(run):
    print(f"\n   lr {run.lr:g}")
    print(f"   {'epoch':>6}  {'train':>12}  {'test':>12}")
    for epoch in (1, 5, 10, 20, 40):
        i = epoch - 1
        print(f"   {epoch:6d}  {run.train_mse[i]:12.3f}  {run.test_mse[i]:12.3f}")


def run_pipeline():
    print("=" * 72)
    print("DAY 73: LEARNING RATE")
    print("=" * 72)

    X, y = make_line(n=240, seed=73)
    print(f"\n1) table {X.shape}, y = 3 x0 - 1.5 x1 plus noise")
    print("   columns are already standard scale, 40 epochs each")

    rates = (0.02, 0.2, 1.9)
    runs = [fit_rate(X, y, lr=lr, epochs=40, seed=73) for lr in rates]

    print("\n2) same split, three step sizes")
    for run in runs:
        _print_run(run)

    plots = RatePlots()
    p1 = plots.train_curves(runs, title="seed 73 — train MSE by step size")
    p2 = plots.final_bars(runs, title="seed 73 — test MSE at epoch 40")

    print("\n3) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 73 done.")


if __name__ == "__main__":
    run_pipeline()
