"""Run the Day 67 learning-curve walkthrough."""

from curve_engine import learning_curve, make_sine
from visualizer import CurvePlots


def _print_curve(curve):
    print(f"\n   degree {curve.degree}")
    print(f"   {'n':>5}  {'train':>8}  {'test':>8}  {'gap':>8}")
    for size, train, test, gap in zip(curve.sizes, curve.train_mse, curve.test_mse, curve.gap()):
        print(f"   {size:5d}  {train:8.3f}  {test:8.3f}  {gap:8.3f}")


def run_pipeline():
    print("=" * 72)
    print("DAY 67: LEARNING CURVES")
    print("=" * 72)

    x, y = make_sine(n=500, noise=0.4, seed=67)
    sizes = [40, 80, 160, 320]
    print(f"\n1) {len(x)} rows of sin(1.4 x) plus noise 0.4")
    print("   100 rows stay as the test slice the whole time")
    print("   each point is the mean of 16 random training subsets")

    line = learning_curve(x, y, degree=1, sizes=sizes, n_test=100, repeats=16, seed=67)
    bend = learning_curve(x, y, degree=4, sizes=sizes, n_test=100, repeats=16, seed=67)

    print("\n2) mean squared error")
    _print_curve(line)
    _print_curve(bend)

    plots = CurvePlots()
    p1 = plots.two_curves(line, bend, title="seed 67 — does more data help?")
    p2 = plots.gap_lines([line, bend], title="how far the test error sits above the train error")

    print("\n3) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 67 done.")


if __name__ == "__main__":
    run_pipeline()
