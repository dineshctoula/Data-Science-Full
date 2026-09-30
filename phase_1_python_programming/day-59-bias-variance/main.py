"""Run the Day 59 bias-variance walkthrough."""

from bv_engine import decompose_degrees, sample_fit_curves
from visualizer import BiasVariancePlots


def run_pipeline():
    print("=" * 72)
    print("DAY 59: BIAS-VARIANCE TRADEOFF")
    print("=" * 72)

    degrees = [1, 2, 3, 5, 8]
    print("\n1) refit each polynomial on 40 fresh samples (n=35, noise=0.45)")
    points = decompose_degrees(degrees, n_train=35, n_runs=40, noise=0.45, seed=59)
    for p in points:
        print("  ", p.summary())

    best = min(points, key=lambda p: p.mse)
    print(f"\n   lowest mse around degree {best.degree}")

    print("\n2) a stiff line vs a degree-8 wiggle, same noise")
    grid, truth, low = sample_fit_curves(degree=1, n_curves=15, n_train=35, noise=0.45, seed=59)
    _, _, high = sample_fit_curves(degree=8, n_curves=15, n_train=35, noise=0.45, seed=59)
    # average, over x, of how much the curves disagree with each other
    print("   degree 1 disagreement:", round(float(low.std(axis=0).mean()), 3))
    print("   degree 8 disagreement:", round(float(high.std(axis=0).mean()), 3))

    plots = BiasVariancePlots()
    p1 = plots.tradeoff(points)
    p2 = plots.spaghetti(grid, truth, low, title="degree 1 — biased, stable", filename="spaghetti_deg1.png")
    p3 = plots.spaghetti(grid, truth, high, title="degree 8 — flexible, jumpy", filename="spaghetti_deg8.png")

    print("\n3) plots saved")
    for p in (p1, p2, p3):
        print("  ", p)
    print("\nDay 59 done.")


if __name__ == "__main__":
    run_pipeline()
