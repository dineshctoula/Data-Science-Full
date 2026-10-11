"""Run the Day 69 calibration walkthrough."""

from cal_engine import holdout_compare, make_labels
from visualizer import CalPlots


def _line(score):
    return (
        f"   {score.name:6} acc={score.accuracy:.3f}  "
        f"brier={score.brier:.3f}  ece={score.ece:.3f}"
    )


def run_pipeline():
    print("=" * 72)
    print("DAY 69: PROBABILITY CALIBRATION")
    print("=" * 72)

    X, y = make_labels(n=600, seed=69)
    report = holdout_compare(X, y, temperature=0.5, seed=69)
    coef = report["platt_coef"]

    print(f"\n1) {X.shape[0]} rows, class-1 rate {y.mean():.3f}")
    print(f"   train {report['n_train']}, val {report['n_val']}, test {report['n_test']}")
    print("   temperature 0.5 divides the logit, which makes probabilities sharper")

    print("\n2) test slice")
    for name in ("raw", "sharp", "platt"):
        print(_line(report[name]))
    print(f"   platt intercept={coef[0]:.3f}  slope={coef[1]:.3f}  (fit on validation only)")

    print("\n3) sharpened bins on the test slice (mean prob, actual frequency, count)")
    for item in report["sharp"].bins:
        print(f"   {item.mean_prob:.3f}  freq={item.freq:.3f}  n={item.count}")

    plots = CalPlots()
    order = [report["raw"], report["sharp"], report["platt"]]
    p1 = plots.reliability(order, title="seed 69 — reliability on the test slice")
    p2 = plots.brier_bars(
        [score.name for score in order],
        [score.brier for score in order],
        title="test Brier after sharpening and after the validation fix",
    )

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 69 done.")


if __name__ == "__main__":
    run_pipeline()
