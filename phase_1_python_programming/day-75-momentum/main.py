"""Run the Day 75 momentum walkthrough."""

from mom_engine import descend
from visualizer import MomPlots


def _print_run(run):
    print(f"\n   beta {run.beta:g}")
    print(f"   {'step':>6}  {'loss':>10}  {'flat':>8}  {'steep':>8}")
    for step in (1, 10, 20, 40, 60, 80):
        i = step - 1
        print(
            f"   {step:6d}  {run.losses[i]:10.4f}  "
            f"{run.weights[i, 0]:8.3f}  {run.weights[i, 1]:8.3f}"
        )


def run_pipeline():
    print("=" * 72)
    print("DAY 75: MOMENTUM")
    print("=" * 72)

    print("\n1) loss = 0.5 * (flat^2 + 18 * steep^2), start at (2, 1)")
    print("   learning rate 0.05, 80 steps")
    print("   velocity is an average of the gradient, then we step that")

    runs = [descend(beta=beta, steps=80) for beta in (0.0, 0.85, 0.9)]

    print("\n2) three betas, same step size")
    for run in runs:
        _print_run(run)

    plots = MomPlots()
    p1 = plots.loss_curves(runs, title="same bowl, three amounts of momentum")
    p2 = plots.flat_axis(runs, title="the flat axis is where momentum earns its keep")

    print("\n3) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 75 done.")


if __name__ == "__main__":
    run_pipeline()
