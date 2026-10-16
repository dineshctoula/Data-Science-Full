"""Run the Day 76 Adam walkthrough."""

from adam_engine import adam, gradient_descent, momentum
from visualizer import AdamPlots


def _print_run(run):
    print(f"\n   {run.name}")
    print(f"   first step: flat {run.first_step[0]:.4f}, steep {run.first_step[1]:.4f}")
    print(f"   {'step':>6}  {'loss':>10}  {'flat':>8}  {'steep':>8}")
    for step in (1, 10, 20, 40, 80):
        i = step - 1
        print(
            f"   {step:6d}  {run.losses[i]:10.4f}  "
            f"{run.weights[i, 0]:8.3f}  {run.weights[i, 1]:8.3f}"
        )


def run_pipeline():
    print("=" * 72)
    print("DAY 76: ADAM")
    print("=" * 72)

    print("\n1) same bowl as Day 75: 0.5 * (flat^2 + 18 * steep^2), start (2, 1)")
    print("   GD and momentum use lr 0.05. Adam uses lr 0.1 — different knob.")

    runs = [
        gradient_descent(lr=0.05, steps=80),
        momentum(lr=0.05, beta=0.85, steps=80),
        adam(lr=0.1, steps=80),
    ]

    print("\n2) three optimizers")
    for run in runs:
        _print_run(run)

    plots = AdamPlots()
    p1 = plots.loss_curves(runs, title="Day 76 — loss on the lopsided bowl")
    p2 = plots.first_step_bars(runs, title="Day 76 — Adam equalizes the first step")

    print("\n3) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 76 done.")


if __name__ == "__main__":
    run_pipeline()
