"""Run the Day 77 learning-rate schedule walkthrough."""

from sched_engine import constant_lr, cosine_lr, fit_schedule, make_line, step_decay_lr
from visualizer import SchedPlots


def _print_run(run):
    print(f"\n   {run.name}")
    print(f"   {'epoch':>6}  {'lr':>8}  {'train':>8}  {'test':>8}")
    for epoch in (1, 5, 15, 20, 30, 45, 60):
        i = epoch - 1
        print(
            f"   {epoch:6d}  {run.lrs[i]:8.4f}  "
            f"{run.train_mse[i]:8.3f}  {run.test_mse[i]:8.3f}"
        )


def run_pipeline():
    print("=" * 72)
    print("DAY 77: LEARNING-RATE SCHEDULES")
    print("=" * 72)

    X, y = make_line(n=240, seed=77)
    print(f"\n1) table {X.shape}, y = 3 x0 - 1.5 x1 plus noise")
    print("   start every run at learning rate 1.8, 60 epochs")

    runs = [
        fit_schedule(X, y, constant_lr, "constant", base_lr=1.8, epochs=60, seed=77),
        fit_schedule(X, y, step_decay_lr, "step decay", base_lr=1.8, epochs=60, seed=77),
        fit_schedule(X, y, cosine_lr, "cosine", base_lr=1.8, epochs=60, seed=77),
    ]

    print("\n2) same split, three schedules")
    for run in runs:
        _print_run(run)

    plots = SchedPlots()
    p1 = plots.lr_curves(runs, title="seed 77 — the rate itself")
    p2 = plots.test_curves(runs, title="seed 77 — test MSE while the rate cools")

    print("\n3) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 77 done.")


if __name__ == "__main__":
    run_pipeline()
