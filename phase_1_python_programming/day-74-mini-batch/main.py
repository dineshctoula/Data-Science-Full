"""Run the Day 74 mini-batch walkthrough."""

from batch_engine import fit_batches, make_line
from visualizer import BatchPlots


def _print_run(run):
    print(f"\n   batch {run.batch_size} — {run.updates_per_epoch} update(s) per epoch")
    print(f"   {'epoch':>6}  {'train':>8}  {'test':>8}")
    for epoch in (1, 2, 5, 10, 20):
        i = epoch - 1
        print(f"   {epoch:6d}  {run.train_mse[i]:8.3f}  {run.test_mse[i]:8.3f}")


def run_pipeline():
    print("=" * 72)
    print("DAY 74: MINI-BATCH GRADIENT DESCENT")
    print("=" * 72)

    X, y = make_line(n=240, seed=74)
    print(f"\n1) table {X.shape}, same line as the learning-rate day, lr fixed at 0.2")
    print("   20 epochs. An epoch is one shuffle of the training rows.")

    runs = [
        fit_batches(X, y, batch_size=10**6, lr=0.2, epochs=20, seed=74),
        fit_batches(X, y, batch_size=32, lr=0.2, epochs=20, seed=74),
        fit_batches(X, y, batch_size=1, lr=0.2, epochs=20, seed=74),
    ]

    print("\n2) test error after each pass")
    for run in runs:
        _print_run(run)

    plots = BatchPlots()
    p1 = plots.test_curves(runs, title="seed 74 — smaller batches step more often")
    p2 = plots.end_bars(runs, title="seed 74 — after 1 pass and after 20")

    print("\n3) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 74 done.")


if __name__ == "__main__":
    run_pipeline()
